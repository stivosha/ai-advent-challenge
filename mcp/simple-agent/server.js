import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";
import { dirname, join, resolve } from "path";

// ─── Config ─────────────────────────────────────────────────────────────────
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const __dir = dirname(fileURLToPath(import.meta.url));

// Path to the simple-agent directory (can be overridden via env)
const AGENT_DIR = process.env.AGENT_DIR
  ? resolve(process.env.AGENT_DIR)
  : resolve(__dir, "../../agents/simple-agent");

const HISTORY_FILE = join(AGENT_DIR, "history.json");
const MEMORY_FILE  = join(AGENT_DIR, "memory.json");

const GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
const GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions";

const CONFIG = {
  model:       "GigaChat",
  system:      "Ты — полезный AI-ассистент. Отвечай чётко и по делу.",
  temperature: 0.7,
  max_tokens:  1024,
  top_p:       0.9,
  keepRecent:  6,
};

const PROFILE_FIELDS = {
  name:        "Имя пользователя",
  role:        "Роль / должность",
  domain:      "Область работы (например: backend, ML, DevOps)",
  style:       "Стиль общения: formal | casual | technical",
  format:      "Формат ответов: short | detailed | bullets | code-first",
  language:    "Язык ответов (например: ru, en)",
  constraints: "Что НЕ нужно делать (ограничения)",
  goals:       "Текущий контекст / цели проекта",
};

// ─── Long-term memory (Layer 3) ─────────────────────────────────────────────
function defaultLongTerm() {
  return { profile: {}, knowledge: {}, decisions: [], updatedAt: null };
}

function loadLongTerm() {
  try {
    if (!existsSync(MEMORY_FILE)) return defaultLongTerm();
    const raw = JSON.parse(readFileSync(MEMORY_FILE, "utf8"));
    const lt = defaultLongTerm();
    Object.assign(lt, raw);
    if (!lt.profile)   lt.profile   = {};
    if (!lt.knowledge) lt.knowledge = {};
    if (!lt.decisions) lt.decisions = [];
    return lt;
  } catch {
    return defaultLongTerm();
  }
}

function saveLongTerm(lt) {
  lt.updatedAt = new Date().toISOString();
  writeFileSync(MEMORY_FILE, JSON.stringify(lt, null, 2), "utf8");
}

function longTermToText(lt) {
  const lines = [];
  const profileEntries = Object.entries(lt.profile ?? {});
  if (profileEntries.length) {
    lines.push("[Профиль пользователя]");
    profileEntries.forEach(([k, v]) => lines.push(`  ${k}: ${v}`));
  }
  const knowledgeEntries = Object.entries(lt.knowledge ?? {});
  if (knowledgeEntries.length) {
    lines.push("[Накопленные знания]");
    knowledgeEntries.forEach(([k, v]) => lines.push(`  ${k}: ${v}`));
  }
  if (lt.decisions?.length) {
    lines.push("[Решения и договорённости]");
    lt.decisions.forEach(d => {
      const when = d.when ? ` (${d.when})` : "";
      const why  = d.why  ? ` — ${d.why}`  : "";
      lines.push(`  • ${d.what}${why}${when}`);
    });
  }
  return lines.join("\n");
}

function buildPersonalizedSystem(lt) {
  const p = lt.profile ?? {};
  let system = CONFIG.system;
  const instructions = [];

  if (p.name)   instructions.push(`Пользователя зовут ${p.name}.`);
  if (p.role)   instructions.push(`Его роль: ${p.role}.`);
  if (p.domain) instructions.push(`Область работы: ${p.domain}.`);

  const styleMap = {
    formal:    "Общайся формально и профессионально.",
    casual:    "Общайся неформально и дружелюбно.",
    technical: "Предпочитай технические термины, избегай чрезмерных упрощений.",
  };
  if (p.style && styleMap[p.style]) instructions.push(styleMap[p.style]);

  const formatMap = {
    short:        "Давай максимально краткие ответы — только суть.",
    detailed:     "Давай подробные, развёрнутые ответы с объяснениями.",
    bullets:      "Структурируй ответы списками и пунктами.",
    "code-first": "При любой возможности давай рабочий код, текста — минимум.",
  };
  if (p.format && formatMap[p.format]) instructions.push(formatMap[p.format]);

  if (p.language && p.language !== "ru") instructions.push(`Отвечай на языке: ${p.language}.`);
  if (p.constraints) instructions.push(`Ограничения: ${p.constraints}.`);
  if (p.goals)       instructions.push(`Контекст проекта: ${p.goals}.`);

  if (instructions.length) {
    system += "\n\n[Персонализация]\n" + instructions.join("\n");
  }
  return system;
}

// ─── Working memory (Layer 2) ────────────────────────────────────────────────
function defaultWorking() {
  return { task: null, vars: {}, notes: [] };
}

function workingToText(w) {
  if (!w) return "";
  const lines = [];
  if (w.task) lines.push(`Задача: ${w.task}`);
  const varEntries = Object.entries(w.vars ?? {});
  if (varEntries.length) {
    lines.push("Переменные:");
    varEntries.forEach(([k, v]) => lines.push(`  ${k} = ${v}`));
  }
  if (w.notes?.length) {
    lines.push("Заметки:");
    w.notes.forEach(n => lines.push(`  • ${n}`));
  }
  return lines.join("\n");
}

// ─── Session (Short-term, Layer 1) ──────────────────────────────────────────
function defaultSession() {
  return {
    strategy:      "sliding-window",
    messages:      [],
    summary:       null,
    facts:         {},
    branches:      { main: { messages: [], summary: null } },
    checkpoints:   {},
    currentBranch: "main",
    turn:          0,
    working:       defaultWorking(),
    totalTokens:   0,
    requests:      [],
  };
}

function loadSession() {
  try {
    const raw = JSON.parse(readFileSync(HISTORY_FILE, "utf8"));
    if (Array.isArray(raw)) {
      const s = defaultSession();
      s.messages = raw;
      s.branches.main.messages = [...raw];
      return s;
    }
    const s = defaultSession();
    Object.assign(s, raw);
    if (!s.facts)         s.facts = {};
    if (!s.branches)      s.branches = { main: { messages: s.messages ?? [], summary: s.summary ?? null } };
    if (!s.checkpoints)   s.checkpoints = {};
    if (!s.currentBranch) s.currentBranch = "main";
    if (!s.strategy)      s.strategy = "sliding-window";
    if (!s.working)       s.working = defaultWorking();
    if (s.turn == null)   s.turn = 0;
    return s;
  } catch {
    return defaultSession();
  }
}

function saveSession(session) {
  writeFileSync(HISTORY_FILE, JSON.stringify(session, null, 2), "utf8");
}

function getActiveBranch(session) {
  const b = session.branches[session.currentBranch];
  if (!b) throw new Error(`Ветка "${session.currentBranch}" не найдена`);
  return b;
}

function makeMessage(role, content, turn) {
  return { role, content, turn, at: new Date().toISOString() };
}

// ─── GigaChat API ────────────────────────────────────────────────────────────
let cachedToken    = null;
let tokenExpiresAt = 0;

async function getAccessToken() {
  if (cachedToken && Date.now() / 1000 < tokenExpiresAt - 60) return cachedToken;
  const apiKey = process.env.GIGACHAT_API_KEY;
  if (!apiKey) throw new Error("GIGACHAT_API_KEY environment variable is not set");

  const res = await fetch(GIGACHAT_AUTH_URL, {
    method: "POST",
    headers: {
      "Authorization": `Basic ${apiKey}`,
      "RqUID":         randomUUID(),
      "Content-Type":  "application/x-www-form-urlencoded",
    },
    body: "scope=GIGACHAT_API_PERS",
  });
  if (!res.ok) throw new Error(`GigaChat auth failed: ${res.status} ${await res.text()}`);
  const data     = await res.json();
  cachedToken    = data.access_token;
  tokenExpiresAt = data.expires_at;
  return cachedToken;
}

async function callGigaChat(messages, { temperature = 0.3, max_tokens = 512 } = {}) {
  const token = await getAccessToken();
  const res = await fetch(GIGACHAT_CHAT_URL, {
    method: "POST",
    headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: CONFIG.model, messages, temperature, max_tokens }),
  });
  if (!res.ok) throw new Error(`GigaChat error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.choices[0].message.content;
}

async function summarizeMessages(messages, existingSummary) {
  const dialog = messages
    .map(m => `${m.role === "user" ? "Пользователь" : "Ассистент"}: ${m.content}`)
    .join("\n\n");
  const userContent = existingSummary
    ? `Обнови краткое резюме диалога, добавив новые сообщения.\n\nТекущее резюме:\n${existingSummary}\n\nНовые сообщения:\n${dialog}`
    : `Создай краткое резюме следующего диалога:\n\n${dialog}`;
  return callGigaChat([
    { role: "system", content: "Ты — помощник, создающий краткие резюме диалогов." },
    { role: "user",   content: userContent },
  ]);
}

async function extractFacts(messages, existingFacts) {
  const dialog = messages
    .slice(-4)
    .map(m => `${m.role === "user" ? "Пользователь" : "Ассистент"}: ${m.content}`)
    .join("\n\n");
  const prompt = `Обнови JSON-объект с ключевыми фактами из диалога.\n\nТекущие факты:\n${JSON.stringify(existingFacts, null, 2)}\n\nНовые сообщения:\n${dialog}\n\nВерни ТОЛЬКО валидный JSON.`;
  try {
    const text = await callGigaChat(
      [
        { role: "system", content: "Ты — помощник для извлечения фактов. Отвечай только валидным JSON." },
        { role: "user",   content: prompt },
      ],
      { temperature: 0.1, max_tokens: 300 },
    );
    const match = text.match(/\{[\s\S]*\}/);
    return match ? JSON.parse(match[0]) : existingFacts;
  } catch {
    return existingFacts;
  }
}

async function autoExtractLongTerm(messages, existingLongTerm) {
  const dialog = messages
    .slice(-20)
    .map(m => `${m.role === "user" ? "Пользователь" : "Ассистент"}: ${m.content}`)
    .join("\n\n");
  const prompt = `Проанализируй диалог и обнови JSON долгосрочной памяти.\n\nТекущая память:\n${JSON.stringify({ profile: existingLongTerm.profile, knowledge: existingLongTerm.knowledge, decisions: existingLongTerm.decisions }, null, 2)}\n\nДиалог:\n${dialog}\n\nВерни ТОЛЬКО валидный JSON с полями profile, knowledge, decisions.`;
  try {
    const text = await callGigaChat(
      [
        { role: "system", content: "Ты — помощник для управления долгосрочной памятью. Отвечай только валидным JSON." },
        { role: "user",   content: prompt },
      ],
      { temperature: 0.1, max_tokens: 600 },
    );
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) return existingLongTerm;
    const extracted = JSON.parse(match[0]);
    return {
      ...existingLongTerm,
      profile:   { ...existingLongTerm.profile,   ...(extracted.profile   ?? {}) },
      knowledge: { ...existingLongTerm.knowledge, ...(extracted.knowledge ?? {}) },
      decisions: [
        ...(existingLongTerm.decisions ?? []),
        ...(extracted.decisions ?? []).filter(d =>
          !existingLongTerm.decisions.some(e => e.what === d.what)
        ),
      ],
    };
  } catch {
    return existingLongTerm;
  }
}

// ─── Build context for API call ──────────────────────────────────────────────
function buildContext(session, longTerm) {
  const { strategy } = session;
  const parts = [buildPersonalizedSystem(longTerm)];

  const ltText = longTermToText(longTerm);
  if (ltText) parts.push("\n[Долгосрочная память]\n" + ltText);

  const wmText = workingToText(session.working);
  if (wmText) parts.push("\n[Рабочая память — текущая задача]\n" + wmText);

  if (strategy === "sliding-window") {
    const systemContent = parts.join("\n");
    const recent = session.messages.slice(-CONFIG.keepRecent);
    return [{ role: "system", content: systemContent }, ...recent];
  }

  if (strategy === "sticky-facts") {
    const entries = Object.entries(session.facts ?? {});
    if (entries.length) {
      parts.push("\n[Ключевые факты диалога]\n" + entries.map(([k, v]) => `• ${k}: ${v}`).join("\n"));
    }
    const systemContent = parts.join("\n");
    const recent = session.messages.slice(-CONFIG.keepRecent);
    return [{ role: "system", content: systemContent }, ...recent];
  }

  if (strategy === "branching") {
    const branch = getActiveBranch(session);
    if (branch.summary) {
      parts.push(`\n[Контекст ветки «${session.currentBranch}»]\n${branch.summary}`);
    }
    const systemContent = parts.join("\n");
    return [{ role: "system", content: systemContent }, ...branch.messages];
  }

  throw new Error(`Unknown strategy: ${strategy}`);
}

async function postProcess(session) {
  const { strategy } = session;

  if (strategy === "sliding-window") {
    session.messages = session.messages.slice(-CONFIG.keepRecent);
    session.summary  = null;
    return;
  }

  if (strategy === "sticky-facts") {
    session.facts = await extractFacts(session.messages, session.facts ?? {});
    if (session.messages.length > CONFIG.keepRecent * 5) {
      session.messages = session.messages.slice(-CONFIG.keepRecent * 5);
    }
    return;
  }

  if (strategy === "branching") {
    const branch = getActiveBranch(session);
    if (branch.messages.length > CONFIG.keepRecent * 3) {
      const toCompress = branch.messages.slice(0, -CONFIG.keepRecent);
      const summary    = await summarizeMessages(toCompress, branch.summary ?? null);
      branch.messages  = branch.messages.slice(-CONFIG.keepRecent);
      branch.summary   = summary;
    }
    return;
  }
}

// ─── Core ask function ───────────────────────────────────────────────────────
async function ask(userQuery) {
  if (!userQuery?.trim()) throw new Error("Query must not be empty");

  const session  = loadSession();
  const longTerm = loadLongTerm();

  session.turn = (session.turn ?? 0) + 1;
  const turn = session.turn;

  if (session.strategy === "branching") {
    getActiveBranch(session).messages.push(makeMessage("user", userQuery, turn));
  } else {
    session.messages.push(makeMessage("user", userQuery, turn));
  }

  const contextMessages = buildContext(session, longTerm);
  const token  = await getAccessToken();

  const apiRes = await fetch(GIGACHAT_CHAT_URL, {
    method: "POST",
    headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model:       CONFIG.model,
      messages:    contextMessages,
      temperature: CONFIG.temperature,
      max_tokens:  CONFIG.max_tokens,
      top_p:       CONFIG.top_p,
      stream:      true,
    }),
  });

  if (!apiRes.ok) throw new Error(`GigaChat API error: ${apiRes.status} ${await apiRes.text()}`);

  const reader  = apiRes.body.getReader();
  const decoder = new TextDecoder();
  let buf    = "";
  let result = "";
  let usage  = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const raw = trimmed.slice(5).trim();
      if (raw === "[DONE]") continue;
      let chunk;
      try { chunk = JSON.parse(raw); } catch { continue; }
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) result += delta;
      if (chunk.usage) usage = chunk.usage;
    }
  }

  if (session.strategy === "branching") {
    getActiveBranch(session).messages.push(makeMessage("assistant", result, turn));
  } else {
    session.messages.push(makeMessage("assistant", result, turn));
  }

  await postProcess(session);

  session.requests.push({
    timestamp:        new Date().toISOString(),
    turn,
    promptTokens:     usage?.prompt_tokens     ?? null,
    completionTokens: usage?.completion_tokens ?? null,
    totalTokens:      usage?.total_tokens      ?? null,
  });
  if (usage?.total_tokens) session.totalTokens += usage.total_tokens;

  saveSession(session);
  return result;
}

// ─── MCP Server ──────────────────────────────────────────────────────────────
const server = new Server(
  { name: "simple-agent", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

const TOOLS = [
  // ── Core ──
  {
    name: "ask",
    description: "Отправить сообщение агенту (GigaChat) и получить ответ. Поддерживает многоходовой диалог с памятью.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Сообщение пользователя" },
      },
      required: ["query"],
    },
  },
  {
    name: "clear_session",
    description: "Очистить краткосрочную и рабочую память (история диалога сбрасывается). Долгосрочная память сохраняется.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "get_token_stats",
    description: "Получить статистику использования токенов (всего потрачено, последние запросы).",
    inputSchema: { type: "object", properties: {} },
  },

  // ── Session strategy ──
  {
    name: "set_strategy",
    description: "Изменить стратегию управления краткосрочной памятью.",
    inputSchema: {
      type: "object",
      properties: {
        strategy: {
          type: "string",
          enum: ["sliding-window", "sticky-facts", "branching"],
          description: "sliding-window — последние N сообщений; sticky-facts — ключевые факты; branching — ветки диалога",
        },
      },
      required: ["strategy"],
    },
  },

  // ── Working memory (Layer 2) ──
  {
    name: "wm_show",
    description: "Показать текущую рабочую память (задача, переменные, заметки).",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "wm_set_task",
    description: "Установить текущую задачу в рабочей памяти.",
    inputSchema: {
      type: "object",
      properties: {
        task: { type: "string", description: "Описание текущей задачи" },
      },
      required: ["task"],
    },
  },
  {
    name: "wm_set_var",
    description: "Сохранить переменную в рабочей памяти.",
    inputSchema: {
      type: "object",
      properties: {
        key:   { type: "string", description: "Имя переменной" },
        value: { type: "string", description: "Значение переменной" },
      },
      required: ["key", "value"],
    },
  },
  {
    name: "wm_add_note",
    description: "Добавить заметку в рабочую память.",
    inputSchema: {
      type: "object",
      properties: {
        note: { type: "string", description: "Текст заметки" },
      },
      required: ["note"],
    },
  },
  {
    name: "wm_clear",
    description: "Очистить рабочую память (задача, переменные, заметки).",
    inputSchema: { type: "object", properties: {} },
  },

  // ── Long-term memory (Layer 3) ──
  {
    name: "mem_list",
    description: "Показать всю долгосрочную память (профиль, знания, решения).",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "mem_set",
    description: "Сохранить факт в долгосрочной памяти.",
    inputSchema: {
      type: "object",
      properties: {
        section: { type: "string", enum: ["profile", "knowledge"], description: "Секция памяти" },
        key:     { type: "string", description: "Ключ" },
        value:   { type: "string", description: "Значение" },
      },
      required: ["section", "key", "value"],
    },
  },
  {
    name: "mem_delete",
    description: "Удалить ключ из долгосрочной памяти.",
    inputSchema: {
      type: "object",
      properties: {
        section: { type: "string", enum: ["profile", "knowledge"], description: "Секция памяти" },
        key:     { type: "string", description: "Ключ для удаления" },
      },
      required: ["section", "key"],
    },
  },
  {
    name: "mem_decide",
    description: "Зафиксировать решение в долгосрочной памяти.",
    inputSchema: {
      type: "object",
      properties: {
        what: { type: "string", description: "Описание решения" },
        why:  { type: "string", description: "Причина / обоснование (опционально)" },
      },
      required: ["what"],
    },
  },
  {
    name: "mem_auto",
    description: "Автоматически извлечь знания из текущего диалога и сохранить в долгосрочную память.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "mem_clear",
    description: "Очистить всю долгосрочную память.",
    inputSchema: { type: "object", properties: {} },
  },

  // ── Profile ──
  {
    name: "profile_show",
    description: "Показать профиль пользователя (имя, роль, стиль, формат ответов и т.д.).",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "profile_set",
    description: "Установить поле профиля пользователя.",
    inputSchema: {
      type: "object",
      properties: {
        field: {
          type: "string",
          enum: ["name", "role", "domain", "style", "format", "language", "constraints", "goals"],
          description: "Поле профиля",
        },
        value: { type: "string", description: "Значение поля" },
      },
      required: ["field", "value"],
    },
  },
  {
    name: "profile_delete",
    description: "Удалить поле из профиля пользователя.",
    inputSchema: {
      type: "object",
      properties: {
        field: {
          type: "string",
          enum: ["name", "role", "domain", "style", "format", "language", "constraints", "goals"],
        },
      },
      required: ["field"],
    },
  },

  // ── Branching strategy tools ──
  {
    name: "branch_list",
    description: "Показать список веток диалога (только для стратегии branching).",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "branch_create",
    description: "Создать новую ветку диалога (только для стратегии branching).",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Имя новой ветки" },
      },
      required: ["name"],
    },
  },
  {
    name: "branch_switch",
    description: "Переключиться на другую ветку диалога (только для стратегии branching).",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Имя ветки" },
      },
      required: ["name"],
    },
  },
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {

      // ── Core ──────────────────────────────────────────────────────────────

      case "ask": {
        const response = await ask(args.query);
        return { content: [{ type: "text", text: response }] };
      }

      case "clear_session": {
        saveSession(defaultSession());
        return { content: [{ type: "text", text: "Краткосрочная и рабочая память очищены. Долгосрочная память сохранена." }] };
      }

      case "get_token_stats": {
        const session = loadSession();
        const { totalTokens, requests } = session;
        const last5 = (requests ?? []).slice(-5);
        const text = [
          `Всего токенов: ${totalTokens ?? 0}`,
          `Запросов: ${(requests ?? []).length}`,
          last5.length ? `\nПоследние ${last5.length} запросов:` : "",
          ...last5.map(r =>
            `  [${r.turn ?? "?"}] prompt=${r.promptTokens ?? "?"} completion=${r.completionTokens ?? "?"} total=${r.totalTokens ?? "?"} at=${r.timestamp}`
          ),
        ].filter(Boolean).join("\n");
        return { content: [{ type: "text", text }] };
      }

      case "set_strategy": {
        const session = loadSession();
        const { strategy } = args;
        session.strategy = strategy;
        saveSession(session);
        return { content: [{ type: "text", text: `Стратегия установлена: ${strategy}` }] };
      }

      // ── Working memory ───────────────────────────────────────────────────

      case "wm_show": {
        const session = loadSession();
        const wmText = workingToText(session.working);
        return { content: [{ type: "text", text: wmText || "Рабочая память пуста." }] };
      }

      case "wm_set_task": {
        const session = loadSession();
        session.working.task = args.task;
        saveSession(session);
        return { content: [{ type: "text", text: `Задача установлена: ${args.task}` }] };
      }

      case "wm_set_var": {
        const session = loadSession();
        session.working.vars[args.key] = args.value;
        saveSession(session);
        return { content: [{ type: "text", text: `Переменная: ${args.key} = ${args.value}` }] };
      }

      case "wm_add_note": {
        const session = loadSession();
        session.working.notes.push(args.note);
        saveSession(session);
        return { content: [{ type: "text", text: `Заметка добавлена: ${args.note}` }] };
      }

      case "wm_clear": {
        const session = loadSession();
        session.working = defaultWorking();
        saveSession(session);
        return { content: [{ type: "text", text: "Рабочая память очищена." }] };
      }

      // ── Long-term memory ─────────────────────────────────────────────────

      case "mem_list": {
        const lt = loadLongTerm();
        const text = longTermToText(lt);
        const footer = lt.updatedAt ? `\nОбновлено: ${lt.updatedAt}` : "";
        return { content: [{ type: "text", text: text ? text + footer : "Долгосрочная память пуста." }] };
      }

      case "mem_set": {
        const { section, key, value } = args;
        const lt = loadLongTerm();
        lt[section][key] = value;
        saveLongTerm(lt);
        return { content: [{ type: "text", text: `Сохранено в ${section}: ${key} = ${value}` }] };
      }

      case "mem_delete": {
        const { section, key } = args;
        const lt = loadLongTerm();
        if (!lt[section] || !(key in lt[section])) {
          return { content: [{ type: "text", text: `Ключ "${key}" не найден в ${section}.` }], isError: true };
        }
        delete lt[section][key];
        saveLongTerm(lt);
        return { content: [{ type: "text", text: `Удалено из ${section}: ${key}` }] };
      }

      case "mem_decide": {
        const lt = loadLongTerm();
        lt.decisions.push({ what: args.what, why: args.why ?? null, when: new Date().toISOString() });
        saveLongTerm(lt);
        return { content: [{ type: "text", text: `Решение сохранено: ${args.what}${args.why ? ` (${args.why})` : ""}` }] };
      }

      case "mem_auto": {
        const session  = loadSession();
        const lt       = loadLongTerm();
        const msgs = session.strategy === "branching"
          ? getActiveBranch(session).messages
          : session.messages;
        if (!msgs.length) {
          return { content: [{ type: "text", text: "Нет диалога для анализа." }] };
        }
        const updated = await autoExtractLongTerm(msgs, lt);
        saveLongTerm(updated);
        const text = longTermToText(updated);
        return { content: [{ type: "text", text: `Долгосрочная память обновлена:\n\n${text || "(ничего не извлечено)"}` }] };
      }

      case "mem_clear": {
        saveLongTerm(defaultLongTerm());
        return { content: [{ type: "text", text: "Долгосрочная память очищена." }] };
      }

      // ── Profile ──────────────────────────────────────────────────────────

      case "profile_show": {
        const lt = loadLongTerm();
        const p  = lt.profile ?? {};
        const lines = ["[ Профиль пользователя ]\n"];
        Object.entries(PROFILE_FIELDS).forEach(([field, desc]) => {
          const val = p[field] ?? "(не задано)";
          lines.push(`  ${field.padEnd(12)} ${val}`);
          lines.push(`               └─ ${desc}`);
        });
        return { content: [{ type: "text", text: lines.join("\n") }] };
      }

      case "profile_set": {
        const { field, value } = args;
        if (!PROFILE_FIELDS[field]) {
          return { content: [{ type: "text", text: `Неизвестное поле "${field}".` }], isError: true };
        }
        if (field === "style" && !["formal", "casual", "technical"].includes(value)) {
          return { content: [{ type: "text", text: "style: допустимые значения — formal | casual | technical" }], isError: true };
        }
        if (field === "format" && !["short", "detailed", "bullets", "code-first"].includes(value)) {
          return { content: [{ type: "text", text: "format: допустимые значения — short | detailed | bullets | code-first" }], isError: true };
        }
        const lt = loadLongTerm();
        lt.profile[field] = value;
        saveLongTerm(lt);
        return { content: [{ type: "text", text: `Профиль обновлён: ${field} = ${value}` }] };
      }

      case "profile_delete": {
        const lt = loadLongTerm();
        if (!(args.field in (lt.profile ?? {}))) {
          return { content: [{ type: "text", text: `Поле "${args.field}" не задано.` }], isError: true };
        }
        delete lt.profile[args.field];
        saveLongTerm(lt);
        return { content: [{ type: "text", text: `Поле "${args.field}" удалено из профиля.` }] };
      }

      // ── Branching ────────────────────────────────────────────────────────

      case "branch_list": {
        const session = loadSession();
        const names = Object.keys(session.branches ?? {});
        const current = session.currentBranch;
        const text = names.map(n => (n === current ? `* ${n} (активная)` : `  ${n}`)).join("\n");
        return { content: [{ type: "text", text: text || "Нет веток." }] };
      }

      case "branch_create": {
        const session = loadSession();
        if (session.strategy !== "branching") {
          return { content: [{ type: "text", text: 'Ветки доступны только при стратегии "branching". Используйте set_strategy.' }], isError: true };
        }
        if (session.branches[args.name]) {
          return { content: [{ type: "text", text: `Ветка "${args.name}" уже существует.` }], isError: true };
        }
        session.branches[args.name] = { messages: [], summary: null };
        saveSession(session);
        return { content: [{ type: "text", text: `Ветка "${args.name}" создана.` }] };
      }

      case "branch_switch": {
        const session = loadSession();
        if (!session.branches[args.name]) {
          return { content: [{ type: "text", text: `Ветка "${args.name}" не найдена.` }], isError: true };
        }
        session.currentBranch = args.name;
        saveSession(session);
        return { content: [{ type: "text", text: `Переключено на ветку "${args.name}".` }] };
      }

      default:
        return { content: [{ type: "text", text: `Неизвестный инструмент: ${name}` }], isError: true };
    }
  } catch (err) {
    return { content: [{ type: "text", text: `Ошибка: ${err.message}` }], isError: true };
  }
});

// ─── Start ───────────────────────────────────────────────────────────────────
const transport = new StdioServerTransport();
await server.connect(transport);
