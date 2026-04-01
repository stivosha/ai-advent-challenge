import { randomUUID } from "crypto";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// GigaChat uses a certificate signed by Russian NCA — disable TLS verification for dev.
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

// ─── Config ────────────────────────────────────────────────────────────────────
const CONFIG = {
  model:       "GigaChat",
  system:      "Ты — полезный AI-ассистент. Отвечай чётко и по делу.",
  temperature: 0.7,
  max_tokens:  1024,
  top_p:       0.9,
  keepRecent:  6, // window size used by all strategies
};

// ─── User profile schema ────────────────────────────────────────────────────────
// Defines supported profile fields with human-readable descriptions.
// Stored in memory.json → profile (key-value).
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

// ─── File paths ────────────────────────────────────────────────────────────────
const __dir = dirname(fileURLToPath(import.meta.url));
const HISTORY_FILE = join(__dir, "history.json");
const MEMORY_FILE  = join(__dir, "memory.json");   // long-term memory (survives --clear)

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY LAYER 3 — ДОЛГОСРОЧНАЯ (long-term)
//
// Хранится в memory.json — отдельном файле, который НЕ сбрасывается при --clear.
// Содержит: профиль пользователя, накопленные знания, важные решения.
// Явно управляется командами: --mem set/delete/clear/list/auto
// ══════════════════════════════════════════════════════════════════════════════

function defaultLongTerm() {
  return {
    profile:   {},          // ключ-значение: имя, роль, предпочтения
    knowledge: {},          // ключ-значение: накопленные знания и факты
    decisions: [],          // массив: { what, why, when }
    updatedAt: null,
  };
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

// Build a compact text block for long-term memory (injected into system prompt)
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

// Build a personalized system prompt from the user profile.
// Translates profile fields into concrete behavioral instructions for the model.
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

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY LAYER 2 — РАБОЧАЯ (working)
//
// Хранится внутри history.json, но явно отделена от session.messages.
// Содержит: описание текущей задачи, переменные, заметки.
// Сбрасывается при --clear (вместе со всей сессией).
// Управляется командами: --wm task/set/note/show/clear
// ══════════════════════════════════════════════════════════════════════════════

function defaultWorking() {
  return {
    task:  null,            // строка: описание текущей задачи
    vars:  {},              // ключ-значение: переменные задачи
    notes: [],              // массив строк: произвольные заметки
  };
}

// Build a compact text block for working memory (injected into system prompt)
function workingToText(w) {
  if (!w) return "";
  const lines = [];

  if (w.task)          lines.push(`Задача: ${w.task}`);
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

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY LAYER 1 — КРАТКОСРОЧНАЯ (short-term)
//
// Хранится как session.messages — текущие сообщения диалога.
// Каждое сообщение имеет timestamp и порядковый номер хода (turn).
// Управляется стратегиями (sliding-window, sticky-facts, branching).
// Сбрасывается при --clear.
// ══════════════════════════════════════════════════════════════════════════════

function makeMessage(role, content, turn) {
  return { role, content, turn, at: new Date().toISOString() };
}

// ─── Persistent session (short-term + working memory + strategy state) ─────────
function defaultSession() {
  return {
    strategy:      "sliding-window",
    // ── short-term (layer 1) ──
    messages:      [],
    summary:       null,
    facts:         {},
    branches:      { main: { messages: [], summary: null } },
    checkpoints:   {},
    currentBranch: "main",
    turn:          0,         // global turn counter
    // ── working memory (layer 2) ──
    working:       defaultWorking(),
    // ── shared stats ──
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

// ─── Branch helpers (branching strategy) ───────────────────────────────────────
function getActiveBranch(session) {
  const b = session.branches[session.currentBranch];
  if (!b) throw new Error(`Ветка "${session.currentBranch}" не найдена`);
  return b;
}

// ─── API ───────────────────────────────────────────────────────────────────────
const GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
const GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions";

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

// Non-streaming helper used by summarize, extractFacts, autoSaveToLongTerm
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

// ─── Shared: Summarize messages ────────────────────────────────────────────────
async function summarizeMessages(messages, existingSummary) {
  const dialog = messages
    .map(m => `${m.role === "user" ? "Пользователь" : "Ассистент"}: ${m.content}`)
    .join("\n\n");

  const userContent = existingSummary
    ? `Обнови краткое резюме диалога, добавив новые сообщения. Сохрани все важные факты и контекст.\n\nТекущее резюме:\n${existingSummary}\n\nНовые сообщения:\n${dialog}`
    : `Создай краткое резюме следующего диалога, сохранив все важные факты и контекст:\n\n${dialog}`;

  return callGigaChat([
    { role: "system", content: "Ты — помощник, создающий краткие резюме диалогов." },
    { role: "user",   content: userContent },
  ]);
}

// Compress tail of messages: summarize old ones, keep last keepRecent
async function compressIfNeeded(messages, existingSummary) {
  if (messages.length <= CONFIG.keepRecent) return { messages, summary: existingSummary };
  const toCompress = messages.slice(0, -CONFIG.keepRecent);
  const summary    = await summarizeMessages(toCompress, existingSummary ?? null);
  return { messages: messages.slice(-CONFIG.keepRecent), summary };
}

// ─── Strategy 2: Extract / update key-value facts ──────────────────────────────
async function extractFacts(messages, existingFacts) {
  const dialog = messages
    .slice(-4)
    .map(m => `${m.role === "user" ? "Пользователь" : "Ассистент"}: ${m.content}`)
    .join("\n\n");

  const prompt = `Обнови JSON-объект с ключевыми фактами из диалога.

Текущие факты:
${JSON.stringify(existingFacts, null, 2)}

Новые сообщения:
${dialog}

Правила:
- Сохраняй только важные факты: цели, ограничения, предпочтения, решения, договорённости
- Обновляй изменившиеся факты, удаляй устаревшие
- Ключи — короткие метки на русском (например: "цель", "язык", "ограничения")
- Верни ТОЛЬКО валидный JSON, без объяснений`;

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

// ─── Long-term: Auto-extract from dialog ───────────────────────────────────────
// Called explicitly via --mem auto. Analyses the full current dialog and suggests
// what to save to long-term memory (profile facts, knowledge, decisions).
async function autoExtractLongTerm(messages, existingLongTerm) {
  const dialog = messages
    .slice(-20) // last 20 messages to avoid overloading context
    .map(m => `${m.role === "user" ? "Пользователь" : "Ассистент"}: ${m.content}`)
    .join("\n\n");

  const prompt = `Проанализируй диалог и обнови JSON долгосрочной памяти ассистента.

Текущая долгосрочная память:
${JSON.stringify({ profile: existingLongTerm.profile, knowledge: existingLongTerm.knowledge, decisions: existingLongTerm.decisions }, null, 2)}

Диалог:
${dialog}

Правила:
- profile: факты о пользователе (имя, роль, предпочтения, стиль работы)
- knowledge: важные знания, накопленные в диалоге (технические решения, договорённости)
- decisions: массив объектов { what, why, when } — только значимые решения

Обнови только те поля, в которых есть реально новая информация.
Верни ТОЛЬКО валидный JSON с полями profile, knowledge, decisions. Без объяснений.`;

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

// ══════════════════════════════════════════════════════════════════════════════
// BUILD CONTEXT
//
// Собирает финальный массив сообщений для API, внедряя все три уровня памяти:
//
//   [system]  CONFIG.system
//             + долгосрочная память (профиль, знания, решения)
//             + рабочая память (задача, переменные, заметки)
//             + стратегия-специфичный контент (факты, резюме ветки)
//   [user/assistant messages …]  ← краткосрочная память (последние N сообщений)
// ══════════════════════════════════════════════════════════════════════════════
function buildContext(session, longTerm) {
  const { strategy } = session;

  // Build layered system prompt (base + personalization from profile)
  const parts = [buildPersonalizedSystem(longTerm)];

  // Layer 3: long-term memory
  const ltText = longTermToText(longTerm);
  if (ltText) parts.push("\n[Долгосрочная память]\n" + ltText);

  // Layer 2: working memory
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

// ─── Post-process after getting the assistant response ─────────────────────────
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
      const { messages, summary } = await compressIfNeeded(branch.messages, branch.summary);
      branch.messages = messages;
      branch.summary  = summary;
    }
    return;
  }
}

// ─── Core: send query, return full response text ───────────────────────────────
export async function ask(userQuery) {
  if (!userQuery?.trim()) throw new Error("Query must not be empty");

  const session  = loadSession();
  const longTerm = loadLongTerm();

  session.turn = (session.turn ?? 0) + 1;
  const turn = session.turn;

  // Append user message (layer 1 — short-term)
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

  // Stream → collect full text + usage from last chunk
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

  // Append assistant response (layer 1 — short-term)
  if (session.strategy === "branching") {
    getActiveBranch(session).messages.push(makeMessage("assistant", result, turn));
  } else {
    session.messages.push(makeMessage("assistant", result, turn));
  }

  // Apply strategy-specific post-processing
  await postProcess(session);

  // Record token usage
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

// ─── Token stats helper ────────────────────────────────────────────────────────
export function getTokenStats() {
  const { totalTokens, requests } = loadSession();
  return { totalTokens, requests };
}

// ─── CLI entry point ───────────────────────────────────────────────────────────
if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replace(/\\/g, "/"))) {
  const args = process.argv.slice(2);

  // ── --clear  Reset short-term + working memory (long-term survives) ──────────
  if (args[0] === "--clear") {
    saveSession(defaultSession());
    console.log("Краткосрочная и рабочая память очищены.");
    console.log("Долгосрочная память сохранена. Для её сброса: --mem clear");
    process.exit(0);
  }

  // ── --mem  Long-term memory management (layer 3) ──────────────────────────────
  if (args[0] === "--mem") {
    const sub = args[1];
    const lt  = loadLongTerm();

    // --mem list  Show all long-term memory
    if (sub === "list" || sub === "show") {
      const ltText = longTermToText(lt);
      if (ltText) {
        console.log("Долгосрочная память:\n");
        console.log(ltText);
        if (lt.updatedAt) console.log(`\nОбновлено: ${lt.updatedAt}`);
      } else {
        console.log("Долгосрочная память пуста.");
      }
      process.exit(0);
    }

    // --mem set <section> <key> <value>  Save to long-term memory
    // section: profile | knowledge
    if (sub === "set") {
      const section = args[2];
      const key     = args[3];
      const value   = args.slice(4).join(" ");
      if (!section || !key || !value) {
        console.error("Использование: --mem set profile|knowledge <ключ> <значение>");
        process.exit(1);
      }
      if (section !== "profile" && section !== "knowledge") {
        console.error("Секции: profile, knowledge. Для решений используйте: --mem decide");
        process.exit(1);
      }
      lt[section][key] = value;
      saveLongTerm(lt);
      console.log(`Сохранено в ${section}: ${key} = ${value}`);
      process.exit(0);
    }

    // --mem decide <что> [--why <почему>]  Add a decision
    if (sub === "decide") {
      const what = args[2];
      if (!what) { console.error("Укажите описание решения."); process.exit(1); }
      const whyIdx = args.indexOf("--why");
      const why    = whyIdx !== -1 ? args.slice(whyIdx + 1).join(" ") : null;
      lt.decisions.push({ what, why, when: new Date().toISOString() });
      saveLongTerm(lt);
      console.log(`Решение сохранено: ${what}${why ? ` (${why})` : ""}`);
      process.exit(0);
    }

    // --mem delete <section> <key>  Remove a key from profile or knowledge
    if (sub === "delete") {
      const section = args[2];
      const key     = args[3];
      if (!section || !key) {
        console.error("Использование: --mem delete profile|knowledge <ключ>");
        process.exit(1);
      }
      if (!lt[section] || !(key in lt[section])) {
        console.error(`Ключ "${key}" не найден в ${section}.`);
        process.exit(1);
      }
      delete lt[section][key];
      saveLongTerm(lt);
      console.log(`Удалено из ${section}: ${key}`);
      process.exit(0);
    }

    // --mem clear  Wipe all long-term memory
    if (sub === "clear") {
      saveLongTerm(defaultLongTerm());
      console.log("Долгосрочная память очищена.");
      process.exit(0);
    }

    // --mem auto  Auto-extract facts from current dialog into long-term memory
    if (sub === "auto") {
      const session = loadSession();
      const msgs = session.strategy === "branching"
        ? getActiveBranch(session).messages
        : session.messages;
      if (!msgs.length) {
        console.log("Нет диалога для анализа.");
        process.exit(0);
      }
      console.log("Анализирую диалог для извлечения долгосрочных знаний…");
      autoExtractLongTerm(msgs, lt).then(updated => {
        saveLongTerm(updated);
        console.log("Долгосрочная память обновлена:\n");
        console.log(longTermToText(updated) || "(ничего не извлечено)");
      }).catch(err => { console.error(err.message); process.exit(1); });
    } else {
    console.error([
      "Управление долгосрочной памятью (сохраняется между сессиями):",
      "  --mem list                           Показать всю долгосрочную память",
      "  --mem set profile|knowledge <k> <v>  Сохранить факт вручную",
      "  --mem decide <что> [--why <почему>]  Зафиксировать решение",
      "  --mem delete profile|knowledge <k>   Удалить факт",
      "  --mem auto                           Авто-извлечь знания из диалога",
      "  --mem clear                          Очистить долгосрочную память",
    ].join("\n"));
    process.exit(1);
  }
  }

  // ── --wm  Working memory management (layer 2) ─────────────────────────────────
  if (args[0] === "--wm") {
    const sub     = args[1];
    const session = loadSession();

    // --wm show  Display current working memory
    if (sub === "show") {
      const wmText = workingToText(session.working);
      if (wmText) {
        console.log("Рабочая память (текущая задача):\n");
        console.log(wmText);
      } else {
        console.log("Рабочая память пуста.");
      }
      process.exit(0);
    }

    // --wm task <описание>  Set current task description
    if (sub === "task") {
      const task = args.slice(2).join(" ");
      if (!task) { console.error("Укажите описание задачи."); process.exit(1); }
      session.working.task = task;
      saveSession(session);
      console.log(`Задача установлена: ${task}`);
      process.exit(0);
    }

    // --wm set <key> <value>  Set a working memory variable
    if (sub === "set") {
      const key   = args[2];
      const value = args.slice(3).join(" ");
      if (!key || !value) {
        console.error("Использование: --wm set <ключ> <значение>");
        process.exit(1);
      }
      session.working.vars[key] = value;
      saveSession(session);
      console.log(`Переменная: ${key} = ${value}`);
      process.exit(0);
    }

    // --wm note <текст>  Add a note
    if (sub === "note") {
      const note = args.slice(2).join(" ");
      if (!note) { console.error("Укажите текст заметки."); process.exit(1); }
      session.working.notes.push(note);
      saveSession(session);
      console.log(`Заметка добавлена: ${note}`);
      process.exit(0);
    }

    // --wm clear  Reset working memory
    if (sub === "clear") {
      session.working = defaultWorking();
      saveSession(session);
      console.log("Рабочая память очищена.");
      process.exit(0);
    }

    console.error([
      "Управление рабочей памятью (текущая задача, сбрасывается при --clear):",
      "  --wm show                 Показать рабочую память",
      "  --wm task <описание>      Установить текущую задачу",
      "  --wm set <ключ> <знач.>   Сохранить переменную задачи",
      "  --wm note <текст>         Добавить заметку",
      "  --wm clear                Очистить рабочую память",
    ].join("\n"));
    process.exit(1);
  }

  // ── --profile  User profile management ───────────────────────────────────────
  if (args[0] === "--profile") {
    const sub = args[1];
    const lt  = loadLongTerm();

    // --profile show  Display current profile with field descriptions
    if (sub === "show") {
      const p = lt.profile ?? {};
      console.log("[ Профиль пользователя ]\n");
      Object.entries(PROFILE_FIELDS).forEach(([field, desc]) => {
        const val = p[field] ?? "(не задано)";
        console.log(`  ${field.padEnd(12)} ${val}`);
        console.log(`               └─ ${desc}`);
      });
      if (Object.keys(p).length === 0) console.log("\n  Профиль пуст. Используйте: --profile set <поле> <значение>");
      process.exit(0);
    }

    // --profile set <field> <value>  Set a profile field (validated against schema)
    if (sub === "set") {
      const field = args[2];
      const value = args.slice(3).join(" ");
      if (!field || !value) {
        console.error("Использование: --profile set <поле> <значение>");
        process.exit(1);
      }
      if (!PROFILE_FIELDS[field]) {
        console.error(`Неизвестное поле "${field}". Доступные поля:\n  ${Object.keys(PROFILE_FIELDS).join(", ")}`);
        process.exit(1);
      }
      // Validate enum fields
      if (field === "style"  && !["formal", "casual", "technical"].includes(value)) {
        console.error('style: допустимые значения — formal | casual | technical');
        process.exit(1);
      }
      if (field === "format" && !["short", "detailed", "bullets", "code-first"].includes(value)) {
        console.error('format: допустимые значения — short | detailed | bullets | code-first');
        process.exit(1);
      }
      lt.profile[field] = value;
      saveLongTerm(lt);
      console.log(`Профиль обновлён: ${field} = ${value}`);
      process.exit(0);
    }

    // --profile clear  Wipe the profile (keep knowledge/decisions)
    if (sub === "clear") {
      lt.profile = {};
      saveLongTerm(lt);
      console.log("Профиль пользователя очищен.");
      process.exit(0);
    }

    const fieldHelp = Object.entries(PROFILE_FIELDS)
      .map(([f, d]) => `    ${f.padEnd(12)} ${d}`)
      .join("\n");
    console.error([
      "Управление профилем пользователя (влияет на каждый запрос):",
      "  --profile show                Показать текущий профиль",
      "  --profile set <поле> <знач.>  Установить поле профиля",
      "  --profile clear               Очистить профиль",
      "",
      "Поля профиля:",
      fieldHelp,
      "",
      "Примеры:",
      '  node agent.js --profile set name "Иван"',
      '  node agent.js --profile set style technical',
      '  node agent.js --profile set format bullets',
      '  node agent.js --profile set language en',
    ].join("\n"));
    process.exit(1);
  }

  // ── --strategy ────────────────────────────────────────────────────────────────
  if (args[0] === "--strategy") {
    const name  = args[1];
    const valid = ["sliding-window", "sticky-facts", "branching"];
    if (!valid.includes(name)) {
      console.error(`Неизвестная стратегия. Доступны: ${valid.join(", ")}`);
      process.exit(1);
    }
    const session    = loadSession();
    session.strategy = name;
    saveSession(session);
    console.log(`Стратегия переключена на: ${name}`);
    process.exit(0);
  }

  // ── --branch ──────────────────────────────────────────────────────────────────
  if (args[0] === "--branch") {
    const session = loadSession();
    if (session.strategy !== "branching") {
      console.error("Ветки доступны только в стратегии branching.");
      console.error("Переключитесь: node agent.js --strategy branching");
      process.exit(1);
    }
    const sub = args[1];

    if (sub === "list") {
      Object.entries(session.branches).forEach(([name, b]) => {
        const marker = name === session.currentBranch ? "* " : "  ";
        console.log(`${marker}${name}  (${b.messages.length} сообщ.${b.summary ? ", есть резюме" : ""})`);
      });
      process.exit(0);
    }

    if (sub === "create") {
      const newName = args[2];
      const fromCp  = args[3];
      if (!newName) { console.error("Укажите имя ветки."); process.exit(1); }
      if (session.branches[newName]) { console.error(`Ветка "${newName}" уже существует.`); process.exit(1); }

      let src;
      if (fromCp) {
        src = session.checkpoints[fromCp];
        if (!src) { console.error(`Checkpoint "${fromCp}" не найден.`); process.exit(1); }
      } else {
        src = getActiveBranch(session);
      }

      session.branches[newName] = {
        messages:   JSON.parse(JSON.stringify(src.messages)),
        summary:    src.summary ?? null,
        checkpoint: fromCp ?? null,
      };
      session.currentBranch = newName;
      saveSession(session);
      const origin = fromCp ? `checkpoint «${fromCp}»` : `текущей ветки`;
      console.log(`Ветка "${newName}" создана от ${origin} и активирована.`);
      process.exit(0);
    }

    if (sub === "switch") {
      const name = args[2];
      if (!name) { console.error("Укажите имя ветки."); process.exit(1); }
      if (!session.branches[name]) { console.error(`Ветка "${name}" не найдена.`); process.exit(1); }
      session.currentBranch = name;
      saveSession(session);
      console.log(`Активная ветка: ${name}  (${session.branches[name].messages.length} сообщений)`);
      process.exit(0);
    }

    if (sub === "delete") {
      const name = args[2];
      if (!name)           { console.error("Укажите имя ветки."); process.exit(1); }
      if (name === "main") { console.error("Нельзя удалить ветку main."); process.exit(1); }
      if (!session.branches[name]) { console.error(`Ветка "${name}" не найдена.`); process.exit(1); }
      if (name === session.currentBranch) {
        session.currentBranch = "main";
        console.log("Активная ветка переключена на main.");
      }
      delete session.branches[name];
      saveSession(session);
      console.log(`Ветка "${name}" удалена.`);
      process.exit(0);
    }

    console.error("Команды: --branch list | create <имя> [<checkpoint>] | switch <имя> | delete <имя>");
    process.exit(1);
  }

  // ── --checkpoint ──────────────────────────────────────────────────────────────
  if (args[0] === "--checkpoint") {
    const session = loadSession();
    if (session.strategy !== "branching") {
      console.error("Checkpoints доступны только в стратегии branching.");
      process.exit(1);
    }
    const sub = args[1];

    if (sub === "save") {
      const cpName = args[2];
      if (!cpName) { console.error("Укажите имя checkpoint."); process.exit(1); }
      const branch = getActiveBranch(session);
      session.checkpoints[cpName] = {
        messages:   JSON.parse(JSON.stringify(branch.messages)),
        summary:    branch.summary ?? null,
        savedAt:    new Date().toISOString(),
        fromBranch: session.currentBranch,
      };
      saveSession(session);
      console.log(`Checkpoint "${cpName}" сохранён (${branch.messages.length} сообщений).`);
      process.exit(0);
    }

    if (sub === "list") {
      const cps = Object.entries(session.checkpoints);
      if (!cps.length) { console.log("Нет сохранённых checkpoints."); process.exit(0); }
      cps.forEach(([name, cp]) =>
        console.log(`  ${name}  (${cp.messages.length} сообщ. | из ветки «${cp.fromBranch}» | ${cp.savedAt})`)
      );
      process.exit(0);
    }

    if (sub === "delete") {
      const cpName = args[2];
      if (!cpName) { console.error("Укажите имя checkpoint."); process.exit(1); }
      if (!session.checkpoints[cpName]) { console.error(`Checkpoint "${cpName}" не найден.`); process.exit(1); }
      delete session.checkpoints[cpName];
      saveSession(session);
      console.log(`Checkpoint "${cpName}" удалён.`);
      process.exit(0);
    }

    console.error("Команды: --checkpoint save <имя> | list | delete <имя>");
    process.exit(1);
  }

  // ── --facts ───────────────────────────────────────────────────────────────────
  if (args[0] === "--facts") {
    const session = loadSession();
    if (session.strategy !== "sticky-facts") {
      console.error("Факты доступны только в стратегии sticky-facts.");
      process.exit(1);
    }
    const facts = session.facts ?? {};
    if (!Object.keys(facts).length) {
      console.log("Факты ещё не извлечены.");
    } else {
      console.log("Ключевые факты диалога:\n");
      Object.entries(facts).forEach(([k, v]) => console.log(`  ${k}: ${v}`));
    }
    process.exit(0);
  }

  // ── --summary ─────────────────────────────────────────────────────────────────
  if (args[0] === "--summary") {
    const session = loadSession();
    const summary = session.strategy === "branching"
      ? getActiveBranch(session).summary
      : session.summary;
    if (summary) {
      console.log("Текущее резюме:\n");
      console.log(summary);
    } else {
      console.log("Резюме ещё нет.");
    }
    process.exit(0);
  }

  // ── --stats ───────────────────────────────────────────────────────────────────
  if (args[0] === "--stats") {
    const { totalTokens, requests } = getTokenStats();
    console.log(`Токенов за сессию: ${totalTokens}`);
    console.log(`Запросов: ${requests.length}`);
    requests.forEach((r, i) =>
      console.log(`  [${i + 1}] ход ${r.turn ?? "?"} | ${r.timestamp}  prompt=${r.promptTokens ?? "?"}  completion=${r.completionTokens ?? "?"}  total=${r.totalTokens ?? "?"}`)
    );
    process.exit(0);
  }

  // ── --status ──────────────────────────────────────────────────────────────────
  if (args[0] === "--status") {
    const session  = loadSession();
    const longTerm = loadLongTerm();
    const s        = session.strategy;

    console.log("═══ Состояние памяти ассистента ═══\n");

    // Layer 1: short-term
    console.log("[ Слой 1: Краткосрочная память ]");
    console.log(`  Стратегия: ${s}`);
    if (s === "sliding-window") {
      console.log(`  Окно: последние ${CONFIG.keepRecent} сообщений (старые отбрасываются)`);
      console.log(`  Сообщений: ${session.messages.length}  |  Ход: ${session.turn}`);
    }
    if (s === "sticky-facts") {
      console.log(`  Окно: последние ${CONFIG.keepRecent} сообщений + факты`);
      console.log(`  Сообщений: ${session.messages.length}  |  Ход: ${session.turn}`);
    }
    if (s === "branching") {
      const b = getActiveBranch(session);
      console.log(`  Ветка: ${session.currentBranch}  (${b.messages.length} сообщений)`);
      console.log(`  Веток: ${Object.keys(session.branches).length}  |  Checkpoints: ${Object.keys(session.checkpoints ?? {}).length}`);
    }

    // Layer 2: working memory
    console.log("\n[ Слой 2: Рабочая память ]");
    const wm = session.working;
    if (wm.task || Object.keys(wm.vars).length || wm.notes.length) {
      if (wm.task)                   console.log(`  Задача: ${wm.task}`);
      Object.entries(wm.vars).forEach(([k, v]) => console.log(`  ${k} = ${v}`));
      wm.notes.forEach(n             => console.log(`  • ${n}`));
    } else {
      console.log("  (пусто)");
    }

    // Layer 3: long-term memory
    console.log("\n[ Слой 3: Долгосрочная память ]");
    const profileEntries = Object.entries(longTerm.profile   ?? {});
    const knowledgeN     = Object.keys(longTerm.knowledge ?? {}).length;
    const decisionsN     = (longTerm.decisions ?? []).length;
    if (profileEntries.length + knowledgeN + decisionsN === 0) {
      console.log("  (пусто)");
    } else {
      if (profileEntries.length) {
        console.log(`  Профиль (${profileEntries.length} полей):`);
        profileEntries.forEach(([k, v]) => console.log(`    ${k}: ${v}`));
      } else {
        console.log("  Профиль: (не задан — используйте --profile set)");
      }
      console.log(`  Знания:  ${knowledgeN} записей`);
      console.log(`  Решения: ${decisionsN}`);
      if (longTerm.updatedAt) console.log(`  Обновлено: ${longTerm.updatedAt}`);
    }

    process.exit(0);
  }

  // ── Default: treat all args as a query ────────────────────────────────────────
  const query = args.join(" ");
  if (!query) {
    console.error([
      "Использование: node agent.js \"<вопрос>\"",
      "",
      "Управление памятью:",
      "  --status                             Состояние всех уровней памяти",
      "",
      "  Краткосрочная (текущий диалог):",
      "  --clear                              Сбросить диалог и рабочую память",
      "  --stats                              Статистика токенов",
      "",
      "  Рабочая память (текущая задача):",
      "  --wm show                            Показать рабочую память",
      "  --wm task <описание>                 Установить задачу",
      "  --wm set <ключ> <значение>           Сохранить переменную",
      "  --wm note <текст>                    Добавить заметку",
      "  --wm clear                           Очистить рабочую память",
      "",
      "  Профиль пользователя (персонализация каждого запроса):",
      "  --profile show                       Показать профиль",
      "  --profile set <поле> <значение>      Установить поле профиля",
      "  --profile clear                      Очистить профиль",
      "",
      "  Долгосрочная память (между сессиями):",
      "  --mem list                           Показать долгосрочную память",
      "  --mem set profile|knowledge <k> <v>  Сохранить факт",
      "  --mem decide <что> [--why <почему>]  Зафиксировать решение",
      "  --mem delete profile|knowledge <k>   Удалить факт",
      "  --mem auto                           Авто-извлечь знания из диалога",
      "  --mem clear                          Очистить долгосрочную память",
      "",
      "  Стратегии управления контекстом:",
      "  --strategy sliding-window            Скользящее окно",
      "  --strategy sticky-facts              Факты + окно",
      "  --strategy branching                 Ветки диалога",
      "",
      "  Ветки (только branching):",
      "  --branch list | create | switch | delete",
      "  --checkpoint save | list | delete",
      "  --facts                              Показать факты (sticky-facts)",
      "  --summary                            Показать резюме сжатия",
    ].join("\n"));
    process.exit(1);
  }

  ask(query)
    .then((answer) => {
      console.log(answer);
      const session = loadSession();
      const { totalTokens, requests } = getTokenStats();
      const last  = requests.at(-1);
      const extra = session.strategy === "branching" ? ` | ветка: ${session.currentBranch}` : "";
      if (last?.totalTokens != null) {
        console.log(`\n[ход: ${session.turn} | токены: ${last.totalTokens} | сессия: ${totalTokens} | стратегия: ${session.strategy}${extra}]`);
      }
    })
    .catch((err) => { console.error(err.message); process.exit(1); });
}
