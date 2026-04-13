import { randomUUID } from "crypto";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// GigaChat uses a certificate signed by Russian NCA — disable TLS verification for dev.
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

// ─── Config ────────────────────────────────────────────────────────────────────
const CONFIG = {
  model:        "GigaChat",
  system:       "Ты — AI-ассистент с доступом к внешним инструментам (functions). ОБЯЗАТЕЛЬНОЕ ПРАВИЛО: если запрос требует реальных данных (цены, файлы, git, статистика) — ВСЕГДА вызывай соответствующий инструмент, НИКОГДА не симулируй данные и не пиши псевдокод. Отвечай на языке пользователя.",
  temperature:  0.7,
  max_tokens:   2048,
  top_p:        0.9,
  keepRecent:   6,
  maxToolCalls: 10,  // максимум вызовов инструментов за один ход
};

// ─── File paths ────────────────────────────────────────────────────────────────
const __dir          = dirname(fileURLToPath(import.meta.url));
const HISTORY_FILE   = join(__dir, "history.json");
const MEMORY_FILE    = join(__dir, "memory.json");
const MCP_CONFIG_FILE = join(__dir, "mcp-servers.json");

// ══════════════════════════════════════════════════════════════════════════════
// MCP MANAGER
//
// Управляет подключениями к нескольким MCP-серверам.
// Поддерживает транспорты: stdio, sse.
// Инструменты именуются как «serverName__toolName» для избежания конфликтов.
// ══════════════════════════════════════════════════════════════════════════════

function loadMcpConfig() {
  try {
    if (!existsSync(MCP_CONFIG_FILE)) return { servers: [] };
    return JSON.parse(readFileSync(MCP_CONFIG_FILE, "utf8"));
  } catch {
    return { servers: [] };
  }
}

function saveMcpConfig(cfg) {
  writeFileSync(MCP_CONFIG_FILE, JSON.stringify(cfg, null, 2), "utf8");
}

class McpManager {
  constructor() {
    // Map<serverName, { client, tools: MCP tool objects }>
    this.clients  = new Map();
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return;
    const { servers } = loadMcpConfig();
    for (const cfg of servers) {
      try {
        await this._connect(cfg);
      } catch (err) {
        console.error(`[MCP] Ошибка подключения "${cfg.name}": ${err.message}`);
      }
    }
    this.initialized = true;
  }

  async _connect(cfg) {
    const client = new Client(
      { name: "mcp-agent", version: "1.0.0" },
      { capabilities: {} },
    );

    let transport;
    if (cfg.type === "stdio") {
      transport = new StdioClientTransport({
        command: cfg.command,
        args:    cfg.args ?? [],
        env:     cfg.env ? { ...process.env, ...cfg.env } : undefined,
      });
    } else if (cfg.type === "sse") {
      const { SSEClientTransport } = await import("@modelcontextprotocol/sdk/client/sse.js");
      transport = new SSEClientTransport(new URL(cfg.url));
    } else {
      throw new Error(`Неизвестный тип транспорта: ${cfg.type}`);
    }

    await client.connect(transport);
    const { tools } = await client.listTools();
    this.clients.set(cfg.name, { client, tools });
    console.error(`[MCP] Подключён "${cfg.name}": ${tools.length} инструментов`);
  }

  // Возвращает все инструменты в формате GigaChat functions.
  // Имя = «serverName__toolName», чтобы не было конфликтов между серверами.
  getAllFunctions() {
    const result = [];
    for (const [serverName, { tools }] of this.clients) {
      for (const tool of tools) {
        result.push({
          name:        `${serverName}__${tool.name}`,
          description: tool.description ?? "",
          parameters:  tool.inputSchema  ?? { type: "object", properties: {} },
        });
      }
    }
    return result;
  }

  // Вызывает инструмент по имени «serverName__toolName» с аргументами.
  async callTool(prefixedName, args) {
    const sep = prefixedName.indexOf("__");
    if (sep === -1) throw new Error(`Некорректное имя инструмента: "${prefixedName}"`);

    const serverName = prefixedName.slice(0, sep);
    const toolName   = prefixedName.slice(sep + 2);

    const entry = this.clients.get(serverName);
    if (!entry) throw new Error(`Сервер "${serverName}" не подключён`);

    const result = await entry.client.callTool({ name: toolName, arguments: args });

    // MCP возвращает массив content-блоков
    if (Array.isArray(result.content)) {
      return result.content.map(c => {
        if (c.type === "text")  return c.text;
        if (c.type === "image") return `[image: ${c.url ?? "base64"}]`;
        return JSON.stringify(c);
      }).join("\n");
    }
    return String(result.content ?? "");
  }

  // Статус подключений
  status() {
    if (!this.initialized) return "не инициализирован";
    if (!this.clients.size)  return "нет подключённых серверов";
    const lines = [];
    for (const [name, { tools }] of this.clients) {
      lines.push(`  ${name}  (${tools.length} инструментов)`);
    }
    return lines.join("\n");
  }

  // Список всех доступных инструментов
  listToolNames() {
    const result = [];
    for (const [serverName, { tools }] of this.clients) {
      for (const t of tools) {
        result.push({ server: serverName, name: t.name, description: t.description ?? "" });
      }
    }
    return result;
  }
}

// Единственный экземпляр на процесс
const mcpManager = new McpManager();

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY LAYER 3 — ДОЛГОСРОЧНАЯ (long-term)
// ══════════════════════════════════════════════════════════════════════════════

function defaultLongTerm() {
  return { profile: {}, knowledge: {}, decisions: [], updatedAt: null };
}

function loadLongTerm() {
  try {
    if (!existsSync(MEMORY_FILE)) return defaultLongTerm();
    const raw = JSON.parse(readFileSync(MEMORY_FILE, "utf8"));
    const lt  = defaultLongTerm();
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
  const profileEntries   = Object.entries(lt.profile   ?? {});
  const knowledgeEntries = Object.entries(lt.knowledge ?? {});

  if (profileEntries.length) {
    lines.push("[Профиль пользователя]");
    profileEntries.forEach(([k, v]) => lines.push(`  ${k}: ${v}`));
  }
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

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY LAYER 2 — РАБОЧАЯ (working)
// ══════════════════════════════════════════════════════════════════════════════

function defaultWorking() {
  return { task: null, vars: {}, notes: [] };
}

function workingToText(w) {
  if (!w) return "";
  const lines = [];
  if (w.task) lines.push(`Задача: ${w.task}`);
  Object.entries(w.vars ?? {}).forEach(([k, v]) => lines.push(`  ${k} = ${v}`));
  (w.notes ?? []).forEach(n => lines.push(`  • ${n}`));
  return lines.join("\n");
}

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY LAYER 1 — КРАТКОСРОЧНАЯ (short-term / session)
// ══════════════════════════════════════════════════════════════════════════════

function makeMessage(role, content, turn) {
  return { role, content, turn, at: new Date().toISOString() };
}

function defaultSession() {
  return {
    messages:    [],
    working:     defaultWorking(),
    turn:        0,
    totalTokens: 0,
    requests:    [],
    // Лог вызовов MCP-инструментов за всю сессию
    toolCalls:   [],
  };
}

function loadSession() {
  try {
    if (!existsSync(HISTORY_FILE)) return defaultSession();
    const raw = JSON.parse(readFileSync(HISTORY_FILE, "utf8"));
    if (Array.isArray(raw)) {
      const s = defaultSession();
      s.messages = raw;
      return s;
    }
    const s = defaultSession();
    Object.assign(s, raw);
    if (!s.working)   s.working   = defaultWorking();
    if (s.turn == null) s.turn    = 0;
    if (!s.toolCalls) s.toolCalls = [];
    return s;
  } catch {
    return defaultSession();
  }
}

function saveSession(session) {
  writeFileSync(HISTORY_FILE, JSON.stringify(session, null, 2), "utf8");
}

// ══════════════════════════════════════════════════════════════════════════════
// BUILD CONTEXT
//
// Собирает массив сообщений для GigaChat API, внедряя все уровни памяти.
// ══════════════════════════════════════════════════════════════════════════════

function buildContext(session, longTerm) {
  const parts = [CONFIG.system];

  const ltText = longTermToText(longTerm);
  if (ltText) parts.push("\n[Долгосрочная память]\n" + ltText);

  const wmText = workingToText(session.working);
  if (wmText) parts.push("\n[Рабочая память — текущая задача]\n" + wmText);

  const systemContent = parts.join("\n");
  const recent        = session.messages.slice(-CONFIG.keepRecent);
  return [{ role: "system", content: systemContent }, ...recent];
}

// ══════════════════════════════════════════════════════════════════════════════
// GIGACHAT API
// ══════════════════════════════════════════════════════════════════════════════

const GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
const GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions";

let cachedToken    = null;
let tokenExpiresAt = 0;

async function getAccessToken() {
  if (cachedToken && Date.now() / 1000 < tokenExpiresAt - 60) return cachedToken;

  const apiKey = process.env.GIGACHAT_API_KEY;
  if (!apiKey) throw new Error("GIGACHAT_API_KEY environment variable is not set");

  const res = await fetch(GIGACHAT_AUTH_URL, {
    method:  "POST",
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

// Однократный (не стриминговый) запрос к GigaChat.
// Используется в tool-calling loop.
async function callGigaChatOnce(messages, { functions, temperature, max_tokens } = {}) {
  const token = await getAccessToken();

  const body = {
    model:       CONFIG.model,
    messages,
    temperature: temperature ?? CONFIG.temperature,
    max_tokens:  max_tokens  ?? CONFIG.max_tokens,
    top_p:       CONFIG.top_p,
  };

  if (functions?.length) {
    body.functions     = functions;
    body.function_call = "auto";
  }

  // DEBUG_PROMPT=1 — печатает структуру запроса в stderr перед отправкой
  if (process.env.DEBUG_PROMPT === "1") {
    const systemMsg = messages.find(m => m.role === "system");
    const userMsg   = messages.filter(m => m.role === "user").at(-1);
    const dump = {
      model:       body.model,
      systemPrompt: systemMsg?.content ?? "",
      userMessage:  userMsg?.content   ?? "",
      historyLen:   messages.filter(m => m.role !== "system").length,
      functions:    (functions ?? []).map(f => ({
        name:        f.name,
        server:      f.name.split("__")[0],
        description: f.description.slice(0, 80),
      })),
    };
    process.stderr.write("[PROMPT:START]\n" + JSON.stringify(dump, null, 2) + "\n[PROMPT:END]\n");
  }

  const res = await fetch(GIGACHAT_CHAT_URL, {
    method:  "POST",
    headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });

  if (!res.ok) throw new Error(`GigaChat error: ${res.status} ${await res.text()}`);
  const data = await res.json();

  return {
    message:      data.choices[0].message,
    finishReason: data.choices[0].finish_reason,
    usage:        data.usage ?? null,
  };
}

// ══════════════════════════════════════════════════════════════════════════════
// CORE: ask()
//
// Основной цикл агента:
//   1. Собрать контекст с памятью
//   2. Инициализировать MCP-серверы
//   3. Tool-calling loop: вызывать модель → выполнять инструменты → повторять
//   4. Финальный ответ — вернуть пользователю
// ══════════════════════════════════════════════════════════════════════════════

export async function ask(userQuery) {
  if (!userQuery?.trim()) throw new Error("Query must not be empty");

  const session  = loadSession();
  const longTerm = loadLongTerm();

  session.turn = (session.turn ?? 0) + 1;
  const turn = session.turn;

  // Добавляем сообщение пользователя в краткосрочную память
  session.messages.push(makeMessage("user", userQuery, turn));

  // Инициализируем MCP (ленивая инициализация)
  await mcpManager.init();
  const functions = mcpManager.getAllFunctions();

  // Строим массив сообщений для API (включает историю + системный промпт)
  const contextMessages = buildContext(session, longTerm);

  // Рабочий массив для tool-calling loop — не смешиваем с session.messages
  const loopMessages = [
    ...contextMessages.slice(0, 1),           // [0] = system
    ...contextMessages.slice(1, -1),          // история без последнего user
    { role: "user", content: userQuery },     // свежий user (без метаданных)
  ];

  let finalReply   = "";
  let toolCallCount = 0;
  let lastUsage     = null;

  // ── Tool-calling loop ────────────────────────────────────────────────────────
  while (true) {
    const { message, finishReason, usage } = await callGigaChatOnce(loopMessages, { functions });
    if (usage) lastUsage = usage;

    // Добавляем ответ ассистента в рабочий контекст
    loopMessages.push(message);

    const isToolCall = finishReason === "function_call" && message.function_call;
    if (isToolCall && toolCallCount < CONFIG.maxToolCalls) {
      const fnCall = message.function_call;
      const fnName = fnCall.name;
      let fnArgs;
      try {
        fnArgs = typeof fnCall.arguments === "string"
          ? JSON.parse(fnCall.arguments)
          : (fnCall.arguments ?? {});
      } catch {
        fnArgs = {};
      }

      toolCallCount++;
      console.error(`[MCP] [${toolCallCount}] ${fnName}(${JSON.stringify(fnArgs)})`);

      let toolResult;
      try {
        toolResult = await mcpManager.callTool(fnName, fnArgs);
      } catch (err) {
        toolResult = `Ошибка выполнения инструмента: ${err.message}`;
      }

      console.error(`[MCP] ← ${toolResult.slice(0, 300)}${toolResult.length > 300 ? "…" : ""}`);

      // Сохраняем вызов в лог сессии
      session.toolCalls.push({
        turn,
        step:   toolCallCount,
        tool:   fnName,
        args:   fnArgs,
        result: toolResult.slice(0, 1000),
        at:     new Date().toISOString(),
      });

      // Добавляем результат инструмента в рабочий контекст
      // GigaChat требует content функции в виде JSON-строки
      loopMessages.push({ role: "function", name: fnName, content: JSON.stringify(toolResult) });
    } else {
      // Финальный ответ — либо нет вызова, либо достигнут лимит
      if (isToolCall && toolCallCount >= CONFIG.maxToolCalls) {
        console.error(`[MCP] Достигнут лимит вызовов инструментов (${CONFIG.maxToolCalls})`);
      }
      finalReply = message.content ?? "";
      break;
    }
  }

  // Добавляем финальный ответ в краткосрочную память (только текст)
  session.messages.push(makeMessage("assistant", finalReply, turn));

  // Скользящее окно
  session.messages = session.messages.slice(-CONFIG.keepRecent * 2);

  // Записываем статистику токенов
  if (lastUsage) {
    session.requests.push({
      timestamp:        new Date().toISOString(),
      turn,
      toolCalls:        toolCallCount,
      promptTokens:     lastUsage.prompt_tokens     ?? null,
      completionTokens: lastUsage.completion_tokens ?? null,
      totalTokens:      lastUsage.total_tokens      ?? null,
    });
    if (lastUsage.total_tokens) session.totalTokens = (session.totalTokens ?? 0) + lastUsage.total_tokens;
  }

  saveSession(session);
  return finalReply;
}

export function getTokenStats() {
  const { totalTokens, requests } = loadSession();
  return { totalTokens, requests };
}

// ══════════════════════════════════════════════════════════════════════════════
// CLI ENTRY POINT
// ══════════════════════════════════════════════════════════════════════════════

if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(
  process.argv[1].replace(/\\/g, "/"),
)) {
  (async () => {
  const args = process.argv.slice(2);

  // ── --clear ──────────────────────────────────────────────────────────────────
  if (args[0] === "--clear") {
    saveSession(defaultSession());
    console.log("Краткосрочная и рабочая память очищены. Долгосрочная память сохранена.");
    process.exit(0);
  }

  // ── --mem ────────────────────────────────────────────────────────────────────
  if (args[0] === "--mem") {
    const sub = args[1];
    const lt  = loadLongTerm();

    if (sub === "list" || sub === "show") {
      const text = longTermToText(lt);
      console.log(text || "Долгосрочная память пуста.");
      if (lt.updatedAt) console.log(`\nОбновлено: ${lt.updatedAt}`);
      process.exit(0);
    }
    if (sub === "set") {
      const [, , section, key, ...rest] = args;
      const value = rest.join(" ");
      if (!section || !key || !value || !["profile", "knowledge"].includes(section)) {
        console.error("Использование: --mem set profile|knowledge <ключ> <значение>");
        process.exit(1);
      }
      lt[section][key] = value;
      saveLongTerm(lt);
      console.log(`Сохранено в ${section}: ${key} = ${value}`);
      process.exit(0);
    }
    if (sub === "delete") {
      const [, , section, key] = args;
      if (!section || !key) { console.error("--mem delete profile|knowledge <ключ>"); process.exit(1); }
      delete lt[section]?.[key];
      saveLongTerm(lt);
      console.log(`Удалено из ${section}: ${key}`);
      process.exit(0);
    }
    if (sub === "clear") {
      saveLongTerm(defaultLongTerm());
      console.log("Долгосрочная память очищена.");
      process.exit(0);
    }
    console.error([
      "Управление долгосрочной памятью:",
      "  --mem list                           Показать всю долгосрочную память",
      "  --mem set profile|knowledge <k> <v>  Сохранить факт",
      "  --mem delete profile|knowledge <k>   Удалить факт",
      "  --mem clear                          Очистить долгосрочную память",
    ].join("\n"));
    process.exit(1);
  }

  // ── --wm ─────────────────────────────────────────────────────────────────────
  if (args[0] === "--wm") {
    const sub     = args[1];
    const session = loadSession();

    if (sub === "show") {
      const text = workingToText(session.working);
      console.log(text || "Рабочая память пуста.");
      process.exit(0);
    }
    if (sub === "task") {
      const task = args.slice(2).join(" ");
      if (!task) { console.error("Укажите описание задачи."); process.exit(1); }
      session.working.task = task;
      saveSession(session);
      console.log(`Задача: ${task}`);
      process.exit(0);
    }
    if (sub === "set") {
      const key = args[2]; const value = args.slice(3).join(" ");
      if (!key || !value) { console.error("--wm set <ключ> <значение>"); process.exit(1); }
      session.working.vars[key] = value;
      saveSession(session);
      console.log(`${key} = ${value}`);
      process.exit(0);
    }
    if (sub === "note") {
      const note = args.slice(2).join(" ");
      if (!note) { console.error("Укажите текст заметки."); process.exit(1); }
      session.working.notes.push(note);
      saveSession(session);
      console.log(`Заметка: ${note}`);
      process.exit(0);
    }
    if (sub === "clear") {
      session.working = defaultWorking();
      saveSession(session);
      console.log("Рабочая память очищена.");
      process.exit(0);
    }
    console.error([
      "Управление рабочей памятью:",
      "  --wm show                 Показать",
      "  --wm task <описание>      Установить задачу",
      "  --wm set <ключ> <знач.>   Сохранить переменную",
      "  --wm note <текст>         Добавить заметку",
      "  --wm clear                Очистить",
    ].join("\n"));
    process.exit(1);
  }

  // ── --mcp ────────────────────────────────────────────────────────────────────
  if (args[0] === "--mcp") {
    const sub = args[1];
    const cfg = loadMcpConfig();

    // --mcp list
    if (sub === "list" || !sub) {
      if (!cfg.servers.length) {
        console.log("Нет настроенных MCP-серверов.");
        console.log("Добавьте: node agent.js --mcp add <имя> stdio <команда> [аргументы...]");
        console.log("          node agent.js --mcp add <имя> sse <url>");
      } else {
        console.log("Настроенные MCP-серверы:\n");
        cfg.servers.forEach(s => {
          const info = s.type === "sse"
            ? `sse  ${s.url}`
            : `stdio  ${s.command}${s.args?.length ? " " + s.args.join(" ") : ""}`;
          const desc = s.description ? `  # ${s.description}` : "";
          console.log(`  ${s.name.padEnd(20)} ${info}${desc}`);
        });
      }
      process.exit(0);
    }

    // --mcp add <name> stdio <command> [args...]
    // --mcp add <name> sse <url>
    if (sub === "add") {
      const [, , name, type, ...rest] = args;
      if (!name || !type || !rest.length) {
        console.error("Использование:");
        console.error("  --mcp add <имя> stdio <команда> [аргументы...]");
        console.error("  --mcp add <имя> sse <url>");
        process.exit(1);
      }
      if (cfg.servers.find(s => s.name === name)) {
        console.error(`Сервер "${name}" уже существует. Удалите сначала: --mcp remove ${name}`);
        process.exit(1);
      }
      if (type !== "stdio" && type !== "sse") {
        console.error("Тип должен быть: stdio | sse");
        process.exit(1);
      }

      if (type === "stdio") {
        cfg.servers.push({ name, type: "stdio", command: rest[0], args: rest.slice(1) });
      } else {
        cfg.servers.push({ name, type: "sse", url: rest[0] });
      }
      saveMcpConfig(cfg);
      console.log(`Сервер "${name}" добавлен.`);
      process.exit(0);
    }

    // --mcp remove <name>
    if (sub === "remove") {
      const name = args[2];
      if (!name) { console.error("Укажите имя сервера."); process.exit(1); }
      const before = cfg.servers.length;
      cfg.servers = cfg.servers.filter(s => s.name !== name);
      if (cfg.servers.length === before) {
        console.error(`Сервер "${name}" не найден.`);
        process.exit(1);
      }
      saveMcpConfig(cfg);
      console.log(`Сервер "${name}" удалён.`);
      process.exit(0);
    }

    // --mcp tools  Список всех доступных инструментов со всех серверов
    if (sub === "tools") {
      mcpManager.init().then(() => {
        const tools = mcpManager.listToolNames();
        if (!tools.length) {
          console.log("Нет доступных инструментов.");
        } else {
          console.log("Доступные инструменты:\n");
          tools.forEach(t => {
            console.log(`  ${t.server}__${t.name}`);
            if (t.description) console.log(`    ${t.description}`);
          });
        }
        process.exit(0);
      }).catch(err => { console.error(err.message); process.exit(1); });
      return;
    }

    // --mcp status
    if (sub === "status") {
      mcpManager.init().then(() => {
        console.log("MCP-серверы:\n");
        console.log(mcpManager.status());
        process.exit(0);
      }).catch(err => { console.error(err.message); process.exit(1); });
      return;
    }

    console.error([
      "Управление MCP-серверами:",
      "  --mcp list                           Список настроенных серверов",
      "  --mcp add <имя> stdio <команда> ...  Добавить stdio-сервер",
      "  --mcp add <имя> sse <url>            Добавить SSE-сервер",
      "  --mcp remove <имя>                   Удалить сервер",
      "  --mcp tools                          Список всех инструментов",
      "  --mcp status                         Состояние подключений",
    ].join("\n"));
    process.exit(1);
  }

  // ── --tools  Быстрый список инструментов ─────────────────────────────────────
  if (args[0] === "--tools") {
    mcpManager.init().then(() => {
      const tools = mcpManager.listToolNames();
      if (!tools.length) {
        console.log("Нет доступных инструментов.");
        console.log("Добавьте MCP-серверы: node agent.js --mcp add ...");
      } else {
        console.log(`Доступных инструментов: ${tools.length}\n`);
        let lastServer = null;
        tools.forEach(t => {
          if (t.server !== lastServer) { console.log(`  [${t.server}]`); lastServer = t.server; }
          console.log(`    ${t.name.padEnd(30)} ${t.description.slice(0, 60)}`);
        });
      }
      process.exit(0);
    }).catch(err => { console.error(err.message); process.exit(1); });
    return;
  }

  // ── --tool-log  История вызовов инструментов в текущей сессии ────────────────
  if (args[0] === "--tool-log") {
    const session = loadSession();
    if (!session.toolCalls?.length) {
      console.log("Вызовов инструментов в текущей сессии нет.");
    } else {
      console.log(`Вызовов инструментов: ${session.toolCalls.length}\n`);
      session.toolCalls.forEach((c, i) => {
        console.log(`  [${i + 1}] ход=${c.turn} шаг=${c.step}  ${c.tool}`);
        console.log(`       аргументы: ${JSON.stringify(c.args)}`);
        console.log(`       результат: ${c.result.slice(0, 120)}${c.result.length > 120 ? "…" : ""}`);
      });
    }
    process.exit(0);
  }

  // ── --stats ───────────────────────────────────────────────────────────────────
  if (args[0] === "--stats") {
    const { totalTokens, requests } = getTokenStats();
    console.log(`Токенов за сессию: ${totalTokens ?? 0}`);
    console.log(`Запросов: ${requests.length}`);
    requests.forEach((r, i) =>
      console.log(
        `  [${i + 1}] ход ${r.turn ?? "?"} | ${r.timestamp}` +
        `  prompt=${r.promptTokens ?? "?"}  completion=${r.completionTokens ?? "?"}` +
        `  total=${r.totalTokens ?? "?"}  tools=${r.toolCalls ?? 0}`,
      )
    );
    process.exit(0);
  }

  // ── --status ──────────────────────────────────────────────────────────────────
  if (args[0] === "--status") {
    const session  = loadSession();
    const longTerm = loadLongTerm();
    const cfg      = loadMcpConfig();

    console.log("═══ Состояние mcp-agent ═══\n");

    console.log("[ MCP-серверы ]");
    if (!cfg.servers.length) {
      console.log("  нет (добавьте: --mcp add)");
    } else {
      cfg.servers.forEach(s => {
        const info = s.type === "sse" ? `sse ${s.url}` : `stdio ${s.command}`;
        console.log(`  ${s.name}  (${info})`);
      });
    }

    console.log("\n[ Слой 1: Краткосрочная память ]");
    console.log(`  Сообщений: ${session.messages.length}  |  Ход: ${session.turn}`);
    console.log(`  Вызовов инструментов: ${session.toolCalls?.length ?? 0}`);

    console.log("\n[ Слой 2: Рабочая память ]");
    const wm = session.working;
    if (wm.task || Object.keys(wm.vars).length || wm.notes.length) {
      if (wm.task) console.log(`  Задача: ${wm.task}`);
      Object.entries(wm.vars).forEach(([k, v]) => console.log(`  ${k} = ${v}`));
      wm.notes.forEach(n => console.log(`  • ${n}`));
    } else {
      console.log("  (пусто)");
    }

    console.log("\n[ Слой 3: Долгосрочная память ]");
    const pCount = Object.keys(longTerm.profile   ?? {}).length;
    const kCount = Object.keys(longTerm.knowledge ?? {}).length;
    const dCount = (longTerm.decisions ?? []).length;
    if (!pCount && !kCount && !dCount) {
      console.log("  (пусто)");
    } else {
      console.log(`  Профиль: ${pCount} полей  |  Знания: ${kCount}  |  Решения: ${dCount}`);
    }

    process.exit(0);
  }

  // ── --mcp-test  Прямое тестирование инструментов без GigaChat ────────────────
  // Вызывает инструменты с разных серверов и проверяет маршрутизацию.
  // Выводит JSON с результатами каждого вызова.
  if (args[0] === "--mcp-test") {
    await mcpManager.init();

    const results = [];

    async function runCall(label, toolName, toolArgs) {
      const start = Date.now();
      try {
        const result = await mcpManager.callTool(toolName, toolArgs);
        results.push({ label, tool: toolName, ok: true, ms: Date.now() - start,
          preview: result.slice(0, 120) });
        console.error(`  [OK]  ${toolName} (${Date.now() - start}ms)`);
      } catch (err) {
        results.push({ label, tool: toolName, ok: false, ms: Date.now() - start,
          error: err.message });
        console.error(`  [ERR] ${toolName}: ${err.message}`);
      }
    }

    console.error("[mcp-test] Сценарий A: разные серверы (btc + git)");
    await runCall("btc_price",  "btc__btc_get_price", {});
    await runCall("git_status", "git__git_status",
      { path: "C:\\Users\\user\\Documents\\claude" });

    console.error("[mcp-test] Сценарий B: цепочка btc → summarize → save");
    await runCall("btc_fetch",    "btc__btc_fetch_24h",             {});
    await runCall("summarize",    "btc-summarize__summarize_btc",   {});
    await runCall("save_report",  "btc-save__save_report",          {});

    console.log(JSON.stringify(results, null, 2));
    process.exit(0);
  }

  // ── Default: отправляем запрос агенту ─────────────────────────────────────────
  const query = args.join(" ");
  if (!query) {
    console.error([
      "Использование: node agent.js \"<вопрос>\"",
      "",
      "MCP-серверы:",
      "  --mcp list                           Список серверов",
      "  --mcp add <имя> stdio <команда> ...  Добавить stdio-сервер",
      "  --mcp add <имя> sse <url>            Добавить SSE-сервер",
      "  --mcp remove <имя>                   Удалить сервер",
      "  --tools                              Список инструментов всех серверов",
      "  --tool-log                           История вызовов инструментов",
      "",
      "Память:",
      "  --status                             Состояние агента",
      "  --clear                              Сбросить диалог и рабочую память",
      "  --stats                              Статистика токенов",
      "",
      "  Рабочая память:",
      "  --wm show | task | set | note | clear",
      "",
      "  Долгосрочная память:",
      "  --mem list | set | delete | clear",
    ].join("\n"));
    process.exit(1);
  }

  try {
    const reply = await ask(query);
    console.log(reply);
    process.exit(0);  // MCP-серверы держат дочерние процессы живыми — нужен явный выход
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
  })();
}
