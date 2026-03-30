import { randomUUID } from "crypto";
import { readFileSync, writeFileSync } from "fs";
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

// ─── Persistent history ────────────────────────────────────────────────────────
const __dir = dirname(fileURLToPath(import.meta.url));
const HISTORY_FILE = join(__dir, "history.json");

function defaultSession() {
  return {
    strategy:      "sliding-window",  // "sliding-window" | "sticky-facts" | "branching"
    // sliding-window & sticky-facts
    messages:      [],
    summary:       null,
    // sticky-facts only
    facts:         {},
    // branching only
    branches:      { main: { messages: [], summary: null } },
    checkpoints:   {},   // name → { messages, summary } — frozen snapshots
    currentBranch: "main",
    // shared
    totalTokens:   0,
    requests:      [],
  };
}

function loadSession() {
  try {
    const raw = JSON.parse(readFileSync(HISTORY_FILE, "utf8"));
    if (Array.isArray(raw)) {
      // Migrate old format (plain array) → new format
      const s = defaultSession();
      s.messages = raw;
      s.branches.main.messages = [...raw];
      return s;
    }
    const s = defaultSession();
    Object.assign(s, raw);
    // Ensure new fields exist after migration from older versions
    if (!s.facts)         s.facts = {};
    if (!s.branches)      s.branches = { main: { messages: s.messages ?? [], summary: s.summary ?? null } };
    if (!s.checkpoints)   s.checkpoints = {};
    if (!s.currentBranch) s.currentBranch = "main";
    if (!s.strategy)      s.strategy = "sliding-window";
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

// Non-streaming helper used by summarize & extractFacts
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
    .slice(-4) // analyse only recent messages for new facts
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
    return existingFacts; // on error, keep existing facts unchanged
  }
}

// ─── Build API message list based on active strategy ──────────────────────────
//
// Strategy 1 — Sliding Window:
//   sends only the last keepRecent messages; older messages are discarded.
//
// Strategy 2 — Sticky Facts:
//   sends a system prompt enriched with key-value facts + last keepRecent messages.
//   facts are updated AFTER each exchange (in postProcess), so they always reflect
//   all previous turns.
//
// Strategy 3 — Branching:
//   sends all messages of the current branch (compressed when too long).
//
function buildContext(session) {
  const { strategy } = session;

  if (strategy === "sliding-window") {
    const recent = session.messages.slice(-CONFIG.keepRecent);
    return [{ role: "system", content: CONFIG.system }, ...recent];
  }

  if (strategy === "sticky-facts") {
    const entries = Object.entries(session.facts ?? {});
    const systemContent = entries.length
      ? `${CONFIG.system}\n\n[Ключевые факты диалога]\n${entries.map(([k, v]) => `• ${k}: ${v}`).join("\n")}`
      : CONFIG.system;
    const recent = session.messages.slice(-CONFIG.keepRecent);
    return [{ role: "system", content: systemContent }, ...recent];
  }

  if (strategy === "branching") {
    const branch = getActiveBranch(session);
    const systemContent = branch.summary
      ? `${CONFIG.system}\n\nКонтекст ветки «${session.currentBranch}»:\n${branch.summary}`
      : CONFIG.system;
    return [{ role: "system", content: systemContent }, ...branch.messages];
  }

  throw new Error(`Unknown strategy: ${strategy}`);
}

// ─── Post-process after getting the assistant response ─────────────────────────
async function postProcess(session) {
  const { strategy } = session;

  if (strategy === "sliding-window") {
    // Discard everything except the last N messages — no summary kept
    session.messages = session.messages.slice(-CONFIG.keepRecent);
    session.summary  = null;
    return;
  }

  if (strategy === "sticky-facts") {
    // Update facts from the full exchange (user + assistant turn just added)
    session.facts = await extractFacts(session.messages, session.facts ?? {});
    // Cap raw history to avoid unbounded growth (context always uses last N anyway)
    if (session.messages.length > CONFIG.keepRecent * 5) {
      session.messages = session.messages.slice(-CONFIG.keepRecent * 5);
    }
    return;
  }

  if (strategy === "branching") {
    // Compress branch when it grows too large
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

  const session = loadSession();

  // Append user message to the right place
  if (session.strategy === "branching") {
    getActiveBranch(session).messages.push({ role: "user", content: userQuery });
  } else {
    session.messages.push({ role: "user", content: userQuery });
  }

  const contextMessages = buildContext(session);

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

  // Append assistant response
  if (session.strategy === "branching") {
    getActiveBranch(session).messages.push({ role: "assistant", content: result });
  } else {
    session.messages.push({ role: "assistant", content: result });
  }

  // Apply strategy-specific post-processing (trim / compress / extract facts)
  await postProcess(session);

  // Record token usage
  const requestEntry = {
    timestamp:        new Date().toISOString(),
    promptTokens:     usage?.prompt_tokens     ?? null,
    completionTokens: usage?.completion_tokens ?? null,
    totalTokens:      usage?.total_tokens      ?? null,
  };
  session.requests.push(requestEntry);
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

  // --clear  Reset everything
  if (args[0] === "--clear") {
    saveSession(defaultSession());
    console.log("История очищена.");
    process.exit(0);
  }

  // --strategy <name>  Switch context management strategy
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

  // --branch <sub>  Branch management (branching strategy only)
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
      // --branch create <имя> [<checkpoint>]
      // If <checkpoint> is given, fork from that saved snapshot.
      // Otherwise fork from the current active branch state.
      const newName  = args[2];
      const fromCp   = args[3]; // optional checkpoint name
      if (!newName) { console.error("Укажите имя ветки."); process.exit(1); }
      if (session.branches[newName]) { console.error(`Ветка "${newName}" уже существует.`); process.exit(1); }

      let src;
      if (fromCp) {
        src = session.checkpoints[fromCp];
        if (!src) { console.error(`Checkpoint "${fromCp}" не найден. Список: --checkpoint list`); process.exit(1); }
      } else {
        src = getActiveBranch(session);
      }

      session.branches[newName] = {
        messages:   JSON.parse(JSON.stringify(src.messages)),
        summary:    src.summary ?? null,
        checkpoint: fromCp ?? null, // remember the origin for display
      };
      session.currentBranch = newName;
      saveSession(session);
      const origin = fromCp ? `checkpoint «${fromCp}»` : `ветки «${Object.keys(session.branches).find(k => session.branches[k] === src) ?? "current"}»`;
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
      if (!name)              { console.error("Укажите имя ветки."); process.exit(1); }
      if (name === "main")    { console.error("Нельзя удалить ветку main."); process.exit(1); }
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

  // --checkpoint <sub>  Save/list/delete named snapshots (branching strategy only)
  if (args[0] === "--checkpoint") {
    const session = loadSession();
    if (session.strategy !== "branching") {
      console.error("Checkpoints доступны только в стратегии branching.");
      console.error("Переключитесь: node agent.js --strategy branching");
      process.exit(1);
    }
    const sub = args[1];

    if (sub === "save") {
      const cpName = args[2];
      if (!cpName) { console.error("Укажите имя checkpoint."); process.exit(1); }
      const branch = getActiveBranch(session);
      session.checkpoints[cpName] = {
        messages:  JSON.parse(JSON.stringify(branch.messages)),
        summary:   branch.summary ?? null,
        savedAt:   new Date().toISOString(),
        fromBranch: session.currentBranch,
      };
      saveSession(session);
      console.log(`Checkpoint "${cpName}" сохранён (${branch.messages.length} сообщений из ветки «${session.currentBranch}»).`);
      process.exit(0);
    }

    if (sub === "list") {
      const cps = Object.entries(session.checkpoints);
      if (!cps.length) { console.log("Нет сохранённых checkpoints."); process.exit(0); }
      cps.forEach(([name, cp]) => {
        console.log(`  ${name}  (${cp.messages.length} сообщ. | из ветки «${cp.fromBranch}» | ${cp.savedAt})`);
      });
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

  // --facts  Show extracted facts (sticky-facts strategy only)
  if (args[0] === "--facts") {
    const session = loadSession();
    if (session.strategy !== "sticky-facts") {
      console.error("Факты доступны только в стратегии sticky-facts.");
      console.error("Переключитесь: node agent.js --strategy sticky-facts");
      process.exit(1);
    }
    const facts = session.facts ?? {};
    if (!Object.keys(facts).length) {
      console.log("Факты ещё не извлечены (пока нет завершённых обменов).");
    } else {
      console.log("Ключевые факты диалога:\n");
      Object.entries(facts).forEach(([k, v]) => console.log(`  ${k}: ${v}`));
    }
    process.exit(0);
  }

  // --summary  Show current compression summary
  if (args[0] === "--summary") {
    const session = loadSession();
    let summary;
    if (session.strategy === "branching") {
      summary = getActiveBranch(session).summary;
    } else {
      summary = session.summary;
    }
    if (summary) {
      console.log("Текущее резюме:\n");
      console.log(summary);
    } else {
      console.log("Резюме ещё нет.");
    }
    process.exit(0);
  }

  // --stats  Token usage statistics
  if (args[0] === "--stats") {
    const { totalTokens, requests } = getTokenStats();
    console.log(`Токенов за сессию: ${totalTokens}`);
    console.log(`Запросов: ${requests.length}`);
    requests.forEach((r, i) =>
      console.log(`  [${i + 1}] ${r.timestamp}  prompt=${r.promptTokens ?? "?"}  completion=${r.completionTokens ?? "?"}  total=${r.totalTokens ?? "?"}`)
    );
    process.exit(0);
  }

  // --status  Show current strategy and session info
  if (args[0] === "--status") {
    const session = loadSession();
    const s = session.strategy;
    console.log(`Стратегия: ${s}`);

    if (s === "sliding-window") {
      console.log(`Окно: последние ${CONFIG.keepRecent} сообщений (старые отбрасываются)`);
      console.log(`Сообщений в истории: ${session.messages.length}`);
    }

    if (s === "sticky-facts") {
      console.log(`Окно: последние ${CONFIG.keepRecent} сообщений + факты`);
      console.log(`Сообщений в истории: ${session.messages.length}`);
      const n = Object.keys(session.facts ?? {}).length;
      console.log(`Фактов в памяти: ${n}`);
    }

    if (s === "branching") {
      console.log(`Веток: ${Object.keys(session.branches).length}`);
      console.log(`Checkpoints: ${Object.keys(session.checkpoints ?? {}).length}`);
      console.log(`Активная ветка: ${session.currentBranch}`);
      const b = getActiveBranch(session);
      console.log(`Сообщений в ветке: ${b.messages.length}${b.checkpoint ? ` (создана от checkpoint «${b.checkpoint}»)` : ""}`);
    }

    process.exit(0);
  }

  // Default: treat all args as a query
  const query = args.join(" ");
  if (!query) {
    console.error([
      "Использование: node agent.js \"<вопрос>\"",
      "",
      "Стратегии управления контекстом:",
      "  --strategy sliding-window   Скользящее окно: хранит только последние N сообщений",
      "  --strategy sticky-facts     Факты + окно: ключевые данные + последние N сообщений",
      "  --strategy branching        Ветки: независимые ветки диалога",
      "",
      "Ветки (только стратегия branching):",
      "  --checkpoint save <имя>     Сохранить текущее состояние как checkpoint",
      "  --checkpoint list           Список checkpoints",
      "  --checkpoint delete <имя>   Удалить checkpoint",
      "  --branch create <имя> [<cp>]  Создать ветку (от checkpoint или текущей) и переключиться",
      "  --branch switch <имя>       Переключиться на ветку",
      "  --branch list               Список всех веток",
      "  --branch delete <имя>       Удалить ветку",
      "",
      "Прочее:",
      "  --status                    Текущая стратегия и состояние сессии",
      "  --facts                     Показать факты (sticky-facts)",
      "  --summary                   Показать резюме (branching / после сжатия)",
      "  --stats                     Статистика токенов",
      "  --clear                     Полный сброс истории",
    ].join("\n"));
    process.exit(1);
  }

  ask(query)
    .then((answer) => {
      console.log(answer);
      const session = loadSession();
      const { totalTokens, requests } = getTokenStats();
      const last = requests.at(-1);
      const extra = session.strategy === "branching" ? ` | ветка: ${session.currentBranch}` : "";
      if (last?.totalTokens != null) {
        console.log(`\n[токены: ${last.totalTokens} | сессия: ${totalTokens} | стратегия: ${session.strategy}${extra}]`);
      }
    })
    .catch((err) => { console.error(err.message); process.exit(1); });
}
