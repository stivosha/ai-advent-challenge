import { randomUUID } from "crypto";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// GigaChat uses a certificate signed by Russian NCA — disable TLS verification for dev.
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

// ─── Config ────────────────────────────────────────────────────────────────────
const CONFIG = {
  model:       "GigaChat",
  temperature: 0.7,
  max_tokens:  1024,
  top_p:       0.9,
};

// ─── File paths ────────────────────────────────────────────────────────────────
const __dir            = dirname(fileURLToPath(import.meta.url));
const TASK_FILE        = join(__dir, "task.json");
const HISTORY_FILE     = join(__dir, "history.json");
const INVARIANTS_FILE  = join(__dir, "invariants.json");

// ══════════════════════════════════════════════════════════════════════════════
// КОНЕЧНЫЙ АВТОМАТ (FSM) — СОСТОЯНИЕ ЗАДАЧИ
//
// Жизненный цикл задачи:
//   planning → execution → validation → done
//
// Каждое состояние содержит:
//   phase          — текущий этап задачи
//   step           — текущий шаг внутри этапа
//   expectedAction — что ожидается от агента/пользователя сейчас
//   history        — лог всех переходов состояний
// ══════════════════════════════════════════════════════════════════════════════

// Допустимые переходы FSM
const FSM_TRANSITIONS = {
  planning:   ["execution"],
  execution:  ["validation", "planning"],  // planning — если нужно переплановать
  validation: ["done", "execution"],        // execution — если валидация не прошла
  done:       ["planning"],                 // planning — начать новую задачу
};

// Что ожидается на каждом этапе
const PHASE_PROMPTS = {
  planning: {
    system: `Ты — агент-планировщик. Ты находишься на этапе ПЛАНИРОВАНИЯ.
Твоя задача: разбить задачу пользователя на конкретные шаги.
Сформируй чёткий план выполнения: пронумерованный список шагов.
В конце ответа явно укажи: "ПЛАН ГОТОВ: <N> шагов"`,
    expectedAction: "Сформировать план выполнения задачи",
  },
  execution: {
    system: `Ты — агент-исполнитель. Ты находишься на этапе ВЫПОЛНЕНИЯ.
Выполняй задачу строго по плану, шаг за шагом.
Для каждого ответа сообщай: какой шаг выполняешь, что сделано, что осталось.
Когда все шаги выполнены, явно укажи: "ВЫПОЛНЕНИЕ ЗАВЕРШЕНО"`,
    expectedAction: "Выполнить текущий шаг плана",
  },
  validation: {
    system: `Ты — агент-валидатор. Ты находишься на этапе ВАЛИДАЦИИ.
Проверь результаты выполнения: соответствуют ли они исходной задаче?
Выяви проблемы, неточности или пропущенные требования.
По итогу явно укажи: "ВАЛИДАЦИЯ ПРОЙДЕНА" или "ВАЛИДАЦИЯ НЕ ПРОЙДЕНА: <причина>"`,
    expectedAction: "Проверить результат на соответствие требованиям",
  },
  done: {
    system: `Ты — агент завершения задачи. Задача ВЫПОЛНЕНА.
Подведи итог: что было сделано, каковы результаты.
Предложи пользователю начать новую задачу командой: --task new <описание>`,
    expectedAction: "Подвести итог выполненной задачи",
  },
};

// ─── FSM: структура задачи ────────────────────────────────────────────────────
function defaultTask() {
  return {
    id:             null,
    description:    null,
    phase:          "planning",
    step:           0,
    totalSteps:     null,
    expectedAction: PHASE_PROMPTS.planning.expectedAction,
    plan:           [],           // шаги плана, заполняется на этапе planning
    validationResult: null,       // результат последней валидации
    createdAt:      null,
    updatedAt:      null,
    history:        [],           // лог переходов: { from, to, reason, at }
  };
}

function loadTask() {
  try {
    if (!existsSync(TASK_FILE)) return defaultTask();
    const raw = JSON.parse(readFileSync(TASK_FILE, "utf8"));
    const t = defaultTask();
    Object.assign(t, raw);
    if (!t.history)  t.history  = [];
    if (!t.plan)     t.plan     = [];
    return t;
  } catch {
    return defaultTask();
  }
}

function saveTask(task) {
  task.updatedAt = new Date().toISOString();
  writeFileSync(TASK_FILE, JSON.stringify(task, null, 2), "utf8");
}

// ─── FSM: переход в новое состояние ──────────────────────────────────────────
function transition(task, toPhase, reason = "") {
  const allowed = FSM_TRANSITIONS[task.phase];
  if (!allowed || !allowed.includes(toPhase)) {
    throw new Error(
      `Недопустимый переход: ${task.phase} → ${toPhase}. ` +
      `Разрешённые: ${(allowed ?? []).join(", ") || "нет"}`
    );
  }

  task.history.push({
    from:   task.phase,
    to:     toPhase,
    reason: reason || `Переход ${task.phase} → ${toPhase}`,
    at:     new Date().toISOString(),
  });

  task.phase          = toPhase;
  task.step           = 0;
  task.expectedAction = PHASE_PROMPTS[toPhase].expectedAction;
}

// ─── FSM: текстовое описание текущего состояния ───────────────────────────────
function taskStateToText(task) {
  if (!task.id) return "";

  const lines = [
    "[Состояние задачи]",
    `  Этап:              ${task.phase.toUpperCase()}`,
    `  Текущий шаг:       ${task.step}${task.totalSteps ? ` / ${task.totalSteps}` : ""}`,
    `  Ожидаемое действие: ${task.expectedAction}`,
  ];

  if (task.description) lines.push(`  Задача:            ${task.description}`);

  if (task.plan.length) {
    lines.push("  План:");
    task.plan.forEach((s, i) => {
      const marker = i < task.step ? "✓" : i === task.step ? "→" : " ";
      lines.push(`    ${marker} ${i + 1}. ${s}`);
    });
  }

  if (task.validationResult) lines.push(`  Последняя валидация: ${task.validationResult}`);

  return lines.join("\n");
}

// ─── FSM: автообновление шага на основе ответа ────────────────────────────────
// Анализирует ответ ассистента и автоматически обновляет состояние FSM.
function autoAdvanceFSM(task, assistantReply) {
  const reply = assistantReply.toLowerCase();

  if (task.phase === "planning") {
    // Ищем "план готов: N шагов" или нумерованный список
    const planMatch = assistantReply.match(/план готов[:\s]+(\d+)\s*шаг/i);
    if (planMatch) {
      task.totalSteps = parseInt(planMatch[1], 10);
      // Извлекаем шаги из ответа (строки вида "1. ...", "2. ...")
      const steps = [...assistantReply.matchAll(/^\s*(\d+)\.\s+(.+)$/gm)]
        .map(m => m[2].trim());
      if (steps.length) task.plan = steps;
    }
  }

  if (task.phase === "execution") {
    if (reply.includes("выполнение завершено")) {
      transition(task, "validation", "Агент сообщил о завершении выполнения");
    }
  }

  if (task.phase === "validation") {
    if (reply.includes("валидация пройдена")) {
      transition(task, "done", "Валидация успешно пройдена");
    } else if (reply.includes("валидация не пройдена")) {
      const reason = assistantReply.match(/валидация не пройдена[:\s]+(.+)/i)?.[1] ?? "";
      transition(task, "execution", `Валидация не пройдена: ${reason}`);
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// ИНВАРИАНТЫ
//
// Инварианты — это жёсткие ограничения проекта, которые ассистент обязан
// учитывать и не имеет права предлагать решения, их нарушающие.
//
// Категории:
//   architecture  — выбранная архитектура
//   decision      — принятые технические решения
//   stack         — ограничения по стеку технологий
//   business      — бизнес-правила
// ══════════════════════════════════════════════════════════════════════════════

const INVARIANT_CATEGORIES = ["architecture", "decision", "stack", "business"];

const CATEGORY_LABELS = {
  architecture: "Архитектура",
  decision:     "Технические решения",
  stack:        "Стек технологий",
  business:     "Бизнес-правила",
};

function defaultInvariants() {
  return { items: [] };
}

function loadInvariants() {
  try {
    if (!existsSync(INVARIANTS_FILE)) return defaultInvariants();
    const raw = JSON.parse(readFileSync(INVARIANTS_FILE, "utf8"));
    const inv = defaultInvariants();
    Object.assign(inv, raw);
    if (!Array.isArray(inv.items)) inv.items = [];
    return inv;
  } catch {
    return defaultInvariants();
  }
}

function saveInvariants(inv) {
  writeFileSync(INVARIANTS_FILE, JSON.stringify(inv, null, 2), "utf8");
}

// Форматирует инварианты для системного промпта
function invariantsToSystemPrompt(inv) {
  if (!inv.items.length) return "";

  const grouped = {};
  for (const item of inv.items) {
    const cat = item.category ?? "decision";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(item);
  }

  const lines = [
    "━━━ ИНВАРИАНТЫ ПРОЕКТА (ЖЁСТКИЕ ОГРАНИЧЕНИЯ) ━━━",
    "Следующие инварианты установлены для этого проекта.",
    "Ты ОБЯЗАН явно учитывать их при планировании и выполнении.",
    "Ты НЕ ИМЕЕШЬ ПРАВА предлагать решения, которые нарушают хотя бы один инвариант.",
    "Если запрос пользователя противоречит инварианту — чётко укажи: какой инвариант нарушается и почему.",
    "",
  ];

  for (const cat of INVARIANT_CATEGORIES) {
    if (!grouped[cat]) continue;
    lines.push(`[${CATEGORY_LABELS[cat]}]`);
    grouped[cat].forEach((item, idx) => {
      lines.push(`  ${idx + 1}. ${item.text}`);
    });
    lines.push("");
  }

  lines.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  return lines.join("\n");
}

// Форматирует инварианты для CLI вывода
function invariantsToText(inv) {
  if (!inv.items.length) return "Инварианты не заданы.";

  const grouped = {};
  for (const item of inv.items) {
    const cat = item.category ?? "decision";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(item);
  }

  const lines = ["[Инварианты проекта]"];
  for (const cat of INVARIANT_CATEGORIES) {
    if (!grouped[cat]) continue;
    lines.push(`\n  ${CATEGORY_LABELS[cat]}:`);
    grouped[cat].forEach(item => {
      const at = item.createdAt ? new Date(item.createdAt).toLocaleString("ru-RU") : "";
      lines.push(`    #${item.id}  ${item.text}  (${at})`);
    });
  }
  return lines.join("\n");
}

// ══════════════════════════════════════════════════════════════════════════════
// ИСТОРИЯ ДИАЛОГА
// ══════════════════════════════════════════════════════════════════════════════

function defaultHistory() {
  return { messages: [], turn: 0, totalTokens: 0, requests: [] };
}

function loadHistory() {
  try {
    if (!existsSync(HISTORY_FILE)) return defaultHistory();
    const raw = JSON.parse(readFileSync(HISTORY_FILE, "utf8"));
    const h = defaultHistory();
    Object.assign(h, raw);
    return h;
  } catch {
    return defaultHistory();
  }
}

function saveHistory(history) {
  writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2), "utf8");
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

// ─── Сборка контекста для API ──────────────────────────────────────────────────
// Системный промпт меняется в зависимости от текущего этапа FSM.
// Инварианты инжектируются первым блоком — до инструкций фазы.
function buildContext(history, task) {
  const phaseConfig = PHASE_PROMPTS[task.phase] ?? PHASE_PROMPTS.planning;
  const inv = loadInvariants();

  const systemParts = [];

  // Инварианты идут первыми — они главнее инструкций фазы
  const invText = invariantsToSystemPrompt(inv);
  if (invText) systemParts.push(invText);

  systemParts.push(phaseConfig.system);

  const stateText = taskStateToText(task);
  if (stateText) systemParts.push("\n" + stateText);

  const systemContent = systemParts.join("\n\n");

  // Последние 10 сообщений для контекста
  const recent = history.messages.slice(-10);

  return [{ role: "system", content: systemContent }, ...recent];
}

// ─── Основная функция: отправить сообщение ────────────────────────────────────
export async function ask(userQuery) {
  if (!userQuery?.trim()) throw new Error("Запрос не может быть пустым");

  const history = loadHistory();
  const task    = loadTask();

  if (!task.id) {
    throw new Error(
      'Задача не создана. Используйте: node agent.js --task new "<описание задачи>"'
    );
  }

  history.turn = (history.turn ?? 0) + 1;
  const turn = history.turn;

  history.messages.push({
    role:    "user",
    content: userQuery,
    turn,
    phase:   task.phase,
    at:      new Date().toISOString(),
  });

  // Если на этапе planning и пользователь прямо просит начать выполнение
  if (task.phase === "planning" && /начин|выполн|старт|go|start/i.test(userQuery)) {
    if (task.plan.length > 0) {
      transition(task, "execution", "Пользователь инициировал выполнение");
    }
  }

  const contextMessages = buildContext(history, task);

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

  // Стриминг ответа
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

  history.messages.push({
    role:    "assistant",
    content: result,
    turn,
    phase:   task.phase,
    at:      new Date().toISOString(),
  });

  // Автоматически обновляем FSM на основе ответа
  autoAdvanceFSM(task, result);

  // Если на этапе execution — двигаем шаг вперёд
  if (task.phase === "execution") {
    task.step = Math.min(task.step + 1, task.totalSteps ?? task.step + 1);
  }

  // Статистика токенов
  history.requests.push({
    timestamp:        new Date().toISOString(),
    turn,
    phase:            task.phase,
    promptTokens:     usage?.prompt_tokens     ?? null,
    completionTokens: usage?.completion_tokens ?? null,
    totalTokens:      usage?.total_tokens      ?? null,
  });
  if (usage?.total_tokens) history.totalTokens += usage.total_tokens;

  saveHistory(history);
  saveTask(task);

  return result;
}

// ══════════════════════════════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════════════════════════════

if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replace(/\\/g, "/"))) {
  const args = process.argv.slice(2);

  // ── --task  Управление задачей и FSM ─────────────────────────────────────────
  if (args[0] === "--task") {
    const sub  = args[1];

    // --task new <описание>  Создать новую задачу
    if (sub === "new") {
      const description = args.slice(2).join(" ");
      if (!description) {
        console.error('Использование: --task new "<описание задачи>"');
        process.exit(1);
      }
      const task = defaultTask();
      task.id          = randomUUID();
      task.description = description;
      task.phase       = "planning";
      task.step        = 0;
      task.expectedAction = PHASE_PROMPTS.planning.expectedAction;
      task.createdAt   = new Date().toISOString();
      saveTask(task);
      // Сбросить историю диалога при новой задаче
      saveHistory(defaultHistory());
      console.log(`Задача создана: "${description}"`);
      console.log(`Этап: planning → Ожидаемое действие: ${task.expectedAction}`);
      console.log('\nНачните диалог, чтобы агент сформировал план. Например:');
      console.log('  node agent.js "Как выполнить задачу?"');
      process.exit(0);
    }

    // --task status  Показать текущее состояние FSM
    if (sub === "status") {
      const task = loadTask();
      if (!task.id) {
        console.log('Задача не создана. Используйте: --task new "<описание>"');
        process.exit(0);
      }
      console.log(taskStateToText(task));
      if (task.history.length) {
        console.log("\n[История переходов]");
        task.history.forEach(h => {
          const at = h.at ? new Date(h.at).toLocaleString("ru-RU") : "";
          console.log(`  ${h.from} → ${h.to}: ${h.reason} (${at})`);
        });
      }
      process.exit(0);
    }

    // --task advance [<этап>] [--reason <причина>]  Принудительный переход
    if (sub === "advance") {
      const task    = loadTask();
      if (!task.id) { console.error("Задача не создана."); process.exit(1); }

      const reasonIdx = args.indexOf("--reason");
      const reason    = reasonIdx !== -1 ? args.slice(reasonIdx + 1).join(" ") : "";

      // Найти целевой этап: либо указан явно, либо берём первый разрешённый
      let toPhase = args[2];
      if (!toPhase || toPhase.startsWith("--")) {
        const allowed = FSM_TRANSITIONS[task.phase];
        if (!allowed?.length) { console.error(`Нет доступных переходов из "${task.phase}".`); process.exit(1); }
        toPhase = allowed[0];
      }

      try {
        transition(task, toPhase, reason || `Принудительный переход (CLI)`);
        saveTask(task);
        console.log(`Переход выполнен: ${task.history.at(-1).from} → ${task.phase}`);
        console.log(`Ожидаемое действие: ${task.expectedAction}`);
      } catch (err) {
        console.error(err.message);
        process.exit(1);
      }
      process.exit(0);
    }

    // --task reset  Сбросить задачу и историю
    if (sub === "reset") {
      saveTask(defaultTask());
      saveHistory(defaultHistory());
      console.log("Задача и история диалога сброшены.");
      process.exit(0);
    }

    // --task fsm  Показать граф переходов
    if (sub === "fsm") {
      console.log("[Граф переходов FSM]");
      for (const [from, tos] of Object.entries(FSM_TRANSITIONS)) {
        tos.forEach(to => console.log(`  ${from} → ${to}`));
      }
      console.log("\n[Описание этапов]");
      for (const [phase, cfg] of Object.entries(PHASE_PROMPTS)) {
        console.log(`  ${phase}: ${cfg.expectedAction}`);
      }
      process.exit(0);
    }

    console.error([
      "Управление задачей (конечный автомат):",
      '  --task new "<описание>"          Создать новую задачу (сбрасывает историю)',
      "  --task status                    Показать текущее состояние FSM",
      "  --task advance [<этап>] [--reason <причина>]  Перейти к следующему этапу",
      "  --task reset                     Сбросить задачу и историю",
      "  --task fsm                       Показать граф переходов",
    ].join("\n"));
    process.exit(1);
  }

  // ── --invariant  Управление инвариантами ──────────────────────────────────────
  if (args[0] === "--invariant") {
    const sub = args[1];

    // --invariant add [--category <cat>] <текст>
    if (sub === "add") {
      const catIdx  = args.indexOf("--category");
      let category  = catIdx !== -1 ? args[catIdx + 1] : "decision";
      if (!INVARIANT_CATEGORIES.includes(category)) {
        console.error(`Неизвестная категория: "${category}". Допустимые: ${INVARIANT_CATEGORIES.join(", ")}`);
        process.exit(1);
      }
      // Текст — всё, что не является флагом
      const textArgs = args.slice(2).filter((a, i, arr) =>
        !(a.startsWith("--")) && !(arr[i - 1] === "--category")
      );
      const text = textArgs.join(" ").trim();
      if (!text) {
        console.error('Использование: --invariant add [--category <cat>] "<текст>"');
        console.error(`Категории: ${INVARIANT_CATEGORIES.join(", ")}`);
        process.exit(1);
      }
      const inv = loadInvariants();
      const nextId = (inv.items.reduce((max, it) => Math.max(max, it.id ?? 0), 0)) + 1;
      inv.items.push({ id: nextId, category, text, createdAt: new Date().toISOString() });
      saveInvariants(inv);
      console.log(`Инвариант #${nextId} добавлен [${CATEGORY_LABELS[category]}]: ${text}`);
      process.exit(0);
    }

    // --invariant list
    if (sub === "list") {
      const inv = loadInvariants();
      console.log(invariantsToText(inv));
      process.exit(0);
    }

    // --invariant remove <id>
    if (sub === "remove") {
      const id  = parseInt(args[2], 10);
      if (isNaN(id)) {
        console.error("Использование: --invariant remove <id>");
        process.exit(1);
      }
      const inv = loadInvariants();
      const before = inv.items.length;
      inv.items = inv.items.filter(it => it.id !== id);
      if (inv.items.length === before) {
        console.error(`Инвариант #${id} не найден.`);
        process.exit(1);
      }
      saveInvariants(inv);
      console.log(`Инвариант #${id} удалён.`);
      process.exit(0);
    }

    console.error([
      "Управление инвариантами:",
      '  --invariant add [--category <cat>] "<текст>"   Добавить инвариант',
      "  --invariant list                               Показать все инварианты",
      "  --invariant remove <id>                        Удалить инвариант по ID",
      "",
      `  Категории: ${INVARIANT_CATEGORIES.join(", ")}`,
    ].join("\n"));
    process.exit(1);
  }

  // ── --clear  Сбросить историю диалога (задача сохраняется) ───────────────────
  if (args[0] === "--clear") {
    saveHistory(defaultHistory());
    console.log("История диалога очищена. Задача и её состояние сохранены.");
    console.log("Для сброса задачи: --task reset");
    process.exit(0);
  }

  // ── Обычный вопрос ────────────────────────────────────────────────────────────
  const query = args.join(" ");
  if (!query.trim()) {
    console.error([
      "Использование:",
      '  node agent.js "<вопрос>"                              Отправить сообщение агенту',
      '  node agent.js --task new "<описание>"                 Создать новую задачу',
      "  node agent.js --task status                           Статус задачи и FSM",
      "  node agent.js --task advance                          Перейти к следующему этапу",
      "  node agent.js --task reset                            Сбросить задачу",
      "  node agent.js --task fsm                              Показать граф переходов",
      '  node agent.js --invariant add [--category <cat>] "<текст>"  Добавить инвариант',
      "  node agent.js --invariant list                        Показать инварианты",
      "  node agent.js --invariant remove <id>                 Удалить инвариант",
      "  node agent.js --clear                                 Очистить историю диалога",
    ].join("\n"));
    process.exit(1);
  }

  const task = loadTask();
  console.log(`[${task.phase?.toUpperCase() ?? "NO TASK"} | шаг ${task.step}] ${task.expectedAction ?? ""}\n`);

  ask(query)
    .then(reply => {
      console.log(reply);

      // Показать новое состояние после ответа
      const updated = loadTask();
      if (updated.phase !== task.phase) {
        console.log(`\n── Переход FSM: ${task.phase} → ${updated.phase} ──`);
        console.log(`   Ожидаемое действие: ${updated.expectedAction}`);
      }
    })
    .catch(err => { console.error("Ошибка:", err.message); process.exit(1); });
}
