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
const __dir       = dirname(fileURLToPath(import.meta.url));
const TASK_FILE   = join(__dir, "task.json");
const HISTORY_FILE = join(__dir, "history.json");

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
}

// ══════════════════════════════════════════════════════════════════════════════
// КОНЕЧНЫЙ АВТОМАТ (FSM)
//
// Жизненный цикл задачи:
//
//   planning ──(approve-plan)──► approved ──(start)──► execution
//      ▲                                                    │
//      │                                                    │ (execution завершено)
//      │                                              validation
//      │                                                    │
//      └──────────────(новая задача)────────────────── done ◄┘
//                                                           │
//                                       execution ◄─ (fail)─┘
//
// Правила (guard-условия):
//   • planning → approved   требует: plan.length > 0
//   • approved → execution  всегда разрешён (план уже утверждён)
//   • execution → validation всегда разрешён
//   • validation → done     требует: validationPassed === true
//   • validation → execution требует: validationPassed === false
//   • done → planning       всегда разрешён (начать новую)
//
// Заблокированные "перепрыгивания":
//   ✗  planning → execution  (пропуск approved)
//   ✗  planning → validation (пропуск approved + execution)
//   ✗  execution → done      (пропуск validation)
// ══════════════════════════════════════════════════════════════════════════════

// Допустимые переходы
const FSM_TRANSITIONS = {
  planning:   ["approved"],
  approved:   ["execution"],
  execution:  ["validation"],
  validation: ["done", "execution"],
  done:       ["planning"],
};

// Guard-условия: бросают Error если условие не выполнено
const TRANSITION_GUARDS = {
  "planning->approved": (task) => {
    if (!task.plan.length) {
      throw new Error(
        "Нельзя утвердить пустой план.\n" +
        "  Сначала попросите агента сформировать план: node agent.js \"составь план\"\n" +
        "  Агент ответит списком шагов, после чего выполните: node agent.js --task approve-plan"
      );
    }
  },
  "validation->done": (task) => {
    if (!task.validationPassed) {
      throw new Error(
        "Нельзя завершить задачу без успешной валидации.\n" +
        "  Дождитесь, когда агент напишет \"ВАЛИДАЦИЯ ПРОЙДЕНА\".\n" +
        "  Для ручного теста: node agent.js --task mark-validated"
      );
    }
  },
  "validation->execution": (task) => {
    if (task.validationPassed) {
      throw new Error(
        "Валидация уже пройдена — нельзя вернуться на выполнение.\n" +
        "  Выполните: node agent.js --task advance done"
      );
    }
  },
};

// Что ожидается на каждом этапе
const PHASE_PROMPTS = {
  planning: {
    system: `Ты — агент-планировщик. Этап: ПЛАНИРОВАНИЕ.

Твоя задача: разбить запрос пользователя на конкретные пронумерованные шаги.
Формат: пронумерованный список, каждый шаг — конкретное действие.
В конце явно напиши: "ПЛАН ГОТОВ: <N> шагов"

ВАЖНО: После составления плана напомни, что:
  • для утверждения плана нужно выполнить: node agent.js --task approve-plan
  • без утверждения переход к выполнению НЕВОЗМОЖЕН
НЕ НАЧИНАЙ выполнение самостоятельно.`,
    expectedAction: "Сформировать план и ожидать утверждения",
  },

  approved: {
    system: `Ты — агент-исполнитель. Этап: ПЛАН УТВЕРЖДЁН, ожидание старта.

План пользователя утверждён. Подтверди готовность к выполнению.
Напомни пользователю запустить выполнение: node agent.js --task start
Или, если пользователь пишет "начинай" / "start" — переход произойдёт автоматически.`,
    expectedAction: "Ожидать команды старта выполнения",
  },

  execution: {
    system: `Ты — агент-исполнитель. Этап: ВЫПОЛНЕНИЕ.

Выполняй задачу строго по утверждённому плану, шаг за шагом.
Для каждого ответа указывай: шаг N из M — что сделано — что осталось.
Когда все шаги выполнены, явно напиши: "ВЫПОЛНЕНИЕ ЗАВЕРШЕНО"`,
    expectedAction: "Выполнить следующий шаг плана",
  },

  validation: {
    system: `Ты — агент-валидатор. Этап: ВАЛИДАЦИЯ.

Проверь результаты выполнения на соответствие исходной задаче.
Выяви ошибки, пропуски и несоответствия требованиям.
Итог ОБЯЗАТЕЛЬНО в одной из форм:
  "ВАЛИДАЦИЯ ПРОЙДЕНА" — если всё соответствует требованиям
  "ВАЛИДАЦИЯ НЕ ПРОЙДЕНА: <причина>" — если есть проблемы`,
    expectedAction: "Проверить результат на соответствие требованиям",
  },

  done: {
    system: `Ты — агент завершения. Этап: ЗАДАЧА ВЫПОЛНЕНА.

Подведи итог: что было сделано, каковы результаты.
Предложи пользователю начать новую задачу: node agent.js --task new "<описание>"`,
    expectedAction: "Подвести итог выполненной задачи",
  },
};

// ─── Структура задачи ─────────────────────────────────────────────────────────
function defaultTask() {
  return {
    id:               null,
    description:      null,
    phase:            "planning",
    step:             0,
    totalSteps:       null,
    expectedAction:   PHASE_PROMPTS.planning.expectedAction,
    plan:             [],
    validationResult: null,
    validationPassed: false,
    createdAt:        null,
    updatedAt:        null,
    history:          [],
  };
}

function loadTask() {
  try {
    if (!existsSync(TASK_FILE)) return defaultTask();
    const raw = readJson(TASK_FILE);
    const t = defaultTask();
    Object.assign(t, raw);
    if (!t.history) t.history = [];
    if (!t.plan)    t.plan    = [];
    return t;
  } catch {
    return defaultTask();
  }
}

function saveTask(task) {
  task.updatedAt = new Date().toISOString();
  writeFileSync(TASK_FILE, JSON.stringify(task, null, 2), "utf8");
}

// ─── FSM: переход состояния ───────────────────────────────────────────────────
function transition(task, toPhase, reason = "") {
  const allowed = FSM_TRANSITIONS[task.phase];
  if (!allowed || !allowed.includes(toPhase)) {
    const allowedStr = (allowed ?? []).join(", ") || "нет";
    throw new Error(
      `Переход ${task.phase} → ${toPhase} запрещён.\n` +
      `  Текущий этап: ${task.phase}\n` +
      `  Допустимые переходы: ${allowedStr}`
    );
  }

  const guardKey = `${task.phase}->${toPhase}`;
  if (TRANSITION_GUARDS[guardKey]) {
    TRANSITION_GUARDS[guardKey](task);
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

  if (toPhase === "planning") {
    task.validationPassed = false;
    task.plan             = [];
  }
}

// ─── Автообновление FSM на основе ответа ────────────────────────────────────
function autoAdvanceFSM(task, assistantReply) {
  const reply = assistantReply.toLowerCase();

  if (task.phase === "planning") {
    const planMatch = assistantReply.match(/план готов[:\s]+(\d+)\s*шаг/i);
    if (planMatch) {
      task.totalSteps = parseInt(planMatch[1], 10);
      const steps = [...assistantReply.matchAll(/^\s*(\d+)\.\s+(.+)$/gm)].map(m => m[2].trim());
      if (steps.length) task.plan = steps;
    }
  }

  if (task.phase === "execution") {
    if (reply.includes("выполнение завершено")) {
      transition(task, "validation", "Агент: выполнение завершено");
    }
  }

  if (task.phase === "validation") {
    if (reply.includes("валидация пройдена") && !reply.includes("не пройдена")) {
      task.validationPassed = true;
      task.validationResult = "Пройдена";
      transition(task, "done", "Агент: валидация пройдена");
    } else if (reply.includes("валидация не пройдена")) {
      task.validationPassed = false;
      const reason = assistantReply.match(/валидация не пройдена[:\s]+(.+)/i)?.[1] ?? "";
      task.validationResult = reason || "Не пройдена";
      transition(task, "execution", `Агент: валидация не пройдена — ${reason}`);
    }
  }
}

// ─── Текстовое состояние ──────────────────────────────────────────────────────
const PHASE_LABELS = {
  planning:   "ПЛАНИРОВАНИЕ",
  approved:   "ПЛАН УТВЕРЖДЁН",
  execution:  "ВЫПОЛНЕНИЕ",
  validation: "ВАЛИДАЦИЯ",
  done:       "ВЫПОЛНЕНО",
};

function taskStateToText(task) {
  if (!task.id) return "";

  const lines = [
    "┌─ Состояние задачи ─────────────────────────────────",
    `│  Этап:    ${PHASE_LABELS[task.phase] ?? task.phase}`,
    `│  Шаг:     ${task.step}${task.totalSteps ? ` / ${task.totalSteps}` : ""}`,
    `│  Действие: ${task.expectedAction}`,
  ];

  if (task.description) lines.push(`│  Задача:  ${task.description}`);

  const allowed = FSM_TRANSITIONS[task.phase] ?? [];
  if (allowed.length) {
    const labels = allowed.map(p => PHASE_LABELS[p] ?? p).join(", ");
    lines.push(`│  Следующий возможный этап: ${labels}`);
  }

  if (task.plan.length) {
    lines.push("│  План:");
    task.plan.forEach((s, i) => {
      const marker = i < task.step ? "✓" : i === task.step ? "→" : " ";
      lines.push(`│    ${marker} ${i + 1}. ${s}`);
    });
  }

  if (task.phase === "validation" || task.validationResult) {
    const status = task.validationPassed ? "ПРОЙДЕНА ✓" : "Не пройдена";
    lines.push(`│  Валидация: ${status}`);
    if (task.validationResult) lines.push(`│  Результат: ${task.validationResult}`);
  }

  lines.push("└────────────────────────────────────────────────────");
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
    const raw = readJson(HISTORY_FILE);
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

// ─── Сборка контекста ─────────────────────────────────────────────────────────
function buildContext(history, task) {
  const phaseConfig  = PHASE_PROMPTS[task.phase] ?? PHASE_PROMPTS.planning;
  const stateText    = taskStateToText(task);
  const systemContent = phaseConfig.system + (stateText ? "\n\n" + stateText : "");
  const recent        = history.messages.slice(-10);
  return [{ role: "system", content: systemContent }, ...recent];
}

// ─── Основная функция: отправить сообщение ────────────────────────────────────
export async function ask(userQuery) {
  if (!userQuery?.trim()) throw new Error("Запрос не может быть пустым");

  const history = loadHistory();
  const task    = loadTask();

  if (!task.id) {
    throw new Error('Задача не создана. Используйте: node agent.js --task new "<описание>"');
  }

  history.turn = (history.turn ?? 0) + 1;
  const turn = history.turn;

  history.messages.push({
    role: "user", content: userQuery, turn, phase: task.phase, at: new Date().toISOString(),
  });

  // Автопереход approved → execution по явной команде пользователя
  if (task.phase === "approved") {
    if (/начин|выполн|старт|go|start/i.test(userQuery)) {
      transition(task, "execution", "Пользователь инициировал выполнение");
    }
  }

  const contextMessages = buildContext(history, task);
  const token  = await getAccessToken();
  const apiRes = await fetch(GIGACHAT_CHAT_URL, {
    method: "POST",
    headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: CONFIG.model, messages: contextMessages,
      temperature: CONFIG.temperature, max_tokens: CONFIG.max_tokens,
      top_p: CONFIG.top_p, stream: true,
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
      if (delta) { process.stdout.write(delta); result += delta; }
      if (chunk.usage) usage = chunk.usage;
    }
  }
  process.stdout.write("\n");

  history.messages.push({
    role: "assistant", content: result, turn, phase: task.phase, at: new Date().toISOString(),
  });

  autoAdvanceFSM(task, result);

  if (task.phase === "execution") {
    task.step = Math.min(task.step + 1, task.totalSteps ?? task.step + 1);
  }

  history.requests.push({
    timestamp: new Date().toISOString(), turn, phase: task.phase,
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

  // ── --task <sub>  Управление задачей и FSM ────────────────────────────────────
  if (args[0] === "--task") {
    const sub = args[1];

    // --task new <описание>
    if (sub === "new") {
      const description = args.slice(2).join(" ");
      if (!description) { console.error('Использование: --task new "<описание>"'); process.exit(1); }
      const task       = defaultTask();
      task.id          = randomUUID();
      task.description = description;
      task.createdAt   = new Date().toISOString();
      saveTask(task);
      saveHistory(defaultHistory());
      console.log(`✓ Задача создана: "${description}"`);
      console.log(`  Этап: ПЛАНИРОВАНИЕ`);
      console.log(`\nНачните диалог: node agent.js "составь план"`);
      process.exit(0);
    }

    // --task status
    if (sub === "status") {
      const task = loadTask();
      if (!task.id) { console.log('Задача не создана. Используйте: --task new "<описание>"'); process.exit(0); }
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

    // --task approve-plan  planning → approved
    if (sub === "approve-plan") {
      const task = loadTask();
      if (!task.id) { console.error("Задача не создана."); process.exit(1); }
      if (task.phase !== "planning") {
        console.error(`Утверждение плана доступно только на этапе ПЛАНИРОВАНИЕ. Текущий: ${task.phase}`);
        process.exit(1);
      }
      try {
        transition(task, "approved", "Пользователь утвердил план (CLI)");
        saveTask(task);
        console.log("✓ План утверждён. Переход: planning → approved");
        console.log("  Для запуска выполнения: node agent.js --task start");
        console.log('  Или напишите агенту: node agent.js "начинай"');
      } catch (err) {
        console.error(err.message);
        process.exit(1);
      }
      process.exit(0);
    }

    // --task start  approved → execution
    if (sub === "start") {
      const task = loadTask();
      if (!task.id) { console.error("Задача не создана."); process.exit(1); }
      try {
        transition(task, "execution", "Пользователь запустил выполнение (CLI)");
        saveTask(task);
        console.log("✓ Переход: approved → execution");
        console.log(`  Ожидаемое действие: ${task.expectedAction}`);
      } catch (err) {
        console.error(err.message);
        process.exit(1);
      }
      process.exit(0);
    }

    // --task mark-validated  (тест: вручную отметить валидацию пройденной)
    if (sub === "mark-validated") {
      const task = loadTask();
      if (!task.id) { console.error("Задача не создана."); process.exit(1); }
      if (task.phase !== "validation") {
        console.error(`mark-validated доступно только на этапе ВАЛИДАЦИЯ. Текущий: ${task.phase}`);
        process.exit(1);
      }
      task.validationPassed = true;
      task.validationResult = "Отмечено вручную (CLI)";
      saveTask(task);
      console.log("✓ Валидация отмечена пройденной.");
      console.log("  Теперь можно завершить: node agent.js --task advance done");
      process.exit(0);
    }

    // --task advance [<phase>] [--reason <текст>]  Переход с проверкой guard-условий
    if (sub === "advance") {
      const task = loadTask();
      if (!task.id) { console.error("Задача не создана."); process.exit(1); }

      const reasonIdx = args.indexOf("--reason");
      const reason    = reasonIdx !== -1 ? args.slice(reasonIdx + 1).join(" ") : "";
      let toPhase     = args[2];

      if (!toPhase || toPhase.startsWith("--")) {
        const allowed = FSM_TRANSITIONS[task.phase];
        if (!allowed?.length) { console.error(`Нет доступных переходов из "${task.phase}".`); process.exit(1); }
        toPhase = allowed[0];
      }

      try {
        const fromPhase = task.phase;
        transition(task, toPhase, reason || "Переход (CLI)");
        saveTask(task);
        console.log(`✓ Переход: ${fromPhase} → ${task.phase}`);
        console.log(`  Ожидаемое действие: ${task.expectedAction}`);
      } catch (err) {
        console.error(err.message);
        process.exit(1);
      }
      process.exit(0);
    }

    // --task force <phase> [--reason <текст>]  Принудительный переход (обходит guards, для тестов)
    if (sub === "force") {
      const task = loadTask();
      if (!task.id) { console.error("Задача не создана."); process.exit(1); }

      const reasonIdx = args.indexOf("--reason");
      const reason    = reasonIdx !== -1 ? args.slice(reasonIdx + 1).join(" ") : "";
      const toPhase   = args[2];

      if (!toPhase) { console.error("Использование: --task force <phase>"); process.exit(1); }

      const allowed = FSM_TRANSITIONS[task.phase];
      if (!allowed || !allowed.includes(toPhase)) {
        console.error(`Переход ${task.phase} → ${toPhase} недопустим. Доступные: ${allowed?.join(", ")}`);
        process.exit(1);
      }

      const fromPhase = task.phase;
      task.history.push({
        from: task.phase, to: toPhase,
        reason: reason || "Принудительный переход (force, CLI)",
        at: new Date().toISOString(),
      });
      task.phase          = toPhase;
      task.step           = 0;
      task.expectedAction = PHASE_PROMPTS[toPhase].expectedAction;

      saveTask(task);
      console.log(`⚡ Принудительный переход: ${fromPhase} → ${toPhase} (guards обойдены)`);
      console.log(`  Ожидаемое действие: ${task.expectedAction}`);
      process.exit(0);
    }

    // --task help
    console.log(`
Управление задачей:
  --task new "<описание>"   Создать новую задачу (сбрасывает историю)
  --task status             Показать текущее состояние FSM
  --task approve-plan       Утвердить план (planning → approved)
  --task start              Запустить выполнение (approved → execution)
  --task mark-validated     Отметить валидацию пройденной (для тестов)
  --task advance [<phase>]  Переход с проверкой guard-условий
  --task force <phase>      Принудительный переход (обходит guards, только для тестов)

Жизненный цикл:
  planning → approved → execution → validation → done → planning

Заблокированные переходы (guards):
  planning → execution       нельзя: нужно сначала approve-plan
  planning → approved        нельзя: если план пустой
  execution → done           нельзя: нужна валидация
  validation → done          нельзя: пока validationPassed=false
`);
    process.exit(0);
  }

  // ── Обычный диалог ────────────────────────────────────────────────────────────
  const query = args.join(" ");
  if (!query) {
    const task = loadTask();
    if (task.id) {
      console.log(taskStateToText(task));
    } else {
      console.log('Использование: node agent.js "<сообщение>"');
      console.log('Создать задачу: node agent.js --task new "<описание>"');
    }
    process.exit(0);
  }

  try {
    await ask(query);
    const task = loadTask();
    console.log(`\n${taskStateToText(task)}`);
  } catch (err) {
    console.error("Ошибка:", err.message);
    process.exit(1);
  }
}
