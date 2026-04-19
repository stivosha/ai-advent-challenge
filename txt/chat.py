"""
RAG Chat CLI
============
Interactive chat over «Four Horsemen of the Apocalypse» corpus.

Features:
  • Dialogue history   — last N turns included in every prompt
  • RAG per question   — FAISS retrieve → rerank → LLM
  • Sources always     — shown after each answer
  • Task state memory  — goal, clarified facts, fixed terms (updated by LLM)

Usage:
  python chat.py
  python chat.py --strategy struct --top_k 10 --top_k_final 4
  python chat.py --no_rerank

In-chat commands:
  /state    show task state
  /history  show dialogue history
  /clear    reset history + state
  /help     command list
  /quit     exit
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from gigachat import GigaChat

from rag import (
    retrieve, rerank_filter, build_context,
    parse_structured_answer, make_unknown_answer, _topic_coverage,
    TOP_K_RETRIEVE, TOP_K_FINAL,
    ask_llm,
)

load_dotenv(Path(__file__).parent / ".env")

TOPIC_COVER_MIN      = 0.10
MAX_HISTORY_CONTEXT  = 5     # turns included in each prompt


# ──────────────────────────────────────────────────────────────────────────────
# Task State
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TaskState:
    goal:        str       = ""
    clarified:   List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.goal and not self.clarified and not self.constraints

    def as_context(self) -> str:
        if self.is_empty():
            return ""
        parts = []
        if self.goal:
            parts.append(f"Цель диалога: {self.goal}")
        if self.clarified:
            parts.append("Уже уточнено: " + "; ".join(self.clarified))
        if self.constraints:
            parts.append("Зафиксированные термины/ограничения: " + "; ".join(self.constraints))
        return "\n".join(parts)

    def render(self) -> str:
        lines = []
        if self.goal:
            lines.append(f"  Цель      : {self.goal}")
        if self.clarified:
            lines.append("  Уточнено  :")
            for item in self.clarified:
                lines.append(f"    • {item}")
        if self.constraints:
            lines.append("  Термины   :")
            for item in self.constraints:
                lines.append(f"    • {item}")
        return "\n".join(lines) if lines else "  (пусто)"


# ──────────────────────────────────────────────────────────────────────────────
# Dialogue History
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Turn:
    question: str
    answer:   str
    sources:  List[str]

    def brief(self) -> str:
        short = self.answer[:200].replace("\n", " ")
        src   = ", ".join(self.sources[:2]) if self.sources else "—"
        return f"В: {self.question}\nО: {short}…\n(источники: {src})"


class DialogHistory:
    def __init__(self) -> None:
        self.turns: List[Turn] = []

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def recent(self, n: int) -> List[Turn]:
        return self.turns[-n:]

    def as_context(self, n: int) -> str:
        if not self.turns:
            return ""
        parts = ["--- Предыдущие вопросы и ответы ---"]
        for t in self.recent(n):
            parts.append(t.brief())
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.turns.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "Ты — литературный ассистент по роману «Четыре всадника Апокалипсиса» "
    "Бласко Ибаньеса. "
    "Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов текста и истории диалога. "
    "ЗАПРЕЩЕНО использовать собственные знания вне фрагментов. "
    "Если вопрос не относится к роману или ответа нет во фрагментах — "
    "раздел ## ОТВЕТ должен начинаться словами 'Не знаю'. "
    "ОБЯЗАТЕЛЬНО используй три раздела: ## ОТВЕТ, ## ИСТОЧНИКИ, ## ЦИТАТЫ."
)

_CHAT_TEMPLATE = """\
{history_block}
{task_block}
Фрагменты романа, релевантные текущему вопросу:

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Текущий вопрос: {question}

Ответь СТРОГО в следующем формате:

## ОТВЕТ
<Развёрнутый ответ, опираясь на фрагменты и историю диалога>

## ИСТОЧНИКИ
<Каждый использованный фрагмент — отдельной строкой: - [chunk_id] Название / Раздел>

## ЦИТАТЫ
<2–4 дословные цитаты из фрагментов. Каждая начинается с ">">
"""

_STATE_PROMPT = """\
Ты — аналитик диалогов. Проанализируй последний обмен и верни JSON.

Текущее состояние:
{current_state}

Вопрос пользователя: {question}
Ответ ассистента: {answer}

Верни JSON (без markdown-обёртки):
{{
  "goal": "цель диалога одним предложением (пустая строка если неясно)",
  "new_clarified": ["новые уточнённые факты из этого обмена"],
  "new_constraints": ["новые термины/имена/ограничения, зафиксированные пользователем"]
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# State updater
# ──────────────────────────────────────────────────────────────────────────────
def _update_state(state: TaskState, question: str, answer: str, giga: GigaChat) -> None:
    prompt = _STATE_PROMPT.format(
        current_state=state.as_context() or "(пусто)",
        question=question,
        answer=answer[:600],
    )
    try:
        raw = ask_llm(prompt, giga, system="Ты — аналитик диалогов. Отвечай только JSON.")
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        data = json.loads(raw)
    except Exception:
        return

    if data.get("goal") and not state.goal:
        state.goal = data["goal"]
    for item in data.get("new_clarified", []):
        if item and item not in state.clarified:
            state.clarified.append(item)
    for item in data.get("new_constraints", []):
        if item and item not in state.constraints:
            state.constraints.append(item)


# ──────────────────────────────────────────────────────────────────────────────
# Single turn
# ──────────────────────────────────────────────────────────────────────────────
def chat_turn(
    question:    str,
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
    history:     DialogHistory,
    state:       TaskState,
    giga:        GigaChat,
) -> Turn:
    chunks_raw = retrieve(question, strategy, fetch_k, giga)

    if use_rerank:
        chunks_final, _, _ = rerank_filter(question, chunks_raw, threshold, top_k_final)
    else:
        chunks_final = chunks_raw[:top_k_final]

    if not chunks_final or _topic_coverage(question, chunks_final) < TOPIC_COVER_MIN:
        sa = make_unknown_answer()
        turn = Turn(question=question, answer=sa.answer, sources=sa.sources)
        history.add(turn)
        _update_state(state, question, sa.answer, giga)
        return turn

    context       = build_context(chunks_final)
    history_block = history.as_context(MAX_HISTORY_CONTEXT)
    task_text     = state.as_context()
    task_block    = f"--- Состояние задачи ---\n{task_text}\n" if task_text else ""

    prompt = _CHAT_TEMPLATE.format(
        history_block=history_block + "\n" if history_block else "",
        task_block=task_block,
        context=context,
        question=question,
    )
    raw = ask_llm(prompt, giga, system=_SYSTEM)
    sa  = parse_structured_answer(raw)

    turn = Turn(question=question, answer=sa.answer, sources=sa.sources)
    history.add(turn)
    _update_state(state, question, sa.answer, giga)
    return turn


# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────
_LINE = "─" * 68

def _print_turn(turn: Turn, n: int) -> None:
    print(f"\n{_LINE}")
    print(f"[{n}] ОТВЕТ:")
    print(turn.answer)
    print(f"\nИСТОЧНИКИ:")
    if turn.sources and turn.sources[0] != "(нет релевантных источников)":
        for src in turn.sources:
            print(f"  • {src}")
    else:
        print("  (нет релевантных источников)")
    print(_LINE)


def _print_state(state: TaskState) -> None:
    print(f"\n{'='*68}")
    print("СОСТОЯНИЕ ЗАДАЧИ:")
    print(state.render())
    print(f"{'='*68}")


def _print_history(history: DialogHistory) -> None:
    print(f"\n{'='*68}")
    if not history.turns:
        print("История пуста.")
    else:
        print(f"ИСТОРИЯ ДИАЛОГА — {len(history.turns)} вопрос(ов):")
        for i, t in enumerate(history.turns, 1):
            print(f"\n  [{i}] В: {t.question}")
            short = t.answer[:100].replace("\n", " ")
            print(f"       О: {short}…")
    print(f"{'='*68}")


_HELP = """\
Команды:
  /state    — состояние задачи (цель, уточнения, термины)
  /history  — история диалога
  /clear    — очистить историю и состояние
  /help     — эта справка
  /quit     — выход
"""


# ──────────────────────────────────────────────────────────────────────────────
# Chat loop
# ──────────────────────────────────────────────────────────────────────────────
def run_chat(
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set")

    history = DialogHistory()
    state   = TaskState()
    turn_n  = 0

    print("=" * 68)
    print("RAG-ЧАТ  «Четыре всадника Апокалипсиса»  (GigaChat + FAISS)")
    print(f"strategy={strategy}  fetch_k={fetch_k}  top_k={top_k_final}  rerank={use_rerank}")
    print(_HELP)

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        while True:
            try:
                user_input = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd in ("/quit", "/exit", "/q"):
                    print("Выход.")
                    break
                elif cmd == "/state":
                    _print_state(state)
                elif cmd == "/history":
                    _print_history(history)
                elif cmd == "/clear":
                    history.clear()
                    state = TaskState()
                    turn_n = 0
                    print("  Очищено.")
                else:
                    print(_HELP)
                continue

            turn_n += 1
            print("  …поиск", end="\r", flush=True)

            try:
                turn = chat_turn(
                    question=user_input,
                    strategy=strategy,
                    fetch_k=fetch_k,
                    top_k_final=top_k_final,
                    threshold=threshold,
                    use_rerank=use_rerank,
                    history=history,
                    state=state,
                    giga=giga,
                )
            except Exception as exc:
                turn_n -= 1
                print(f"\n[ОШИБКА] {exc}")
                print("  Вопрос пропущен. Повторите или проверьте соединение с API.")
                continue

            _print_turn(turn, turn_n)

            # Compact state hint after each answer
            if not state.is_empty():
                hint_parts = []
                if state.goal:
                    hint_parts.append(f"цель: {state.goal}")
                if state.clarified:
                    hint_parts.append(f"уточнено: {len(state.clarified)}")
                if state.constraints:
                    hint_parts.append(f"терминов: {len(state.constraints)}")
                print(f"[задача] {' | '.join(hint_parts)}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Chat CLI")
    parser.add_argument("--strategy",    default="struct", choices=["fixed", "struct"])
    parser.add_argument("--top_k",       default=TOP_K_RETRIEVE, type=int)
    parser.add_argument("--top_k_final", default=TOP_K_FINAL,    type=int)
    parser.add_argument("--threshold",   default=None,           type=float)
    parser.add_argument("--no_rerank",   action="store_true")
    args = parser.parse_args()

    run_chat(
        strategy=args.strategy,
        fetch_k=args.top_k,
        top_k_final=args.top_k_final,
        threshold=args.threshold,
        use_rerank=not args.no_rerank,
    )


if __name__ == "__main__":
    main()
