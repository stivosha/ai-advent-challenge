"""
RAG Pipeline  (GigaChat Embeddings + Chat)
==========================================
  question → retrieve chunks → augment prompt → LLM answer
  + side-by-side comparison: no-RAG vs RAG
  + 10-question evaluation benchmark

Usage:
  python rag.py                          # run full benchmark
  python rag.py --query "кто такой Хулио?"
  python rag.py --query "..." --strategy struct --top_k 5
  python rag.py --eval                   # only benchmark, no interactive query

Env:
  GIGACHAT_API_KEY=<your_key>   (or .env file)
"""

from __future__ import annotations

import os
import json
import sqlite3
import textwrap
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np
import faiss
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

INDEX_DIR  = Path(__file__).parent / "index"
EMBED_MODEL = "Embeddings"
CHAT_MODEL  = "GigaChat"          # or "GigaChat-Pro" / "GigaChat-Max"
TOP_K       = 5                   # chunks to retrieve

SYSTEM_PROMPT = (
    "Ты — литературный ассистент. Отвечай строго по тексту романа "
    "«Четыре всадника Апокалипсиса» Бласко Ибаньеса. "
    "Если ответа нет в предоставленных фрагментах — честно скажи об этом. "
    "Ссылайся на части и главы, если они указаны."
)

RAG_TEMPLATE = """\
Ниже приведены фрагменты романа, релевантные вопросу:

{context}

---
На основе этих фрагментов ответь на вопрос:
{question}"""

NO_RAG_TEMPLATE = "{question}"

# ---------------------------------------------------------------------------
# Eval benchmark  (10 questions)
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "Кто такой Мадариага и откуда он родом?",
        "expected": (
            "Мадариага — старый аргентинский скотовод-патриарх испанского происхождения, "
            "богатый владелец эстансии в пампасах, дед главных героев."
        ),
        "sources": ["Книга первая", "глава 1"],
    },
    {
        "id": 2,
        "question": "Кто такой Хулио Дезноайерс и чем он занимается в Париже?",
        "expected": (
            "Хулио — молодой аргентинец, внук Мадариаги, живёт в Париже, "
            "занимается живописью, ведёт праздный образ жизни, увлечён танго и любовными приключениями."
        ),
        "sources": ["Книга первая", "главы 2–4"],
    },
    {
        "id": 3,
        "question": "На ком женат Карл фон Харт и какова его связь с семьёй Дезноайерс?",
        "expected": (
            "Карл фон Харт женат на Елене, дочери Дезноайерса. "
            "Они — шурины: Хулио и Елена дети Марселино Дезноайерса, "
            "а их дядья — немецкой ветви наследников Мадариаги через его дочь Лунгиту."
        ),
        "sources": ["Книга первая", "глава 1–2"],
    },
    {
        "id": 4,
        "question": "Что символизируют четыре всадника Апокалипсиса в романе?",
        "expected": (
            "Четыре всадника — Чума, Война, Голод и Смерть — олицетворяют бедствия Первой мировой войны. "
            "Образ взят из Откровения Иоанна и проходит через весь роман как лейтмотив неизбежной катастрофы."
        ),
        "sources": ["Книга вторая", "апокалипсические главы"],
    },
    {
        "id": 5,
        "question": "Как изменился Хулио Дезноайерс с началом войны?",
        "expected": (
            "Под влиянием патриотического долга и любви к Маргарите Хулио вступает во французскую армию, "
            "меняет беззаботный образ жизни на воинский, проявляет смелость на фронте."
        ),
        "sources": ["Книга вторая–третья"],
    },
    {
        "id": 6,
        "question": "Что происходит с замком Вильблан во время немецкой оккупации?",
        "expected": (
            "Замок Вильблан — имение Дезноайерса — захватывают немецкие войска. "
            "Немецкие офицеры устраивают в нём штаб, реквизируют добро, "
            "а сам Марселино Дезноайерс оказывается фактически под домашним арестом."
        ),
        "sources": ["Книга вторая", "оккупация"],
    },
    {
        "id": 7,
        "question": "Кто такая Маргарита и каковы её отношения с Хулио?",
        "expected": (
            "Маргарита — замужняя женщина, в которую влюблён Хулио. "
            "Их роман начинается в Париже, но с началом войны Маргарита разрывает связь, "
            "посвящает себя уходу за ранеными; впоследствии её муж погибает на фронте."
        ),
        "sources": ["Книга первая–вторая"],
    },
    {
        "id": 8,
        "question": "Каков конфликт между французской и немецкой ветвями наследников Мадариаги?",
        "expected": (
            "Мадариага разделил состояние между двумя дочерьми: одна вышла за француза Дезноайерса, "
            "другая — за немца фон Харта. С началом войны семьи оказываются по разные стороны фронта, "
            "что символизирует трагедию разорванных войной человеческих связей."
        ),
        "sources": ["Книга первая", "глава 1"],
    },
    {
        "id": 9,
        "question": "Какова судьба Хулио Дезноайерса в конце романа?",
        "expected": (
            "Хулио погибает на фронте, отдав жизнь за Францию. "
            "Его гибель — искупление прежнего легкомыслия и воплощение темы жертвы, "
            "пронизывающей весь роман."
        ),
        "sources": ["Книга третья", "финал"],
    },
    {
        "id": 10,
        "question": "Какую роль играет Tchernoff (Чернов) в романе?",
        "expected": (
            "Чернов — русский эмигрант-интеллектуал, сосед Хулио в Париже. "
            "Он первым говорит о четырёх всадниках Апокалипсиса как метафоре надвигающейся войны, "
            "выступает голосом пророческого предупреждения и философского осмысления катастрофы."
        ),
        "sources": ["Книга первая–вторая", "апокалипсические сцены"],
    },
]

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    rank:    int
    dist:    float
    chunk_id: str
    title:   str
    section: str
    text:    str


def retrieve(query: str, strategy: str, top_k: int, giga: GigaChat) -> List[RetrievedChunk]:
    faiss_path = INDEX_DIR / f"{strategy}.faiss"
    db_path    = INDEX_DIR / f"{strategy}.db"

    resp  = giga.embeddings([query], model=EMBED_MODEL)
    q_vec = np.array([resp.data[0].embedding], dtype="float32")

    index = faiss.read_index(str(faiss_path))
    distances, ids = index.search(q_vec, top_k)

    con = sqlite3.connect(db_path)
    results: List[RetrievedChunk] = []
    for rank, (dist, row_id) in enumerate(zip(distances[0], ids[0]), 1):
        row = con.execute(
            "SELECT chunk_id, title, section, text FROM chunks WHERE rowid=?",
            (int(row_id),),
        ).fetchone()
        if row:
            cid, title, section, text = row
            results.append(RetrievedChunk(rank, float(dist), cid, title, section, text))
    con.close()
    return results


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        header = f"[{c.rank}] {c.title} / {c.section}  (id={c.chunk_id}, dist={c.dist:.4f})"
        parts.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def ask_llm(prompt: str, giga: GigaChat, system: str = SYSTEM_PROMPT) -> str:
    chat = Chat(
        model=CHAT_MODEL,
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=system),
            Messages(role=MessagesRole.USER,   content=prompt),
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    response = giga.chat(chat)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Single RAG query (interactive)
# ---------------------------------------------------------------------------
def run_query(question: str, strategy: str, top_k: int) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set")

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        print(f"\n{'='*70}")
        print(f"ВОПРОС: {question}")
        print(f"{'='*70}")

        # --- No RAG ---
        print("\n[БЕЗ RAG] Запрос к модели...")
        answer_no_rag = ask_llm(NO_RAG_TEMPLATE.format(question=question), giga)
        print(f"\n  ОТВЕТ (без RAG):\n{textwrap.indent(answer_no_rag, '  ')}")

        # --- Retrieve ---
        print(f"\n[RAG] Поиск в индексе '{strategy}' (top_k={top_k})...")
        chunks = retrieve(question, strategy, top_k, giga)
        for c in chunks:
            print(f"  [{c.rank}] dist={c.dist:.4f}  {c.title[:40]} / {c.section[:40]}")

        context = build_context(chunks)
        rag_prompt = RAG_TEMPLATE.format(context=context, question=question)

        print("\n[RAG] Запрос к модели с контекстом...")
        answer_rag = ask_llm(rag_prompt, giga)
        print(f"\n  ОТВЕТ (с RAG):\n{textwrap.indent(answer_rag, '  ')}")

        print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Benchmark: 10 eval questions
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    id:       int
    question: str
    expected: str
    sources:  List[str]
    retrieved_sources: List[str] = field(default_factory=list)
    answer_no_rag: str = ""
    answer_rag:    str = ""


def run_benchmark(strategy: str, top_k: int) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set")

    results: List[EvalResult] = []

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        for q in EVAL_QUESTIONS:
            print(f"\n{'='*70}")
            print(f"[Q{q['id']:02d}] {q['question']}")
            print(f"{'='*70}")

            er = EvalResult(
                id=q["id"],
                question=q["question"],
                expected=q["expected"],
                sources=q["sources"],
            )

            # No RAG
            print("  → Без RAG...")
            er.answer_no_rag = ask_llm(
                NO_RAG_TEMPLATE.format(question=q["question"]), giga
            )

            # RAG
            print(f"  → Поиск в индексе '{strategy}'...")
            chunks = retrieve(q["question"], strategy, top_k, giga)
            er.retrieved_sources = [
                f"{c.title} / {c.section}" for c in chunks
            ]

            context = build_context(chunks)
            rag_prompt = RAG_TEMPLATE.format(context=context, question=q["question"])
            print("  → С RAG...")
            er.answer_rag = ask_llm(rag_prompt, giga)

            results.append(er)

            # Print comparison
            print(f"\n  ОЖИДАНИЕ:\n    {er.expected}")
            print(f"\n  ИСТОЧНИКИ (ожидаемые): {', '.join(er.sources)}")
            print(f"  ИСТОЧНИКИ (retrieved):  {', '.join(er.retrieved_sources[:3])}")
            print(f"\n  БЕЗ RAG:\n{textwrap.indent(er.answer_no_rag, '    ')}")
            print(f"\n  С RAG:\n{textwrap.indent(er.answer_rag, '    ')}")

    # Save report
    report_path = INDEX_DIR / "rag_benchmark.json"
    report = []
    for r in results:
        report.append({
            "id": r.id,
            "question": r.question,
            "expected": r.expected,
            "expected_sources": r.sources,
            "retrieved_sources": r.retrieved_sources,
            "answer_no_rag": r.answer_no_rag,
            "answer_rag": r.answer_rag,
        })
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n\nОтчёт сохранён → {report_path}")

    # Summary table
    print("\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("="*70)
    print(f"{'#':>3} | {'Вопрос (кратко)':<45} | {'Источники совпали?'}")
    print("-"*70)
    for r in results:
        retrieved_flat = " ".join(r.retrieved_sources).lower()
        match = any(s.lower()[:10] in retrieved_flat for s in r.sources)
        mark = "✓" if match else "·"
        short_q = r.question[:44]
        print(f"{r.id:>3} | {short_q:<45} | {mark}")
    print("="*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline (GigaChat)")
    parser.add_argument("--query",    default=None, help="Одиночный вопрос")
    parser.add_argument("--strategy", default="struct",
                        choices=["fixed", "struct"], help="Индекс (default: struct)")
    parser.add_argument("--top_k",   default=TOP_K, type=int, help="Число чанков")
    parser.add_argument("--eval",    action="store_true",
                        help="Только бенчмарк (10 вопросов)")
    args = parser.parse_args()

    if args.query:
        run_query(args.query, args.strategy, args.top_k)
    else:
        run_benchmark(args.strategy, args.top_k)


if __name__ == "__main__":
    main()
