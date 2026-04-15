"""
RAG Pipeline  (GigaChat Embeddings + Chat)
==========================================
  question → retrieve(top_k=TOP_K_RETRIEVE) → rerank/filter → augment prompt → LLM answer
  + side-by-side comparison: no-RAG vs RAG vs RAG+filter
  + 10-question evaluation benchmark

Usage:
  python rag.py                          # full benchmark (both modes)
  python rag.py --query "кто такой Хулио?"
  python rag.py --query "..." --strategy struct --top_k 10 --top_k_final 3
  python rag.py --eval                   # only benchmark, no interactive query
  python rag.py --eval --no_rerank       # benchmark without reranker stage

Env:
  GIGACHAT_API_KEY=<your_key>   (or .env file)
"""

from __future__ import annotations

import os
import re
import json
import sqlite3
import textwrap
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

INDEX_DIR   = Path(__file__).parent / "index"
EMBED_MODEL = "Embeddings"
CHAT_MODEL  = "GigaChat"          # or "GigaChat-Pro" / "GigaChat-Max"

# Stage 1 — initial retrieval (cast a wide net)
TOP_K_RETRIEVE = 10

# Stage 2 — reranker / filter
DIST_THRESHOLD = 0.80   # L2 distance; chunks with dist > threshold are dropped
TOP_K_FINAL    = 5      # max chunks passed to LLM after filtering

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
    rank:     int
    dist:     float
    chunk_id: str
    title:    str
    section:  str
    text:     str


def retrieve(query: str, strategy: str, fetch_k: int, giga: GigaChat) -> List[RetrievedChunk]:
    """Stage 1: embed query → FAISS search → return fetch_k raw candidates."""
    faiss_path = INDEX_DIR / f"{strategy}.faiss"
    db_path    = INDEX_DIR / f"{strategy}.db"

    resp  = giga.embeddings([query], model=EMBED_MODEL)
    q_vec = np.array([resp.data[0].embedding], dtype="float32")

    index = faiss.read_index(str(faiss_path))
    distances, ids = index.search(q_vec, fetch_k)

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


# ---------------------------------------------------------------------------
# Stage 2 — Reranker / Relevance Filter
# ---------------------------------------------------------------------------

def _keyword_overlap(query: str, text: str) -> float:
    """Heuristic: fraction of query words (len>=4) found in chunk text."""
    words = re.findall(r'\b\w{4,}\b', query.lower())
    if not words:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for w in words if w in text_lower)
    return hits / len(words)


def rerank_filter(
    query: str,
    chunks: List[RetrievedChunk],
    threshold: float,
    top_k_final: int,
) -> Tuple[List[RetrievedChunk], List[RetrievedChunk]]:
    """
    Stage 2: filter by distance threshold + heuristic keyword boost.

    Returns:
        (filtered_chunks, rejected_chunks)
        filtered_chunks — kept, re-ranked, limited to top_k_final
        rejected_chunks — dropped by threshold
    """
    kept:     List[Tuple[float, RetrievedChunk]] = []
    rejected: List[RetrievedChunk] = []

    for c in chunks:
        if c.dist > threshold:
            rejected.append(c)
            continue

        # Combined score: lower dist is better; keyword overlap boosts rank.
        # Normalise dist to [0,1] relative to threshold, then subtract keyword bonus.
        norm_dist    = c.dist / threshold          # 0..1 (lower = better)
        kw_bonus     = _keyword_overlap(query, c.text) * 0.15   # up to -0.15
        final_score  = norm_dist - kw_bonus
        kept.append((final_score, c))

    # Sort by combined score ascending (best first)
    kept.sort(key=lambda x: x[0])

    # Re-assign ranks
    filtered = []
    for new_rank, (score, c) in enumerate(kept[:top_k_final], 1):
        filtered.append(RetrievedChunk(new_rank, c.dist, c.chunk_id, c.title, c.section, c.text))

    return filtered, rejected


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
def run_query(
    question:    str,
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   float,
    use_rerank:  bool,
) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set")

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        print(f"\n{'='*70}")
        print(f"ВОПРОС: {question}")
        print(f"{'='*70}")
        print(f"Параметры: strategy={strategy}, fetch_k={fetch_k}, "
              f"top_k_final={top_k_final}, threshold={threshold}, rerank={use_rerank}")

        # --- No RAG ---
        print("\n[БЕЗ RAG] Запрос к модели...")
        answer_no_rag = ask_llm(NO_RAG_TEMPLATE.format(question=question), giga)
        print(f"\n  ОТВЕТ (без RAG):\n{textwrap.indent(answer_no_rag, '  ')}")

        # --- Stage 1: Retrieve ---
        print(f"\n[STAGE 1] Поиск в индексе '{strategy}' (fetch_k={fetch_k})...")
        chunks_raw = retrieve(question, strategy, fetch_k, giga)
        print(f"  Получено {len(chunks_raw)} чанков:")
        for c in chunks_raw:
            flag = "✓" if c.dist <= threshold else "✗"
            print(f"  {flag} [{c.rank}] dist={c.dist:.4f}  {c.title[:35]} / {c.section[:30]}")

        # --- Stage 2: Rerank/filter ---
        if use_rerank:
            chunks_final, rejected = rerank_filter(question, chunks_raw, threshold, top_k_final)
            print(f"\n[STAGE 2] После фильтра (threshold={threshold}): "
                  f"осталось {len(chunks_final)}, отброшено {len(rejected)}")
            if rejected:
                print("  Отброшены:")
                for c in rejected:
                    print(f"    dist={c.dist:.4f}  {c.title[:35]} / {c.section[:30]}")
        else:
            chunks_final = chunks_raw[:top_k_final]
            print(f"\n[STAGE 2] Фильтр отключён — берём первые {len(chunks_final)} чанков.")

        # --- RAG without filter (using raw top_k_final) ---
        chunks_no_filter = chunks_raw[:top_k_final]
        context_no_filter = build_context(chunks_no_filter)
        rag_prompt_no_filter = RAG_TEMPLATE.format(context=context_no_filter, question=question)

        print(f"\n[RAG БЕЗ ФИЛЬТРА] top_k={len(chunks_no_filter)} — запрос к модели...")
        answer_rag_raw = ask_llm(rag_prompt_no_filter, giga)
        print(f"\n  ОТВЕТ (RAG без фильтра):\n{textwrap.indent(answer_rag_raw, '  ')}")

        if use_rerank and chunks_final:
            context_filtered = build_context(chunks_final)
            rag_prompt_filtered = RAG_TEMPLATE.format(context=context_filtered, question=question)
            print(f"\n[RAG С ФИЛЬТРОМ] top_k={len(chunks_final)} — запрос к модели...")
            answer_rag_filtered = ask_llm(rag_prompt_filtered, giga)
            print(f"\n  ОТВЕТ (RAG с фильтром):\n{textwrap.indent(answer_rag_filtered, '  ')}")
        elif use_rerank and not chunks_final:
            print("\n[RAG С ФИЛЬТРОМ] Все чанки отброшены — LLM не вызывается.")

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
    # retrieval stats
    chunks_before_filter: int = 0
    chunks_after_filter:  int = 0
    dists_before: List[float] = field(default_factory=list)
    dists_after:  List[float] = field(default_factory=list)
    retrieved_sources_raw:      List[str] = field(default_factory=list)
    retrieved_sources_filtered: List[str] = field(default_factory=list)
    # answers
    answer_no_rag:      str = ""
    answer_rag_raw:     str = ""   # no filter
    answer_rag_filtered: str = ""  # with filter


def _source_match(expected_sources: List[str], retrieved: List[str]) -> bool:
    flat = " ".join(retrieved).lower()
    return any(s.lower()[:10] in flat for s in expected_sources)


def run_benchmark(
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   float,
    use_rerank:  bool,
) -> None:
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

            # Stage 1: retrieve
            print(f"  → Stage 1: поиск в '{strategy}' (fetch_k={fetch_k})...")
            chunks_raw = retrieve(q["question"], strategy, fetch_k, giga)
            er.chunks_before_filter = len(chunks_raw)
            er.dists_before = [c.dist for c in chunks_raw]
            er.retrieved_sources_raw = [f"{c.title} / {c.section}" for c in chunks_raw]

            # RAG without filter
            chunks_no_filter = chunks_raw[:top_k_final]
            context_raw = build_context(chunks_no_filter)
            rag_prompt_raw = RAG_TEMPLATE.format(context=context_raw, question=q["question"])
            print("  → RAG без фильтра...")
            er.answer_rag_raw = ask_llm(rag_prompt_raw, giga)

            # Stage 2: rerank/filter
            if use_rerank:
                chunks_filtered, _ = rerank_filter(q["question"], chunks_raw, threshold, top_k_final)
            else:
                chunks_filtered = chunks_no_filter

            er.chunks_after_filter = len(chunks_filtered)
            er.dists_after = [c.dist for c in chunks_filtered]
            er.retrieved_sources_filtered = [
                f"{c.title} / {c.section}" for c in chunks_filtered
            ]

            if chunks_filtered:
                context_filtered = build_context(chunks_filtered)
                rag_prompt_filtered = RAG_TEMPLATE.format(
                    context=context_filtered, question=q["question"]
                )
                print("  → RAG с фильтром...")
                er.answer_rag_filtered = ask_llm(rag_prompt_filtered, giga)
            else:
                er.answer_rag_filtered = "[все чанки отброшены фильтром]"

            results.append(er)

            # Print inline comparison
            print(f"\n  ОЖИДАНИЕ: {er.expected[:80]}...")
            print(f"  Источники ожидаемые : {', '.join(er.sources)}")
            print(f"  Чанков до фильтра   : {er.chunks_before_filter}  "
                  f"dist=[{min(er.dists_before):.3f}..{max(er.dists_before):.3f}]")
            print(f"  Чанков после фильтра: {er.chunks_after_filter}  "
                  f"(threshold={threshold})")
            print(f"  Источники retrieved : {', '.join(er.retrieved_sources_filtered[:3])}")

    # ---- Save report ----
    report_path = INDEX_DIR / "rag_benchmark.json"
    report = []
    for r in results:
        report.append({
            "id": r.id,
            "question": r.question,
            "expected": r.expected,
            "expected_sources": r.sources,
            "retrieval": {
                "chunks_before_filter": r.chunks_before_filter,
                "chunks_after_filter":  r.chunks_after_filter,
                "threshold": threshold,
                "dist_min_before": round(min(r.dists_before, default=0), 4),
                "dist_max_before": round(max(r.dists_before, default=0), 4),
                "dist_mean_after": round(
                    sum(r.dists_after) / len(r.dists_after), 4
                ) if r.dists_after else None,
            },
            "retrieved_sources_raw":      r.retrieved_sources_raw,
            "retrieved_sources_filtered": r.retrieved_sources_filtered,
            "source_match_raw":      _source_match(r.sources, r.retrieved_sources_raw),
            "source_match_filtered": _source_match(r.sources, r.retrieved_sources_filtered),
            "answer_no_rag":       r.answer_no_rag,
            "answer_rag_raw":      r.answer_rag_raw,
            "answer_rag_filtered": r.answer_rag_filtered,
        })
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n\nОтчёт сохранён → {report_path}")

    # ---- Summary table ----
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print(f"{'#':>3} | {'Вопрос (кратко)':<40} | "
          f"{'БЕЗ RAG':^7} | {'RAG':^7} | {'RAG+filter':^10} | "
          f"{'до':^4} | {'после':^5}")
    print("-"*80)
    for r in results:
        match_raw      = "✓" if _source_match(r.sources, r.retrieved_sources_raw) else "·"
        match_filtered = "✓" if _source_match(r.sources, r.retrieved_sources_filtered) else "·"
        short_q = r.question[:39]
        print(
            f"{r.id:>3} | {short_q:<40} | "
            f"{'—':^7} | {match_raw:^7} | {match_filtered:^10} | "
            f"{r.chunks_before_filter:^4} | {r.chunks_after_filter:^5}"
        )
    print("="*80)

    total = len(results)
    hit_raw      = sum(1 for r in results if _source_match(r.sources, r.retrieved_sources_raw))
    hit_filtered = sum(1 for r in results if _source_match(r.sources, r.retrieved_sources_filtered))
    avg_before = sum(r.chunks_before_filter for r in results) / total
    avg_after  = sum(r.chunks_after_filter  for r in results) / total
    print(f"\nSource recall  RAG raw: {hit_raw}/{total}  |  RAG+filter: {hit_filtered}/{total}")
    print(f"Avg chunks     before : {avg_before:.1f}  |  after: {avg_after:.1f}  "
          f"(reduction {100*(1-avg_after/avg_before):.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline (GigaChat) with reranker")
    parser.add_argument("--query",       default=None,  help="Одиночный вопрос")
    parser.add_argument("--strategy",    default="struct",
                        choices=["fixed", "struct"], help="Индекс (default: struct)")
    parser.add_argument("--top_k",       default=TOP_K_RETRIEVE, type=int,
                        help=f"Чанков на этапе поиска (default: {TOP_K_RETRIEVE})")
    parser.add_argument("--top_k_final", default=TOP_K_FINAL, type=int,
                        help=f"Чанков после фильтра (default: {TOP_K_FINAL})")
    parser.add_argument("--threshold",   default=DIST_THRESHOLD, type=float,
                        help=f"L2-порог отсечения (default: {DIST_THRESHOLD})")
    parser.add_argument("--no_rerank",   action="store_true",
                        help="Отключить этап rerank/filter")
    parser.add_argument("--eval",        action="store_true",
                        help="Только бенчмарк (10 вопросов)")
    args = parser.parse_args()

    use_rerank = not args.no_rerank

    if args.query:
        run_query(
            question=args.query,
            strategy=args.strategy,
            fetch_k=args.top_k,
            top_k_final=args.top_k_final,
            threshold=args.threshold,
            use_rerank=use_rerank,
        )
    else:
        run_benchmark(
            strategy=args.strategy,
            fetch_k=args.top_k,
            top_k_final=args.top_k_final,
            threshold=args.threshold,
            use_rerank=use_rerank,
        )


if __name__ == "__main__":
    main()
