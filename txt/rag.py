"""
RAG Pipeline  (GigaChat Embeddings + Chat)
==========================================
  question → retrieve(fetch_k) → rerank/filter → structured LLM answer
  Structured answer always contains:
    ## ОТВЕТ      — actual answer
    ## ИСТОЧНИКИ  — list of chunk_id / section used
    ## ЦИТАТЫ     — verbatim quotes from chunks
  If no chunk passes the relevance threshold → "не знаю" without LLM call.

Usage:
  python rag.py                             # full benchmark
  python rag.py --query "кто такой Хулио?"
  python rag.py --query "..." --strategy struct --top_k 10 --top_k_final 3
  python rag.py --eval                      # only benchmark
  python rag.py --eval --no_rerank          # skip reranker (baseline)

Env:
  GIGACHAT_API_KEY=<your_key>   (or .env file)
"""

from __future__ import annotations

import sys
import io
# Force UTF-8 output on Windows (avoids CP1251 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import json
import sqlite3
import textwrap
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

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
CHAT_MODEL  = "GigaChat"

TOP_K_RETRIEVE = 10    # candidates from FAISS (wide net)
TOP_K_FINAL    = 5     # max chunks sent to LLM after filter

# Adaptive threshold: keep chunks with dist <= best_dist * AUTO_FACTOR.
# GigaChat embeddings are NOT unit-normalized (norm ≈ 28),
# so L2 distances are in the hundreds — never use a fixed small threshold.
# AUTO_FACTOR=1.05 means "within 5% of the best match distance".
AUTO_FACTOR      = 1.05  # keep chunks within 5% of best match
DIST_THRESHOLD   = None  # None → adaptive; float → absolute cutoff
TOPIC_COVER_MIN  = 0.10  # min fraction of query words (>=4 chars) that must
                          # appear in at least one chunk; below → "не знаю"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Ты — литературный ассистент по роману «Четыре всадника Апокалипсиса» "
    "Бласко Ибаньеса. "
    "КРИТИЧЕСКИ ВАЖНО: отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов текста. "
    "ЗАПРЕЩЕНО использовать собственные знания, энциклопедические факты "
    "или информацию вне фрагментов. "
    "Если вопрос не относится к роману или ответа нет во фрагментах — "
    "раздел ## ОТВЕТ должен начинаться словами 'Не знаю' и объяснить почему. "
    "ОБЯЗАТЕЛЬНО используй ровно три раздела: ## ОТВЕТ, ## ИСТОЧНИКИ, ## ЦИТАТЫ."
)

RAG_TEMPLATE = """\
Фрагменты романа, релевантные вопросу:

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Вопрос: {question}

Ответь СТРОГО в следующем формате (три раздела, заголовки точно как ниже):

## ОТВЕТ
<Развёрнутый ответ на вопрос, опираясь только на фрагменты выше>

## ИСТОЧНИКИ
<Каждый использованный фрагмент — отдельной строкой: - [chunk_id] Название / Раздел>

## ЦИТАТЫ
<2–4 дословные цитаты из фрагментов, подтверждающие ответ. Каждая цитата начинается с символа ">">
"""

NO_CONTEXT_ANSWER = """\
## ОТВЕТ
Не знаю. По данному вопросу в тексте романа не найдено релевантных фрагментов (все кандидаты ниже порога релевантности). Уточните, пожалуйста, вопрос или перефразируйте его.

## ИСТОЧНИКИ
(нет релевантных источников)

## ЦИТАТЫ
(нет цитат — фрагменты не прошли фильтр релевантности)
"""

NO_RAG_TEMPLATE = "{question}"

# ---------------------------------------------------------------------------
# Eval questions
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "Кто такой Юрий Анохин и какие у него профессии в экспедиции?",
        "expected": (
            "Юрий Анохин — рассказчик и главный герой. "
            "В экспедиции совмещает обязанности кинооператора, киномеханика и радиста."
        ),
        "sources": ["ЧАСТЬ ПЕРВАЯ", "1. КАТАСТРОФА"],
    },
    {
        "id": 2,
        "question": "Что такое розовые облака и где они впервые появились?",
        "expected": (
            "Розовые облака — загадочные густо-розовые объекты, похожие на дирижабли или линзы, "
            "летящие значительно ниже кучевых облаков. "
            "Впервые их наблюдали американские зимовщики с базы Мак-Мёрдо над островом Росса в Антарктиде."
        ),
        "sources": ["ЧАСТЬ ПЕРВАЯ", "3. РОЗОВЫЕ \"ОБЛАКА\""],
    },
    {
        "id": 3,
        "question": "Кто такой Зернов и какова его роль в экспедиции?",
        "expected": (
            "Борис Аркадьевич Зернов — начальник антарктической экспедиции, гляциолог, "
            "способный заменить геофизика и сейсмолога. "
            "Требователен и непреклонен, ведёт научное расследование феномена розовых облаков."
        ),
        "sources": ["ЧАСТЬ ПЕРВАЯ", "1. КАТАСТРОФА"],
    },
    {
        "id": 4,
        "question": "Что произошло с членами экспедиции после контакта с розовыми облаками?",
        "expected": (
            "После контакта с облаками у каждого члена экспедиции появился двойник — "
            "точная копия человека, способная говорить и имитировать поведение оригинала. "
            "Двойники оказались порождением инопланетного разума."
        ),
        "sources": ["ЧАСТЬ ПЕРВАЯ", "2. ДВОЙНИКИ"],
    },
    {
        "id": 5,
        "question": "Кто такой Дьячук и чем он занимается в экспедиции?",
        "expected": (
            "Анатолий (Толька) Дьячук — метеоролог, фельдшер и кок экспедиции. "
            "Скептик и насмешник, любит шутить, но добросовестно выполняет свои обязанности."
        ),
        "sources": ["ЧАСТЬ ПЕРВАЯ", "1. КАТАСТРОФА"],
    },
    {
        "id": 6,
        "question": "Чем занимался Вано Чохели в экспедиции?",
        "expected": (
            "Вано Чохели — водитель и механик гигантского снегохода «Харьковчанка». "
            "Умел починить всё — от лопнувшей гусеницы до перегоревшей электроплитки."
        ),
        "sources": ["ЧАСТЬ ПЕРВАЯ", "1. КАТАСТРОФА"],
    },
    {
        "id": 7,
        "question": "Что произошло на отчётном совещании по итогам экспедиции?",
        "expected": (
            "Зернов признал, что никаких научных материалов, кроме личных впечатлений "
            "и фильма Анохина, у экспедиции нет. "
            "Астрономические наблюдения не дали оснований для определённых выводов. "
            "Факт появления ледяных скоплений в атмосфере был зафиксирован несколькими обсерваториями."
        ),
        "sources": ["ЧАСТЬ ПЕРВАЯ", "8. ПОСЛЕДНИЙ ДВОЙНИК"],
    },
    {
        "id": 8,
        "question": "Кто такая Ира Фатеева и какова её роль в истории?",
        "expected": (
            "Ира Фатеева — стенографистка, будущий секретарь особой комиссии Академии наук. "
            "Характеризуется как «кобра, полиглот и всезнайка». "
            "Позднее в романе упоминается её двойник."
        ),
        "sources": ["ЧАСТЬ ВТОРАЯ", "11. ОНИ ВИДЯТ, СЛЫШАТ И ЧУЮТ"],
    },
    {
        "id": 9,
        "question": "Что такое «фиолетовый туман» и где он был замечен?",
        "expected": (
            "Фиолетовый (багровый) туман — явление, подобное розовым облакам, "
            "при котором видимость сохранялась отличной, но всё окрашивалось в необычный цвет. "
            "Наблюдалось в американском городе Санд-Сити (Калифорния) в вечернее время."
        ),
        "sources": ["ЧАСТЬ ТРЕТЬЯ", "16. МОСКВА-ПАРИЖ"],
    },
    {
        "id": 10,
        "question": "Чем завершается роман и что происходит с двойником Анохина в финале?",
        "expected": (
            "В финале двойник (пришелец в облике Анохина) уходит, "
            "намекая что они улетают. Вано признаётся, что поначалу не узнал настоящего Юрку — "
            "двойник казался умнее. Дьячук отмечает феноменальную память двойника."
        ),
        "sources": ["ЧАСТЬ ЧЕТВЕРТАЯ", "32. НА ВЕКА!"],
    },
    # ── "не знаю"-тест: намеренно нерелевантный вопрос ──
    {
        "id": 11,
        "question": "Какова молекулярная масса воды?",
        "expected": "не знаю",
        "sources": [],
    },
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    rank:     int
    dist:     float
    chunk_id: str
    title:    str
    section:  str
    text:     str


@dataclass
class StructuredAnswer:
    """Parsed structured response from LLM."""
    answer:     str
    sources:    List[str]    # ["[struct_0012] Книга первая / 1", ...]
    quotes:     List[str]    # verbatim quote strings
    is_unknown: bool = False  # triggered by empty-context rule
    has_sources: bool = False
    has_quotes:  bool = False
    raw: str = ""            # original LLM output

    def __post_init__(self):
        self.has_sources = bool(self.sources and self.sources[0] != "(нет релевантных источников)")
        self.has_quotes  = bool(self.quotes  and self.quotes[0]  != "(нет цитат — фрагменты не прошли фильтр релевантности)")


# ---------------------------------------------------------------------------
# Retrieval  (Stage 1)
# ---------------------------------------------------------------------------
def retrieve(query: str, strategy: str, fetch_k: int, giga: GigaChat) -> List[RetrievedChunk]:
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
# Reranker / Filter  (Stage 2)
# ---------------------------------------------------------------------------
def _topic_coverage(query: str, chunks: List["RetrievedChunk"]) -> float:
    """
    Fraction of significant query words (>=4 chars) found in ANY chunk.
    Low coverage → question is off-topic for this corpus → "не знаю".
    """
    words = list(set(re.findall(r'\b\w{4,}\b', query.lower())))
    if not words:
        return 1.0
    all_text = " ".join(c.text.lower() for c in chunks)
    hits = sum(1 for w in words if w in all_text)
    return hits / len(words)


def _keyword_overlap(query: str, text: str) -> float:
    words = re.findall(r'\b\w{4,}\b', query.lower())
    if not words:
        return 0.0
    tl = text.lower()
    return sum(1 for w in words if w in tl) / len(words)


def rerank_filter(
    query:       str,
    chunks:      List[RetrievedChunk],
    threshold:   Optional[float],   # None → adaptive
    top_k_final: int,
) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], float]:
    """
    Returns (kept, rejected, applied_threshold).

    Threshold logic:
      • threshold=None  → adaptive: best_dist * AUTO_FACTOR
        Keeps chunks "close enough" to the best match.
        Works with any embedding scale (normalized or not).
      • threshold=float → absolute L2 cutoff (user-supplied).
    """
    if not chunks:
        return [], [], 0.0

    best_dist = chunks[0].dist   # FAISS results are sorted by distance

    if threshold is None:
        applied = best_dist * AUTO_FACTOR   # within 5% of best match
    else:
        applied = threshold

    kept:     List[Tuple[float, RetrievedChunk]] = []
    rejected: List[RetrievedChunk] = []

    for c in chunks:
        if c.dist > applied:
            rejected.append(c)
            continue
        # Score: normalised distance (lower=better) minus keyword overlap bonus
        norm_dist   = c.dist / applied
        kw_bonus    = _keyword_overlap(query, c.text) * 0.15
        final_score = norm_dist - kw_bonus
        kept.append((final_score, c))

    kept.sort(key=lambda x: x[0])
    filtered = []
    for new_rank, (_, c) in enumerate(kept[:top_k_final], 1):
        filtered.append(RetrievedChunk(new_rank, c.dist, c.chunk_id, c.title, c.section, c.text))
    return filtered, rejected, applied


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        header = f"[{c.chunk_id}] {c.title} / {c.section}  (dist={c.dist:.4f})"
        parts.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Structured answer parser
# ---------------------------------------------------------------------------
_SEC = re.DOTALL | re.IGNORECASE

def _extract_section(text: str, header: str) -> str:
    """Extract text between ## HEADER and the next ## or end-of-string."""
    pat = rf'##\s*{re.escape(header)}\s*\n(.*?)(?=\n##|\Z)'
    m = re.search(pat, text, _SEC)
    return m.group(1).strip() if m else ""


def parse_structured_answer(raw: str) -> StructuredAnswer:
    answer_text  = _extract_section(raw, "ОТВЕТ")
    sources_text = _extract_section(raw, "ИСТОЧНИКИ")
    quotes_text  = _extract_section(raw, "ЦИТАТЫ")

    # Fallback: if no ## markers at all, treat whole text as answer
    if not answer_text:
        answer_text = raw.strip()

    sources = [
        ln.strip().lstrip("-• ").strip()
        for ln in sources_text.splitlines()
        if ln.strip() and ln.strip() not in ("-", "•")
    ]

    quotes = []
    for ln in quotes_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith(">"):
            quotes.append(ln[1:].strip())
        else:
            quotes.append(ln)

    # Detect if LLM itself admitted it doesn't know
    _unkn_patterns = re.compile(
        r'не\s+знаю|не\s+упоминается|нет\s+в\s+фрагмент|не\s+содержит|'
        r'нет\s+информации|не\s+нашел|не\s+нашёл|отсутствует\s+в\s+тексте',
        re.IGNORECASE,
    )
    llm_says_unknown = bool(_unkn_patterns.search(answer_text[:200]))

    return StructuredAnswer(
        answer=answer_text,
        sources=sources,
        quotes=quotes,
        is_unknown=llm_says_unknown,
        raw=raw,
    )


def make_unknown_answer() -> StructuredAnswer:
    sa = parse_structured_answer(NO_CONTEXT_ANSWER)
    sa.is_unknown = True
    return sa


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
        temperature=0.2,
        max_tokens=1536,
    )
    return giga.chat(chat).choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Full RAG answer: retrieve → filter → LLM (or "не знаю")
# ---------------------------------------------------------------------------
def rag_answer(
    question:    str,
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
    giga:        GigaChat,
) -> Tuple[StructuredAnswer, List[RetrievedChunk], List[RetrievedChunk]]:
    """
    Returns (structured_answer, chunks_before_filter, chunks_after_filter).
    If no chunks pass filter → is_unknown=True, no LLM call.
    """
    chunks_raw = retrieve(question, strategy, fetch_k, giga)

    if use_rerank:
        chunks_final, rejected, applied_thr = rerank_filter(
            question, chunks_raw, threshold, top_k_final
        )
    else:
        chunks_final  = chunks_raw[:top_k_final]
        rejected      = chunks_raw[top_k_final:]
        applied_thr   = threshold or (chunks_raw[0].dist * AUTO_FACTOR if chunks_raw else 0.0)

    if not chunks_final:
        return make_unknown_answer(), chunks_raw, []

    # Topic coverage check: if query words barely appear in chunks → off-topic
    coverage = _topic_coverage(question, chunks_final)
    if coverage < TOPIC_COVER_MIN:
        return make_unknown_answer(), chunks_raw, chunks_final

    context = build_context(chunks_final)
    prompt  = RAG_TEMPLATE.format(context=context, question=question)
    raw     = ask_llm(prompt, giga)
    sa      = parse_structured_answer(raw)
    return sa, chunks_raw, chunks_final


# ---------------------------------------------------------------------------
# Interactive query
# ---------------------------------------------------------------------------
def run_query(
    question:    str,
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set")

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        print(f"\n{'='*72}")
        print(f"ВОПРОС: {question}")
        print(f"{'='*72}")
        print(f"  strategy={strategy}  fetch_k={fetch_k}  "
              f"top_k_final={top_k_final}  threshold={threshold}  rerank={use_rerank}")

        # ── No RAG ──
        print("\n[БЕЗ RAG]")
        no_rag_raw = ask_llm(NO_RAG_TEMPLATE.format(question=question), giga,
                             system="Ты — литературный ассистент.")
        print(textwrap.indent(no_rag_raw, "  "))

        # ── RAG ──
        sa, chunks_raw, chunks_final = rag_answer(
            question, strategy, fetch_k, top_k_final, threshold, use_rerank, giga
        )

        # Compute what threshold will actually be applied (for display)
        diag_thr = threshold if threshold is not None else (
            chunks_raw[0].dist * AUTO_FACTOR if chunks_raw else 0.0
        )
        thr_label = f"{diag_thr:.2f}" if threshold is not None else f"{diag_thr:.2f} (auto)"

        print(f"\n[STAGE 1] Получено {len(chunks_raw)} кандидатов  "
              f"(applied threshold={thr_label}):")
        for c in chunks_raw:
            flag = "ok" if c.dist <= diag_thr else "--"
            print(f"  [{flag}] dist={c.dist:.2f}  [{c.chunk_id}]  "
                  f"{c.title[:30]} / {c.section[:25]}")

        print(f"\n[STAGE 2] После фильтра: {len(chunks_final)} чанков  "
              f"({'НЕ ЗНАЮ' if sa.is_unknown else 'передаём в LLM'})")

        print(f"\n{'─'*72}")
        print(sa.raw if sa.raw else NO_CONTEXT_ANSWER)
        print(f"{'─'*72}")
        print(f"  has_sources={sa.has_sources}  has_quotes={sa.has_quotes}  "
              f"is_unknown={sa.is_unknown}")
        print()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    id:       int
    question: str
    expected: str
    sources:  List[str]

    chunks_before:  int = 0
    chunks_after:   int = 0
    dists_before:   List[float] = field(default_factory=list)
    retrieved_sources_raw:      List[str] = field(default_factory=list)
    retrieved_sources_filtered: List[str] = field(default_factory=list)

    answer_no_rag:           str = ""
    sa_raw:      StructuredAnswer = field(default_factory=lambda: StructuredAnswer("","",""))
    sa_filtered: StructuredAnswer = field(default_factory=lambda: StructuredAnswer("","",""))


def _source_match(expected: List[str], retrieved: List[str]) -> bool:
    flat = " ".join(retrieved).lower()
    return any(s.lower()[:10] in flat for s in expected)


def run_benchmark(
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set")

    results: List[EvalResult] = []

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        for q in EVAL_QUESTIONS:
            print(f"\n{'='*72}")
            print(f"[Q{q['id']:02d}] {q['question']}")
            print(f"{'='*72}")

            er = EvalResult(
                id=q["id"], question=q["question"],
                expected=q["expected"], sources=q["sources"],
            )

            # No RAG
            print("  → без RAG...")
            er.answer_no_rag = ask_llm(
                NO_RAG_TEMPLATE.format(question=q["question"]), giga,
                system="Ты — литературный ассистент."
            )

            # RAG without filter (baseline)
            print("  → RAG без фильтра...")
            chunks_raw = retrieve(q["question"], strategy, fetch_k, giga)
            er.chunks_before  = len(chunks_raw)
            er.dists_before   = [c.dist for c in chunks_raw]
            er.retrieved_sources_raw = [f"{c.chunk_id} / {c.title} / {c.section}" for c in chunks_raw]

            raw_top = chunks_raw[:top_k_final]
            if raw_top:
                ctx_raw = build_context(raw_top)
                sa_raw  = parse_structured_answer(
                    ask_llm(RAG_TEMPLATE.format(context=ctx_raw, question=q["question"]), giga)
                )
            else:
                sa_raw = make_unknown_answer()
            er.sa_raw = sa_raw

            # RAG with filter
            print("  → RAG с фильтром...")
            if use_rerank:
                chunks_final, _, _thr = rerank_filter(
                    q["question"], chunks_raw, threshold, top_k_final
                )
            else:
                chunks_final = raw_top

            er.chunks_after = len(chunks_final)
            er.retrieved_sources_filtered = [
                f"{c.chunk_id} / {c.title} / {c.section}" for c in chunks_final
            ]

            if chunks_final:
                ctx_filt = build_context(chunks_final)
                sa_filt  = parse_structured_answer(
                    ask_llm(RAG_TEMPLATE.format(context=ctx_filt, question=q["question"]), giga)
                )
            else:
                sa_filt = make_unknown_answer()
            er.sa_filtered = sa_filt

            results.append(er)

            # ── Print full comparison ──────────────────────────────────────
            W = 72
            print(f"\n  {'ОЖИДАЕМЫЙ ОТВЕТ':}")
            print(textwrap.indent(er.expected, "    "))

            print(f"\n  {'─'*W}")
            print(f"  БЕЗ RAG:")
            print(textwrap.indent(er.answer_no_rag, "    "))

            print(f"\n  {'─'*W}")
            print(f"  RAG БЕЗ ФИЛЬТРА  "
                  f"(чанков: {er.chunks_before}, "
                  f"src={sa_raw.has_sources}, qts={sa_raw.has_quotes}, unk={sa_raw.is_unknown}):")
            print(textwrap.indent(sa_raw.raw or sa_raw.answer, "    "))

            print(f"\n  {'─'*W}")
            print(f"  RAG С ФИЛЬТРОМ  "
                  f"(чанков: {er.chunks_before} → {er.chunks_after}, "
                  f"src={sa_filt.has_sources}, qts={sa_filt.has_quotes}, unk={sa_filt.is_unknown}):")
            print(textwrap.indent(sa_filt.raw or sa_filt.answer, "    "))
            print(f"  {'─'*W}\n")

    # ── Save report ──
    report_path = INDEX_DIR / "rag_benchmark.json"
    report = []
    for r in results:
        is_unknown_question = r.expected == "не знаю"
        report.append({
            "id": r.id,
            "question": r.question,
            "expected": r.expected,
            "expected_sources": r.sources,
            "is_unknown_question": is_unknown_question,
            "retrieval": {
                "chunks_before_filter": r.chunks_before,
                "chunks_after_filter":  r.chunks_after,
                "threshold": threshold,
                "dist_min": round(min(r.dists_before, default=0.0), 4),
                "dist_max": round(max(r.dists_before, default=0.0), 4),
            },
            "source_match_raw":      _source_match(r.sources, r.retrieved_sources_raw),
            "source_match_filtered": _source_match(r.sources, r.retrieved_sources_filtered),
            "answer_no_rag": r.answer_no_rag,
            # ── RAG without filter ──
            "rag_raw": {
                "answer":      r.sa_raw.answer,
                "sources":     r.sa_raw.sources,
                "quotes":      r.sa_raw.quotes,
                "has_sources": r.sa_raw.has_sources,
                "has_quotes":  r.sa_raw.has_quotes,
                "is_unknown":  r.sa_raw.is_unknown,
                "full_response": r.sa_raw.raw,
            },
            # ── RAG with filter / reranker ──
            "rag_filtered": {
                "answer":      r.sa_filtered.answer,
                "sources":     r.sa_filtered.sources,
                "quotes":      r.sa_filtered.quotes,
                "has_sources": r.sa_filtered.has_sources,
                "has_quotes":  r.sa_filtered.has_quotes,
                "is_unknown":  r.sa_filtered.is_unknown,
                "full_response": r.sa_filtered.raw,
            },
            # ── Checks ──
            "checks": {
                "rag_raw_has_sources":        r.sa_raw.has_sources,
                "rag_raw_has_quotes":         r.sa_raw.has_quotes,
                "rag_filtered_has_sources":   r.sa_filtered.has_sources,
                "rag_filtered_has_quotes":    r.sa_filtered.has_quotes,
                "unknown_rule_correct":
                    # for Q11: filtered should be is_unknown=True
                    # for Q1-10: filtered should NOT be is_unknown
                    (r.sa_filtered.is_unknown == is_unknown_question),
            },
        })

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n\nОтчёт → {report_path}")

    # ── Summary table ──
    print("\n" + "="*90)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'#':>3} | {'Вопрос (кратко)':<38} | "
          f"{'src↑':^5} | {'qts↑':^5} | "
          f"{'src↑f':^6} | {'qts↑f':^6} | "
          f"{'unk?':^5} | {'до':^4} | {'после':^5}")
    print("-"*90)
    for r in results:
        item = report[r.id - 1]  # indexed by id
        chk  = item["checks"]
        print(
            f"{r.id:>3} | {r.question[:37]:<38} | "
            f"{'✓' if item['rag_raw']['has_sources']      else '·':^5} | "
            f"{'✓' if item['rag_raw']['has_quotes']       else '·':^5} | "
            f"{'✓' if item['rag_filtered']['has_sources'] else '·':^6} | "
            f"{'✓' if item['rag_filtered']['has_quotes']  else '·':^6} | "
            f"{'✓' if chk['unknown_rule_correct']         else '✗':^5} | "
            f"{r.chunks_before:^4} | {r.chunks_after:^5}"
        )
    print("="*90)

    total  = len(results)
    hits   = lambda key: sum(1 for x in report if x["rag_filtered"][key])
    unk_ok = sum(1 for x in report if x["checks"]["unknown_rule_correct"])
    print(f"\n  has_sources (filtered): {hits('has_sources')}/{total}")
    print(f"  has_quotes  (filtered): {hits('has_quotes')}/{total}")
    print(f"  unknown rule correct  : {unk_ok}/{total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline with structured output + reranker")
    parser.add_argument("--query",       default=None)
    parser.add_argument("--strategy",    default="struct", choices=["fixed", "struct"])
    parser.add_argument("--top_k",       default=TOP_K_RETRIEVE, type=int)
    parser.add_argument("--top_k_final", default=TOP_K_FINAL,    type=int)
    parser.add_argument("--threshold",   default=None, type=float,
                        help="Absolute L2 cutoff. Default: adaptive (best_dist * AUTO_FACTOR)")
    parser.add_argument("--no_rerank",   action="store_true")
    parser.add_argument("--eval",        action="store_true")
    args = parser.parse_args()

    use_rerank = not args.no_rerank

    if args.query:
        run_query(args.query, args.strategy, args.top_k,
                  args.top_k_final, args.threshold, use_rerank)
    else:
        run_benchmark(args.strategy, args.top_k,
                      args.top_k_final, args.threshold, use_rerank)


if __name__ == "__main__":
    main()
