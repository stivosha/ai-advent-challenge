"""
Local RAG — Optimized (GigaChat Embeddings + Gemma3 via Ollama)
===============================================================
Extends local_rag.py with tuning profiles, quantization comparison,
an improved prompt template, and rich performance metrics.

Usage:
  python local_rag_optimized.py                          # compare profiles on full benchmark
  python local_rag_optimized.py --query "кто такой Зернов?"
  python local_rag_optimized.py --profile optimized      # single profile
  python local_rag_optimized.py --models gemma3:1b,gemma3:4b,gemma3  # quant comparison
  python local_rag_optimized.py --eval --profile fast    # eval only, fast profile

Env:
  GIGACHAT_API_KEY=<your_key>
  OLLAMA_URL=http://localhost:11434
  OLLAMA_MODEL=gemma3
"""
from __future__ import annotations

import sys
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import json
import time
import sqlite3
import textwrap
import argparse
import requests
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from gigachat import GigaChat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

INDEX_DIR   = Path(__file__).parent / "index"
EMBED_MODEL = "Embeddings"

OLLAMA_BASE  = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3")
OLLAMA_CHAT  = f"{OLLAMA_BASE}/api/chat"

TOP_K_RETRIEVE = 10
TOP_K_FINAL    = 5
AUTO_FACTOR    = 1.05
DIST_THRESHOLD = None
TOPIC_COVER_MIN = 0.10

# ---------------------------------------------------------------------------
# Tuning profiles
# ---------------------------------------------------------------------------
# Baseline reproduces the original local_rag.py settings exactly.
# Optimized: lower temperature for factual precision, explicit context window,
#            reduced max tokens (3-section answer fits in ~600 tokens),
#            nucleus sampling + repeat penalty to cut rambling.
# Fast:      smallest quantized model available (gemma3:1b).

PROFILES: Dict[str, dict] = {
    "baseline": {
        "model":          OLLAMA_MODEL,
        "temperature":    0.2,
        "num_predict":    1536,
        "num_ctx":        2048,   # Ollama default
        "top_p":          None,
        "repeat_penalty": None,
        "description":    "Original settings (temperature=0.2, num_predict=1536)",
    },
    "optimized": {
        "model":          OLLAMA_MODEL,
        "temperature":    0.05,   # near-deterministic — best for factual RAG
        "num_predict":    768,    # 3-section answer is ~400-600 tokens
        "num_ctx":        4096,   # wider context window for long documents
        "top_p":          0.85,   # nucleus sampling avoids low-probability tokens
        "repeat_penalty": 1.15,   # penalise repetition
        "description":    "Optimized (temp=0.05, ctx=4096, reduced tokens, nucleus+repeat)",
    },
    "fast": {
        "model":          "gemma3:1b",   # smallest quantized variant
        "temperature":    0.05,
        "num_predict":    512,
        "num_ctx":        2048,
        "top_p":          0.85,
        "repeat_penalty": 1.15,
        "description":    "Fast/quantized (gemma3:1b — Q4 by default in Ollama)",
    },
}

# ---------------------------------------------------------------------------
# Prompt templates — baseline vs optimised
# ---------------------------------------------------------------------------

# Original system prompt (baseline)
SYSTEM_PROMPT_BASELINE = (
    "Ты — литературный ассистент по роману «Четыре всадника Апокалипсиса» "
    "Бласко Ибаньеса. "
    "КРИТИЧЕСКИ ВАЖНО: отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов текста. "
    "ЗАПРЕЩЕНО использовать собственные знания, энциклопедические факты "
    "или информацию вне фрагментов. "
    "Если вопрос не относится к роману или ответа нет во фрагментах — "
    "раздел ## ОТВЕТ должен начинаться словами 'Не знаю' и объяснить почему. "
    "ОБЯЗАТЕЛЬНО используй ровно три раздела: ## ОТВЕТ, ## ИСТОЧНИКИ, ## ЦИТАТЫ."
)

# Optimised system prompt: concise, clearer constraints, avoids repeated
# emphasis that can confuse smaller models.
SYSTEM_PROMPT_OPTIMIZED = (
    "Ты — ассистент по роману «Они» Александра Абрамова. "
    "Правило 1: отвечай только по предоставленным фрагментам, никаких внешних знаний. "
    "Правило 2: если ответа нет во фрагментах — ## ОТВЕТ начни с «Не знаю». "
    "Правило 3: строго три раздела — ## ОТВЕТ / ## ИСТОЧНИКИ / ## ЦИТАТЫ."
)

# Original RAG template (baseline)
RAG_TEMPLATE_BASELINE = """\
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

# Optimised RAG template: removes Unicode decorators that cost tokens,
# shortens boilerplate, gives explicit length hints to the model.
RAG_TEMPLATE_OPTIMIZED = """\
ФРАГМЕНТЫ:
{context}

---
ВОПРОС: {question}

Ответ в трёх разделах (не более 3 предложений в каждом):

## ОТВЕТ
(только по фрагментам выше, 2-4 предложения)

## ИСТОЧНИКИ
(список: - [chunk_id] Раздел)

## ЦИТАТЫ
(2-3 цитаты, каждая строка начинается с >)
"""

NO_CONTEXT_ANSWER = """\
## ОТВЕТ
Не знаю. По данному вопросу в тексте романа не найдено релевантных фрагментов.

## ИСТОЧНИКИ
(нет релевантных источников)

## ЦИТАТЫ
(нет цитат)
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
    {
        "id": 11,
        "question": "Какова молекулярная масса воды?",
        "expected": "не знаю",
        "sources": [],
    },
]

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ensure_ollama_running() -> None:
    try:
        requests.get(OLLAMA_BASE, timeout=2)
        return
    except requests.exceptions.ConnectionError:
        pass

    print("Запускаю ollama...", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        time.sleep(0.5)
        try:
            requests.get(OLLAMA_BASE, timeout=1)
            print("ollama запущена.\n", flush=True)
            return
        except requests.exceptions.ConnectionError:
            pass
    print("Не удалось запустить ollama. Запусти вручную: ollama serve")
    sys.exit(1)


@dataclass
class LLMMetrics:
    wall_ms:          int   = 0
    prompt_tokens:    int   = 0
    generated_tokens: int   = 0
    tokens_per_sec:   float = 0.0
    # Ollama reports eval_duration in nanoseconds
    eval_duration_ms: int   = 0

    def summary(self) -> str:
        return (
            f"wall={self.wall_ms}ms  "
            f"prompt_tok={self.prompt_tokens}  "
            f"gen_tok={self.generated_tokens}  "
            f"tok/s={self.tokens_per_sec:.1f}"
        )


def _build_options(profile: dict) -> dict:
    opts: dict = {
        "temperature": profile["temperature"],
        "num_predict": profile["num_predict"],
        "num_ctx":     profile["num_ctx"],
    }
    if profile.get("top_p") is not None:
        opts["top_p"] = profile["top_p"]
    if profile.get("repeat_penalty") is not None:
        opts["repeat_penalty"] = profile["repeat_penalty"]
    return opts


def ask_llm(
    prompt:    str,
    system:    str,
    profile:   dict,
    timeout:   int = 180,
) -> Tuple[str, LLMMetrics]:
    payload = {
        "model":    profile["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream":  False,
        "options": _build_options(profile),
    }

    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_CHAT, json=payload, timeout=timeout)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    resp.raise_for_status()

    data    = resp.json()
    content = data["message"]["content"].strip()

    m = LLMMetrics(wall_ms=wall_ms)
    m.prompt_tokens    = data.get("prompt_eval_count", 0)
    m.generated_tokens = data.get("eval_count", 0)
    eval_dur_ns        = data.get("eval_duration", 0)
    m.eval_duration_ms = eval_dur_ns // 1_000_000
    if eval_dur_ns > 0:
        m.tokens_per_sec = m.generated_tokens / (eval_dur_ns / 1e9)

    return content, m


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
    answer:      str
    sources:     List[str]
    quotes:      List[str]
    is_unknown:  bool = False
    has_sources: bool = False
    has_quotes:  bool = False
    raw:         str  = ""

    def __post_init__(self):
        self.has_sources = bool(self.sources and self.sources[0] != "(нет релевантных источников)")
        self.has_quotes  = bool(self.quotes  and self.quotes[0]  not in (
            "(нет цитат)", "(нет цитат — фрагменты не прошли фильтр релевантности)"
        ))


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

def _topic_coverage(query: str, chunks: List[RetrievedChunk]) -> float:
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
    threshold:   Optional[float],
    top_k_final: int,
) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], float]:
    if not chunks:
        return [], [], 0.0

    best_dist = chunks[0].dist
    applied   = best_dist * AUTO_FACTOR if threshold is None else threshold

    kept:     List[Tuple[float, RetrievedChunk]] = []
    rejected: List[RetrievedChunk] = []

    for c in chunks:
        if c.dist > applied:
            rejected.append(c)
            continue
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
    pat = rf'##\s*{re.escape(header)}\s*\n(.*?)(?=\n##|\Z)'
    m   = re.search(pat, text, _SEC)
    return m.group(1).strip() if m else ""


def parse_structured_answer(raw: str) -> StructuredAnswer:
    answer_text  = _extract_section(raw, "ОТВЕТ")
    sources_text = _extract_section(raw, "ИСТОЧНИКИ")
    quotes_text  = _extract_section(raw, "ЦИТАТЫ")

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
        quotes.append(ln[1:].strip() if ln.startswith(">") else ln)

    _unkn = re.compile(
        r'не\s+знаю|не\s+упоминается|нет\s+в\s+фрагмент|не\s+содержит|'
        r'нет\s+информации|не\s+нашел|не\s+нашёл|отсутствует\s+в\s+тексте',
        re.IGNORECASE,
    )
    return StructuredAnswer(
        answer=answer_text,
        sources=sources,
        quotes=quotes,
        is_unknown=bool(_unkn.search(answer_text[:200])),
        raw=raw,
    )


def make_unknown_answer() -> StructuredAnswer:
    sa = parse_structured_answer(NO_CONTEXT_ANSWER)
    sa.is_unknown = True
    return sa


# ---------------------------------------------------------------------------
# Profile-aware RAG answer
# ---------------------------------------------------------------------------

def rag_answer(
    question:    str,
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
    giga:        GigaChat,
    profile:     dict,
    use_optimized_prompt: bool = True,
) -> Tuple[StructuredAnswer, List[RetrievedChunk], List[RetrievedChunk], LLMMetrics]:
    chunks_raw = retrieve(question, strategy, fetch_k, giga)

    if use_rerank:
        chunks_final, _, _ = rerank_filter(question, chunks_raw, threshold, top_k_final)
    else:
        chunks_final = chunks_raw[:top_k_final]

    if not chunks_final:
        return make_unknown_answer(), chunks_raw, [], LLMMetrics()

    if _topic_coverage(question, chunks_final) < TOPIC_COVER_MIN:
        return make_unknown_answer(), chunks_raw, chunks_final, LLMMetrics()

    context = build_context(chunks_final)

    if use_optimized_prompt:
        sys_prompt = SYSTEM_PROMPT_OPTIMIZED
        tmpl       = RAG_TEMPLATE_OPTIMIZED
    else:
        sys_prompt = SYSTEM_PROMPT_BASELINE
        tmpl       = RAG_TEMPLATE_BASELINE

    prompt = tmpl.format(context=context, question=question)
    raw, metrics = ask_llm(prompt, sys_prompt, profile)
    sa = parse_structured_answer(raw)
    return sa, chunks_raw, chunks_final, metrics


# ---------------------------------------------------------------------------
# Single profile benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    id:             int
    question:       str
    expected:       str
    sources:        List[str]
    profile_name:   str
    chunks_before:  int = 0
    chunks_after:   int = 0
    sa:             StructuredAnswer = field(default_factory=lambda: StructuredAnswer("", [], []))
    metrics:        LLMMetrics       = field(default_factory=LLMMetrics)
    source_match:   bool = False


def _source_match(expected: List[str], retrieved: List[str]) -> bool:
    flat = " ".join(retrieved).lower()
    return any(s.lower()[:10] in flat for s in expected)


def run_profile_benchmark(
    profile_name: str,
    strategy:     str,
    fetch_k:      int,
    top_k_final:  int,
    threshold:    Optional[float],
    use_rerank:   bool,
    giga:         GigaChat,
) -> List[BenchResult]:
    profile = PROFILES[profile_name]
    use_opt = (profile_name != "baseline")

    results: List[BenchResult] = []
    print(f"\n{'#'*72}")
    print(f"ПРОФИЛЬ: {profile_name}  — {profile['description']}")
    print(f"  model={profile['model']}  temp={profile['temperature']}  "
          f"num_predict={profile['num_predict']}  num_ctx={profile['num_ctx']}")
    print(f"  top_p={profile.get('top_p')}  repeat_penalty={profile.get('repeat_penalty')}")
    print(f"  prompt={'OPTIMIZED' if use_opt else 'BASELINE'}")
    print(f"{'#'*72}")

    for q in EVAL_QUESTIONS:
        print(f"  [Q{q['id']:02d}] {q['question'][:60]}...", end=" ", flush=True)
        sa, chunks_raw, chunks_final, metrics = rag_answer(
            q["question"], strategy, fetch_k, top_k_final, threshold, use_rerank,
            giga, profile, use_opt,
        )
        src_retrieved = [f"{c.chunk_id} / {c.title} / {c.section}" for c in chunks_final]
        match = _source_match(q["sources"], src_retrieved)

        br = BenchResult(
            id=q["id"],
            question=q["question"],
            expected=q["expected"],
            sources=q["sources"],
            profile_name=profile_name,
            chunks_before=len(chunks_raw),
            chunks_after=len(chunks_final),
            sa=sa,
            metrics=metrics,
            source_match=match,
        )
        results.append(br)
        print(f"✓  {metrics.summary()}")

    return results


# ---------------------------------------------------------------------------
# Multi-profile comparison benchmark
# ---------------------------------------------------------------------------

def run_comparison_benchmark(
    profiles:     List[str],
    strategy:     str,
    fetch_k:      int,
    top_k_final:  int,
    threshold:    Optional[float],
    use_rerank:   bool,
) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set (needed for embeddings)")

    ensure_ollama_running()

    all_results: Dict[str, List[BenchResult]] = {}

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        for pname in profiles:
            if pname not in PROFILES:
                print(f"WARNING: unknown profile '{pname}', skipping")
                continue
            all_results[pname] = run_profile_benchmark(
                pname, strategy, fetch_k, top_k_final, threshold, use_rerank, giga
            )

    _print_comparison(all_results)
    _save_report(all_results)


def _print_comparison(all_results: Dict[str, List[BenchResult]]) -> None:
    profiles = list(all_results.keys())
    n_q      = len(EVAL_QUESTIONS)

    print("\n" + "="*110)
    print("ИТОГОВОЕ СРАВНЕНИЕ ПРОФИЛЕЙ")
    print("="*110)

    # Per-question table
    hdr = f"{'#':>3} | {'Вопрос':^40}"
    for p in profiles:
        hdr += f" | {p:^20}"
    print(hdr)
    print("-"*110)

    for qi in range(n_q):
        q_short = EVAL_QUESTIONS[qi]["question"][:38]
        row = f"{qi+1:>3} | {q_short:<40}"
        for p in profiles:
            br = all_results[p][qi]
            is_unk_q = br.expected == "не знаю"
            unk_ok   = br.sa.is_unknown == is_unk_q
            row += (
                f" | src={'✓' if br.sa.has_sources else '·'}"
                f" qts={'✓' if br.sa.has_quotes  else '·'}"
                f" unk={'✓' if unk_ok else '✗'}"
                f" {br.metrics.wall_ms:>5}ms"
                f" {br.metrics.tokens_per_sec:>5.1f}t/s"
            )
        print(row)

    print("="*110)

    # Aggregate metrics
    print(f"\n{'Метрика':<30}", end="")
    for p in profiles:
        print(f"  {p:^22}", end="")
    print()
    print("-"*110)

    def agg(pname: str, key_fn) -> str:
        vals = [key_fn(r) for r in all_results[pname]]
        return f"{sum(vals)/len(vals):.1f}" if vals else "—"

    def cnt(pname: str, key_fn) -> str:
        return f"{sum(1 for r in all_results[pname] if key_fn(r))}/{n_q}"

    rows = [
        ("has_sources (filtered)",    lambda r: r.sa.has_sources),
        ("has_quotes  (filtered)",    lambda r: r.sa.has_quotes),
        ("unknown rule correct",      lambda r: r.sa.is_unknown == (r.expected == "не знаю")),
        ("source_match",              lambda r: r.source_match),
    ]
    for label, fn in rows:
        print(f"  {label:<28}", end="")
        for p in profiles:
            print(f"  {cnt(p, fn):^22}", end="")
        print()

    print(f"  {'avg wall_ms':<28}", end="")
    for p in profiles:
        print(f"  {agg(p, lambda r: r.metrics.wall_ms):^22}", end="")
    print()

    print(f"  {'avg tok/s':<28}", end="")
    for p in profiles:
        print(f"  {agg(p, lambda r: r.metrics.tokens_per_sec):^22}", end="")
    print()

    print(f"  {'avg gen_tokens':<28}", end="")
    for p in profiles:
        print(f"  {agg(p, lambda r: r.metrics.generated_tokens):^22}", end="")
    print()

    print("="*110)


def _save_report(all_results: Dict[str, List[BenchResult]]) -> None:
    report = {}
    for pname, results in all_results.items():
        report[pname] = []
        for r in results:
            is_unk_q = r.expected == "не знаю"
            report[pname].append({
                "id":             r.id,
                "question":       r.question,
                "expected":       r.expected,
                "expected_sources": r.sources,
                "chunks_before":  r.chunks_before,
                "chunks_after":   r.chunks_after,
                "source_match":   r.source_match,
                "metrics": {
                    "wall_ms":          r.metrics.wall_ms,
                    "prompt_tokens":    r.metrics.prompt_tokens,
                    "generated_tokens": r.metrics.generated_tokens,
                    "tokens_per_sec":   round(r.metrics.tokens_per_sec, 2),
                },
                "answer": {
                    "text":        r.sa.answer,
                    "sources":     r.sa.sources,
                    "quotes":      r.sa.quotes,
                    "has_sources": r.sa.has_sources,
                    "has_quotes":  r.sa.has_quotes,
                    "is_unknown":  r.sa.is_unknown,
                    "unk_correct": r.sa.is_unknown == is_unk_q,
                    "full":        r.sa.raw,
                },
            })

    path = INDEX_DIR / "local_rag_optimized_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт сохранён → {path}")


# ---------------------------------------------------------------------------
# Custom model comparison (quantization test)
# ---------------------------------------------------------------------------

def run_model_comparison(
    models:      List[str],
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
    n_questions: int = 3,
) -> None:
    """Quick benchmark across different model variants (quantization levels)."""
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set (needed for embeddings)")

    ensure_ollama_running()

    sample_qs = EVAL_QUESTIONS[:n_questions]

    print(f"\n{'='*80}")
    print(f"ТЕСТ КВАНТОВАНИЯ  ({n_questions} вопросов из {len(EVAL_QUESTIONS)})")
    print(f"Модели: {', '.join(models)}")
    print(f"{'='*80}")

    row_header = f"{'Модель':<20} | {'Q':^3} | {'wall_ms':>8} | {'tok/s':>7} | {'gen_tok':>7} | src | qts | unk"
    print(row_header)
    print("-"*80)

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        for model in models:
            profile = dict(PROFILES["optimized"])
            profile["model"] = model

            walls, tps, gens = [], [], []
            has_src, has_qts, unk_ok = 0, 0, 0

            for q in sample_qs:
                chunks_raw = retrieve(q["question"], strategy, fetch_k, giga)
                if use_rerank:
                    chunks_final, _, _ = rerank_filter(q["question"], chunks_raw, threshold, top_k_final)
                else:
                    chunks_final = chunks_raw[:top_k_final]

                if not chunks_final:
                    continue

                context = build_context(chunks_final)
                prompt  = RAG_TEMPLATE_OPTIMIZED.format(context=context, question=q["question"])

                try:
                    raw, m = ask_llm(prompt, SYSTEM_PROMPT_OPTIMIZED, profile, timeout=240)
                except Exception as e:
                    print(f"  ERROR on {model}: {e}")
                    continue

                sa = parse_structured_answer(raw)
                is_unk_q = q["expected"] == "не знаю"
                walls.append(m.wall_ms)
                tps.append(m.tokens_per_sec)
                gens.append(m.generated_tokens)
                has_src += int(sa.has_sources)
                has_qts += int(sa.has_quotes)
                unk_ok  += int(sa.is_unknown == is_unk_q)

            n = len(walls) or 1
            print(
                f"{model:<20} | {n:^3} | {sum(walls)//n:>8} | "
                f"{sum(tps)/n:>7.1f} | {sum(gens)//n:>7} | "
                f"{has_src}/{n} | {has_qts}/{n} | {unk_ok}/{n}"
            )

    print("="*80)


# ---------------------------------------------------------------------------
# Interactive query (single profile)
# ---------------------------------------------------------------------------

def run_query(
    question:    str,
    profile_name: str,
    strategy:    str,
    fetch_k:     int,
    top_k_final: int,
    threshold:   Optional[float],
    use_rerank:  bool,
) -> None:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError("GIGACHAT_API_KEY not set (needed for embeddings)")

    ensure_ollama_running()

    profile = PROFILES[profile_name]
    use_opt = (profile_name != "baseline")

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        print(f"\n{'='*72}")
        print(f"ВОПРОС: {question}")
        print(f"{'='*72}")
        print(f"  profile={profile_name}  model={profile['model']}  "
              f"strategy={strategy}  top_k={fetch_k}→{top_k_final}  rerank={use_rerank}")
        print(f"  {profile['description']}")

        sa, chunks_raw, chunks_final, metrics = rag_answer(
            question, strategy, fetch_k, top_k_final, threshold, use_rerank,
            giga, profile, use_opt,
        )

        print(f"\n[STAGE 1] {len(chunks_raw)} кандидатов:")
        for c in chunks_raw:
            flag = "ok" if c in chunks_final else "--"
            print(f"  [{flag}] dist={c.dist:.4f}  [{c.chunk_id}]  "
                  f"{c.title[:30]} / {c.section[:25]}")

        print(f"\n[STAGE 2] {len(chunks_final)} чанков после фильтра")
        print(f"\n{'─'*72}")
        print(sa.raw if sa.raw else NO_CONTEXT_ANSWER)
        print(f"{'─'*72}")
        print(f"  {metrics.summary()}")
        print(f"  has_sources={sa.has_sources}  has_quotes={sa.has_quotes}  "
              f"is_unknown={sa.is_unknown}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local RAG — optimized profiles + quantization comparison"
    )
    parser.add_argument("--query",       default=None,
                        help="одиночный вопрос вместо бенчмарка")
    parser.add_argument("--profile",     default=None,
                        choices=list(PROFILES.keys()),
                        help="запустить один профиль вместо сравнения")
    parser.add_argument("--profiles",    default="baseline,optimized",
                        help="запятая-разделённый список профилей для сравнения")
    parser.add_argument("--models",      default=None,
                        help="запятая-разделённые модели для теста квантования, "
                             "напр. gemma3:1b,gemma3:4b,gemma3")
    parser.add_argument("--strategy",    default="struct", choices=["fixed", "struct"])
    parser.add_argument("--top_k",       default=TOP_K_RETRIEVE, type=int)
    parser.add_argument("--top_k_final", default=TOP_K_FINAL,    type=int)
    parser.add_argument("--threshold",   default=None, type=float)
    parser.add_argument("--no_rerank",   action="store_true")
    parser.add_argument("--eval",        action="store_true",
                        help="только бенчмарк (то же что и без --query)")
    args = parser.parse_args()

    use_rerank = not args.no_rerank

    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        run_model_comparison(
            model_list, args.strategy, args.top_k, args.top_k_final,
            args.threshold, use_rerank,
        )
    elif args.query:
        pname = args.profile or "optimized"
        run_query(
            args.query, pname, args.strategy, args.top_k,
            args.top_k_final, args.threshold, use_rerank,
        )
    elif args.profile:
        creds = os.environ.get("GIGACHAT_API_KEY")
        if not creds:
            raise RuntimeError("GIGACHAT_API_KEY not set")
        ensure_ollama_running()
        with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
            results = run_profile_benchmark(
                args.profile, args.strategy, args.top_k, args.top_k_final,
                args.threshold, use_rerank, giga,
            )
        _print_comparison({args.profile: results})
        _save_report({args.profile: results})
    else:
        profile_list = [p.strip() for p in args.profiles.split(",") if p.strip()]
        run_comparison_benchmark(
            profile_list, args.strategy, args.top_k, args.top_k_final,
            args.threshold, use_rerank,
        )


if __name__ == "__main__":
    main()
