#!/usr/bin/env python3
# RAG4REPORT VERSION
# Adapted from minna_judge.py (mission3/kiddie) for the rag4reports-2026 dataset.
#
# KEY DIFFERENCE FROM ORIGINAL:
#   The original code (anise format) had document text embedded directly in each
#   Report object under response.documents. The rag4reports-2026 format (adventure-
#   continue style) does NOT embed document text — it only stores doc IDs in
#   response.references and fragment.citations. The actual text lives in external
#   corpus files (~/ragtime1/eng-docs.jsonl etc.).
#
#   Fix: _load_doc_store() + _inject_documents() below load the corpus into RAM
#   once and populate response.documents before any scoring logic runs.
#   This is a no-op for anise-style runs that already have documents populated,
#   so the code stays compatible with both formats.

import warnings
import logging

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import asyncio
import json
import dataclasses
import os
from autojudge_base import (
    LlmConfigProtocol,
    Report,
    Request,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    MeasureSpec,
    Qrels,
    QrelsSpec,
    build_qrels,
    doc_id_md5,
    NuggetBanks,
    NuggetBanksProtocol,
)
from autojudge_base.nugget_data import (
    NuggetBank,
    NuggetQuestion,
)
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, OpenAIMinimaLlm
from transformers import AutoModelForSequenceClassification, T5Tokenizer
from sentence_transformers import CrossEncoder
import torch


# =============================================================================
# FIX: Document store for rag4report (adventure-style runs)
# =============================================================================
#
# WHY THIS EXISTS:
#   adventure-continue JSONL has no "documents" field. Each Report object handed
#   to the judge will have response.documents = None or {}, so all attribution
#   NLI and citation accuracy scoring would silently produce 0 for every run.
#
# HOW IT WORKS:
#   1. _TASK_TO_CORPUS maps the "task" metadata field to the right corpus file.
#      e.g. topic with task="english" → ~/ragtime1/eng-docs.jsonl
#
#   2. _load_doc_store(task) reads that JSONL file once and builds a dict:
#         { doc_id -> doc_text }
#      The result is stored in the module-level _doc_store_cache dict so the
#      1.6GB file is only read ONCE per process, no matter how many responses
#      need documents. Subsequent calls for the same task hit the in-memory
#      dict instantly (RAM lookup, nanoseconds vs disk I/O).
#      NOTE: this cache lives only in RAM — it disappears when the process ends.
#      That is fine because the expensive work (NLI scores, LLM calls) is
#      separately persisted to disk via load_cache/save_cache.
#
#   3. _inject_documents(responses) iterates over all Report objects. For any
#      response that already has documents (anise-style), it does nothing.
#      For adventure-style responses with empty documents, it collects all
#      cited doc IDs from fragment.citations, looks them up in the store,
#      and injects _Doc stubs with a .text attribute — matching the interface
#      the rest of the code expects (doc.text).

# Path to the corpus directory (outside auto-judge-starter-kit)
_DOCS_DIR = os.path.expanduser("~/ragtime1")

# Map task name → corpus file path
_TASK_TO_CORPUS: Dict[str, str] = {
    "english": os.path.join(_DOCS_DIR, "eng-docs.jsonl"),
    "arabic":  os.path.join(_DOCS_DIR, "arb-docs.jsonl"),
    "russian": os.path.join(_DOCS_DIR, "rus-docs.jsonl"),
    "chinese": os.path.join(_DOCS_DIR, "zho-docs.jsonl"),
}

# Module-level in-memory cache: task -> {doc_id -> doc_text}
# Populated lazily on first call per task, reused for all subsequent calls.
_doc_store_cache: Dict[str, Dict[str, str]] = {}


def _load_doc_store(task: str) -> Dict[str, str]:
    """
    Load and return the document corpus for a given task language.
    Args:
        task: value of metadata.task, e.g. "english", "arabic", etc.

    Returns:
        dict mapping doc_id -> doc_text (empty dict if corpus not found)
    """
    # Return immediately if already loaded for this task
    if task in _doc_store_cache:
        return _doc_store_cache[task]

    # Default to english if task not recognised
    path = _TASK_TO_CORPUS.get(task, _TASK_TO_CORPUS["english"])

    store: Dict[str, str] = {}
    if not os.path.exists(path):
        print(f"WARNING _load_doc_store: corpus file not found at {path} (task={task!r})")
        _doc_store_cache[task] = store
        return store

    print(f"_load_doc_store: loading corpus for task={task!r} from {path} ...")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            store[doc["id"]] = doc["text"]

    print(f"_load_doc_store: loaded {len(store):,} docs for task={task!r} (now cached in RAM)")
    _doc_store_cache[task] = store
    return store


class _Doc:
    """
    Minimal document stub that matches the interface the judge code expects.
    The original anise Report objects had autojudge_base Document instances
    with a .text attribute. We replicate just that attribute so all existing
    doc.text accesses work unchanged.
    """
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def _inject_documents(responses: List[Report]) -> None:
    """
    Populate response.documents for adventure-style runs that don't embed
    document text in the Report object.

    For each response:
      - If response.documents is already populated (anise-style): skip (no-op).
      - If response.documents is empty/None (adventure-style): collect all
        doc IDs cited across all fragments, look them up in the corpus store,
        and inject _Doc stubs into response.documents.

    This must be called once after responses = list(rag_responses) and before
    any scoring stage that accesses response.documents.
    """
    injected_count = 0
    missing_ids: List[str] = []

    for response in responses:
        # Already populated — anise-style run, nothing to do
        if response.documents:
            continue

        task = getattr(response.metadata, "task", "english") or "english"
        store = _load_doc_store(task)
        if not store:
            continue

        # Collect all doc IDs cited in this response's fragments.
        # Citations are dicts {doc_id: retrieval_score} in adventure format.
        # We only need the keys (doc IDs) — the float score is the BM25/retrieval
        # ranking score from the RAG system, not used in our NLI scoring.
        cited_ids = list({
            cid
            for frag in (response.responses or [])
            for cid in (frag.citations or {}).keys()
        })

        # Build documents dict: {doc_id -> _Doc(text)}
        # Log any IDs not found in the corpus (shouldn't happen but good to know)
        docs = {}
        for doc_id in cited_ids:
            if doc_id in store:
                docs[doc_id] = _Doc(store[doc_id])
            else:
                missing_ids.append(doc_id)

        response.documents = docs
        injected_count += 1

    print(f"_inject_documents: injected documents for {injected_count} responses")
    if missing_ids:
        print(f"  WARNING: {len(missing_ids)} cited doc IDs not found in corpus: "
              f"{missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}")


# =============================================================================
# NLI Models 
# =============================================================================

class _VectaraHHEM:
    """Wrapper around HHEMv2 with .predict() matching CrossEncoder interface."""

    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        ).to(self._device)
        self._model.eval()
        print(f"_VectaraHHEM: using device={self._device}")

    def predict(self, sentence_pairs, batch_size: int = 256):
        scores = []
        for i in range(0, len(sentence_pairs), batch_size):
            batch = sentence_pairs[i:i + batch_size]
            inputs = self._tokenizer(
                [p[0] for p in batch], [p[1] for p in batch],
                return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            scores.extend(torch.sigmoid(logits[:, 0]).cpu().tolist())
        return scores


nli_model = _VectaraHHEM()

# deberta NLI for citation accuracy (run4 approach: strict 3-class entailment)
citation_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
print("citation_nli_model: loaded cross-encoder/nli-deberta-v3-base")


# FINAL = 0.5*(max(cite, attr) + ret)
MINIMAL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("ATTRIBUTION_SCORE"),
    MeasureSpec("CITATION_ACCURACY"),
    MeasureSpec("RETRIEVAL_QUALITY"),
    MeasureSpec("RESPONSE_NUGGET_SCORE"),
    MeasureSpec("FINAL"),
))


class GradeRecord:
    """Simple record for qrels building."""
    def __init__(self, topic_id: str, text: str, grade: int):
        self.topic_id = topic_id
        self.text = text
        self.grade = grade


MINIMAL_QRELS_SPEC = QrelsSpec[GradeRecord](
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: doc_id_md5(r.text),
    grade=lambda r: r.grade,
    on_duplicate="keep_max",
)


# =============================================================================
# MinnaNuggetCreator 
# =============================================================================

class MinnaNuggetCreator:
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        questions_per_topic: int = 10,
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:

        limit = kwargs.get("max_nuggets", questions_per_topic)

        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)

        # collect diverse response samples to inform nugget generation
        responses = list(rag_responses)
        response_samples: Dict[str, List[str]] = {}
        for resp in responses:
            tid = resp.metadata.topic_id
            text = resp.get_report_text()
            if tid not in response_samples:
                response_samples[tid] = []
            if len(response_samples[tid]) < 3 and len(text) > 50:
                response_samples[tid].append(text[:500])

        banks: List[NuggetBank] = []
        requests = []

        for topic in rag_topics:
            if topic.title:
                context = f"Title: {topic.title}\n\nQuestion: {topic.problem_statement}"
            else:
                context = topic.problem_statement

            samples = response_samples.get(topic.request_id, [])
            if samples:
                sample_text = "\n---\n".join(samples)
                context += (
                    f"\n\nHere are some example system responses for reference "
                    f"(use these to identify what aspects differentiate good vs bad answers):\n"
                    f"{sample_text}"
                )

            requests.append(MinimaLlmRequest(
                request_id=topic.request_id,
                messages=[
                    {"role": "system", "content": (
                        "You are an evaluation expert. Given a question, break it into specific subquestions "
                        "that a complete answer must address. Each subquestion should be atomic (cover only 1 aspect), "
                        "independently assessable (don't need further subquestions to answer it), "
                        "and cover a distinct aspect of the question. "
                        "Return ONLY a JSON array of strings. "
                        'Example: ["What year was the US founded?", "How many children did Britney Spears have?"]'
                    )},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
            ))

        results = asyncio.run(backend.run_batched(requests))

        for topic, result in zip(rag_topics, results):
            bank = NuggetBank(
                query_id=topic.request_id,
                title_query=topic.title or topic.request_id,
            )
            try:
                parsed = json.loads(result.text)
                if not isinstance(parsed, list):
                    print(f"WARNING: LLM returned non-list for topic {topic.request_id}: {type(parsed)}")
                    parsed = []
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"WARNING: Failed to parse LLM response for topic {topic.request_id}: {e}")
                print(f"  Raw response: {getattr(result, 'text', None)!r:.200}")
                parsed = []

            if not parsed:
                fallback_q = topic.problem_statement or topic.title or f"What is the answer to topic {topic.request_id}?"
                print(f"WARNING: Using fallback nugget for topic {topic.request_id}")
                parsed = [fallback_q]

            questions = []
            for item in parsed[:limit]:
                questions.append(NuggetQuestion.from_lazy(
                    query_id=topic.request_id,
                    question=item,
                    gold_answers=[],
                ))
            bank.add_nuggets(questions)
            banks.append(bank)

        return NuggetBanks.from_banks_list(banks)


# =============================================================================
# MinnaQrelsCreator
# =============================================================================

class MinnaQrelsCreator:
    CACHE_DIR = "output-rag4report"
    CACHE_TAG = "rag4report"
    CACHE_PATH = f"{CACHE_DIR}/qrels_grades_cache_{CACHE_TAG}.json"

    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        grade_range: Tuple[int, int] = (0, 3),
        **kwargs: Any,
    ) -> Optional[Qrels]:

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)

        cache = load_cache(self.CACHE_PATH)

        responses = list(rag_responses)
        grade_records: List[GradeRecord] = []
        requests_info: List[Tuple[str, str, str, MinimaLlmRequest]] = []

        for response in responses:
            topic_id = response.metadata.topic_id
            run_id = response.metadata.run_id
            text = response.get_report_text()

            if topic_id not in nugget_banks.banks:
                continue

            nuggets = nugget_banks.banks[topic_id].nuggets_as_list()

            for nugget in nuggets:
                cache_key = f"{run_id}_{topic_id}_{nugget.question_id}"
                if cache_key in cache:
                    continue
                requests_info.append((
                    run_id, topic_id, nugget.question_id,
                    MinimaLlmRequest(
                        request_id=cache_key,
                        messages=[
                            {"role": "system", "content": (
                                "How well does this response answer this question? "
                                "Reply with a single number from 0 to 5:\n"
                                "  5 = fully answers with specific evidence and details\n"
                                "  4 = fully answers in general terms\n"
                                "  3 = partially answers most of the question\n"
                                "  2 = partially answers some of the question\n"
                                "  1 = barely addresses the question\n"
                                "  0 = does not address the question"
                            )},
                            {"role": "user", "content": f"Question: {nugget.question}\n\nResponse: {text}"},
                        ],
                        temperature=0.0,
                    )
                ))

        if requests_info:
            print(f"MinnaQrelsCreator: Sending {len(requests_info)} LLM grading requests "
                  f"({len(cache)} already cached)...")
            BATCH_SIZE = 5000
            for batch_start in range(0, len(requests_info), BATCH_SIZE):
                batch = requests_info[batch_start:batch_start + BATCH_SIZE]
                try:
                    results = asyncio.run(backend.run_batched([req for _, _, _, req in batch]))
                    for (run_id, topic_id, nug_id, _), result in zip(batch, results):
                        cache_key = f"{run_id}_{topic_id}_{nug_id}"
                        cache[cache_key] = self._parse_graded(result)
                except Exception as e:
                    print(f"WARNING: Batch {batch_start}-{batch_start + len(batch)} failed: {e}")
                    print(f"  Saving {len(cache)} cached grades before re-raising...")
                    save_cache(cache, self.CACHE_PATH)
                    raise
                save_cache(cache, self.CACHE_PATH)
                print(f"  Qrels progress: {min(batch_start + BATCH_SIZE, len(requests_info))}/{len(requests_info)} "
                      f"submitted, {len(cache)} total cached")

        max_grade = grade_range[1]
        for response in responses:
            topic_id = response.metadata.topic_id
            run_id = response.metadata.run_id
            text = response.get_report_text()
            if topic_id not in nugget_banks.banks:
                continue
            nuggets = nugget_banks.banks[topic_id].nuggets_as_list()
            if not nuggets:
                continue
            tallies = [
                cache.get(f"{run_id}_{topic_id}_{n.question_id}", 0)
                for n in nuggets
            ]
            avg = sum(tallies) / (len(tallies) * 5.0)
            grade = round(max_grade * avg)
            grade_records.append(GradeRecord(topic_id, text, grade))

        qrels = build_qrels(records=grade_records, spec=MINIMAL_QRELS_SPEC)
        print(f"MinnaQrelsCreator: Created qrels for {len(grade_records)} responses")
        return qrels

    def _parse_graded(self, result) -> int:
        """Parse 0-5 graded response from LLM."""
        try:
            text = result.text.strip()
            for char in text:
                if char in ("0", "1", "2", "3", "4", "5"):
                    return int(char)
            return 0
        except:
            return 0


# =============================================================================
# Cache helpers 
# =============================================================================

def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_cache(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


class MinnaLeaderboardJudge:

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        on_missing_evals: str = "fix_aggregate",
        **kwargs: Any,
    ) -> Leaderboard:
        """Judge RAG responses and produce a leaderboard."""

        cache_dir = "output-rag4report"
        cache_tag = "rag4report"

        os.makedirs(cache_dir, exist_ok=True)
        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)
        responses = list(rag_responses)

        # FIX: inject document text into response.documents for adventure-style
        # runs (rag4reports-2026 format) where documents are not embedded in the
        # Report object.
       
        _inject_documents(responses)

        retrieval_cache_path = f"{cache_dir}/retrieval_quality_cache_{cache_tag}.json"
        retrieval_cache = load_cache(retrieval_cache_path)
        retrieval_requests: List[Tuple[str, str, str, MinimaLlmRequest]] = []
        retrieval_quality: Dict[Tuple[str, str], float] = {}

        if nugget_banks:
            for response in responses:
                topic_id = response.metadata.topic_id
                run_id = response.metadata.run_id
                if topic_id not in nugget_banks.banks:
                    continue
                if not response.documents:
                    continue

                nuggets = nugget_banks.banks[topic_id].nuggets_as_list()

                # Truncate each doc to 1000 chars & cap at 20 docs
                doc_texts = []
                for doc_id, doc in response.documents.items():
                    doc_texts.append(doc.text[:1000])
                combined_docs = "\n---\n".join(doc_texts[:20])

                for nugget in nuggets:
                    cache_key = f"{run_id}_{topic_id}_{nugget.question_id}"
                    if cache_key in retrieval_cache:
                        continue

                    retrieval_requests.append((
                        run_id, topic_id, nugget.question_id,
                        MinimaLlmRequest(
                            request_id=cache_key,
                            messages=[
                                {"role": "system", "content": (
                                    "Do these retrieved documents contain information "
                                    "that answers this question? Reply with a single number:\n"
                                    "  2 = documents fully answer the question\n"
                                    "  1 = documents partially answer the question\n"
                                    "  0 = documents do not answer the question"
                                )},
                                {"role": "user", "content": (
                                    f"Question: {nugget.question}\n\n"
                                    f"Retrieved documents:\n{combined_docs}"
                                )},
                            ],
                            temperature=0.0,
                        )
                    ))

            if retrieval_requests:
                print(f"RetrievalQuality: Sending {len(retrieval_requests)} LLM requests...")
                BATCH_SIZE = 5000
                for batch_start in range(0, len(retrieval_requests), BATCH_SIZE):
                    batch = retrieval_requests[batch_start:batch_start + BATCH_SIZE]
                    try:
                        ret_results = asyncio.run(backend.run_batched(
                            [req for _, _, _, req in batch]
                        ))
                        for (run_id, topic_id, nug_id, _), result in zip(batch, ret_results):
                            cache_key = f"{run_id}_{topic_id}_{nug_id}"
                            try:
                                text = result.text.strip()
                                score = 0
                                for char in text:
                                    if char in ("0", "1", "2"):
                                        score = int(char)
                                        break
                            except:
                                score = 0
                            retrieval_cache[cache_key] = score
                    except Exception as e:
                        print(f"WARNING: Retrieval batch {batch_start}-{batch_start + len(batch)} failed: {e}")
                        print(f"  Saving {len(retrieval_cache)} cached scores before re-raising...")
                        save_cache(retrieval_cache, retrieval_cache_path)
                        raise
                    save_cache(retrieval_cache, retrieval_cache_path)
                    print(f"  RetrievalQuality progress: {min(batch_start + BATCH_SIZE, len(retrieval_requests))}/{len(retrieval_requests)} "
                          f"submitted, {len(retrieval_cache)} total cached")

            # normalize to 0-1 (max per nugget is 2)
            for response in responses:
                topic_id = response.metadata.topic_id
                run_id = response.metadata.run_id
                if topic_id not in nugget_banks.banks:
                    continue
                nuggets = nugget_banks.banks[topic_id].nuggets_as_list()
                if not nuggets:
                    continue
                scores_list = []
                for nugget in nuggets:
                    cache_key = f"{run_id}_{topic_id}_{nugget.question_id}"
                    scores_list.append(retrieval_cache.get(cache_key, 0))
                retrieval_quality[(run_id, topic_id)] = sum(scores_list) / (len(scores_list) * 2.0)

        print(f"RetrievalQuality: Scored {len(retrieval_quality)} (run, topic) pairs")

    
        qrels_grades_cache_path = MinnaQrelsCreator.CACHE_PATH
        qrels_grades_cache = load_cache(qrels_grades_cache_path)
        response_nugget_score: Dict[Tuple[str, str], float] = {}

        if nugget_banks:
            for response in responses:
                topic_id = response.metadata.topic_id
                run_id = response.metadata.run_id
                if topic_id not in nugget_banks.banks:
                    continue
                nuggets = nugget_banks.banks[topic_id].nuggets_as_list()
                if not nuggets:
                    continue
                tallies = [
                    qrels_grades_cache.get(f"{run_id}_{topic_id}_{n.question_id}", 0)
                    for n in nuggets
                ]
                # 0-5 per nugget → normalize to [0, 1]
                response_nugget_score[(run_id, topic_id)] = sum(tallies) / (len(tallies) * 5.0)

        print(f"ResponseNugget: Scored {len(response_nugget_score)} (run, topic) pairs "
              f"from {len(qrels_grades_cache)} cached grades")

        # ──────────────── Claims extraction ────────────────
        claims_cache_path = f"{cache_dir}/claims_cache_{cache_tag}.json"
        claims_cache = load_cache(claims_cache_path)
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []

        for response in responses:
            topic_id = response.metadata.topic_id
            key = f"{response.metadata.run_id}_{topic_id}"
            if key in claims_cache:
                continue
            text = response.get_report_text()
            requests_info.append((
                response.metadata.run_id, topic_id,
                MinimaLlmRequest(
                    request_id=f"{response.metadata.run_id}_{topic_id}",
                    messages=[
                        {"role": "system", "content": (
                            "You are a claim extractor. Break this response into specific claims that "
                            "could be verified against a source document. Each claim should be a single, "
                            "self-contained statement. Include all types: facts, opinions, predictions, "
                            "and hedged statements. For hedged claims, preserve the hedging language. "
                            "Return ONLY a JSON array of strings. "
                            "Example: [\"The war started in 1914.\", "
                            "\"Everybody in the family agreed that it was a bad meal.\"] "
                        )},
                        {"role": "user", "content": f"Response: {text}"},
                    ],
                    temperature=0.3,
                )
            ))

        claims: Dict[Tuple[str, str], List[str]] = {}
        for key, value in claims_cache.items():
            r_id, t_id = key.split("_", 1)
            claims[(r_id, t_id)] = value

        if requests_info:
            print(f"Claims extraction: Sending {len(requests_info)} LLM requests "
                  f"({len(claims_cache)} already cached)...")
            try:
                results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))
                for (run_id, topic_id, _), result in zip(requests_info, results):
                    key = f"{run_id}_{topic_id}"
                    try:
                        parsed = json.loads(result.text)
                    except (json.JSONDecodeError, AttributeError):
                        parsed = []
                    parsed = [c for c in parsed if isinstance(c, str) and len(c.strip()) >= 10]
                    claims[(run_id, topic_id)] = parsed
                    claims_cache[key] = parsed
            except Exception as e:
                print(f"WARNING: Claims batch failed: {e}")
                print(f"  Saving {len(claims_cache)} cached claims before re-raising...")
                save_cache(claims_cache, claims_cache_path)
                raise
            save_cache(claims_cache, claims_cache_path)

        # ── Stage 2a: Attribution score (claim × all docs, continuous max-pool) ──
        nli_scores_cache_path = f"{cache_dir}/nli_scores_continuous_{cache_tag}.json"
        nli_scores_cache = load_cache(nli_scores_cache_path)
        score_dict: Dict[Tuple[Tuple[str, str], str], float] = {}

        pairs: List[Tuple[str, str]] = []
        pair_index: List[Tuple[Tuple[str, str], str, str]] = []

        for response in responses:
            if not response.documents:
                continue
            key = (response.metadata.run_id, response.metadata.topic_id)
            for claim in claims.get(key, []):
                cached_max = 0.0
                for doc_id, doc in response.documents.items():
                    key_str = f"{key[0]}_{key[1]}_{doc_id}_{claim}"
                    if key_str in nli_scores_cache:
                        cached_max = max(cached_max, float(nli_scores_cache[key_str]))
                    else:
                        pairs.append((doc.text, claim))
                        pair_index.append((key, doc_id, claim))
                score_dict[(key, claim)] = cached_max

        CHUNK_SIZE = 500
        if pairs:
            print(f"Attribution NLI: {len(pairs)} pairs to score on {nli_model._device}", flush=True)
            for i in range(0, len(pairs), CHUNK_SIZE):
                chunk_scores = nli_model.predict(pairs[i:i + CHUNK_SIZE])
                print(f"  Attribution NLI: {min(i + CHUNK_SIZE, len(pairs))}/{len(pairs)} done", flush=True)
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, doc_id, claim = pair_index[idx]
                    key_str = f"{key[0]}_{key[1]}_{doc_id}_{claim}"
                    float_score = float(score)
                    nli_scores_cache[key_str] = float_score
                    if float_score > score_dict.get((key, claim), 0.0):
                        score_dict[(key, claim)] = float_score
                if (i // CHUNK_SIZE) % 10 == 9:
                    save_cache(nli_scores_cache, nli_scores_cache_path)

        save_cache(nli_scores_cache, nli_scores_cache_path)

        # ── Stage 2b: Citation accuracy ────
        citation_cache_path = f"{cache_dir}/citation_deberta_cache_{cache_tag}.json"
        citation_cache = load_cache(citation_cache_path)

        cite_pairs: List[Tuple[str, str]] = []
        cite_pair_index: List[Tuple[Tuple[str, str], int]] = []
        citation_info: Dict[Tuple[str, str], Tuple[int, int]] = {}

        for response in responses:
            if not response.responses:
                continue
            key = (response.metadata.run_id, response.metadata.topic_id)
            docs = response.documents or {}

            supported = 0
            total_cited = 0

            for frag_idx, fragment in enumerate(response.responses):
                cited_ids = []
                if fragment.citations:
                    if isinstance(fragment.citations, dict):
                        cited_ids = list(fragment.citations.keys())
                    elif isinstance(fragment.citations, list):
                        cited_ids = [str(c) for c in fragment.citations]

                if not cited_ids:
                    continue

                total_cited += 1
                cache_key = f"{key[0]}_{key[1]}_{frag_idx}"

                if cache_key in citation_cache:
                    if citation_cache[cache_key] == 1:
                        supported += 1
                    continue

                frag_text = fragment.text
                for cid in cited_ids:
                    if cid in docs:
                        cite_pairs.append((docs[cid].text, frag_text))
                        cite_pair_index.append((key, frag_idx))
                        break

            citation_info[key] = (supported, total_cited)

        if cite_pairs:
            print(f"Citation NLI (deberta): {len(cite_pairs)} pairs", flush=True)
            for i in range(0, len(cite_pairs), CHUNK_SIZE):
                chunk_scores = citation_nli_model.predict(cite_pairs[i:i + CHUNK_SIZE])
                print(f"  Citation NLI (deberta): {min(i + CHUNK_SIZE, len(cite_pairs))}/{len(cite_pairs)} done", flush=True)
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, frag_idx = cite_pair_index[idx]
                    cache_key = f"{key[0]}_{key[1]}_{frag_idx}"
                    # entailment class must outscore both neutral and contradiction
                    is_supported = 1 if (score[1] > score[0] and score[1] > score[2]) else 0
                    citation_cache[cache_key] = is_supported
                    if is_supported:
                        prev_sup, prev_total = citation_info[key]
                        citation_info[key] = (prev_sup + 1, prev_total)

        save_cache(citation_cache, citation_cache_path)

        # ── Build leaderboard ──────────────────────────────
        builder: LeaderboardBuilder = LeaderboardBuilder(MINIMAL_SPEC)
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            topic_id = response.metadata.topic_id

            claim_list = claims.get(key, [])
            if claim_list:
                attribution = sum(score_dict.get((key, c), 0.0) for c in claim_list) / len(claim_list)
            else:
                attribution = 0.0

            cite_sup, cite_total = citation_info.get(key, (0, 0))
            cite_acc = cite_sup / cite_total if cite_total > 0 else 0.0

            ret_qual = retrieval_quality.get(key, 0.0)
            resp_nug = response_nugget_score.get(key, 0.0)

            final = 0.5 * (max(cite_acc, attribution) + ret_qual)

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=topic_id,
                values={
                    "ATTRIBUTION_SCORE": attribution,
                    "CITATION_ACCURACY": cite_acc,
                    "RETRIEVAL_QUALITY": ret_qual,
                    "RESPONSE_NUGGET_SCORE": resp_nug,
                    "FINAL": final,
                },
            )

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )
        print(f"MinnaLeaderboardJudge: Built leaderboard with {len(leaderboard.entries)} entries"
              f" (FINAL = 0.5*(max(cite,attr) + ret))")

        return leaderboard


# =============================================================================
# CLI Entry Point 
# =============================================================================

if __name__ == "__main__":
    from autojudge_base import AutoJudge, auto_judge_to_click_command

    class CompleteMinnaJudge(AutoJudge):
        nugget_banks_type = NuggetBanks

        def __init__(self):
            self._nugget_creator = MinnaNuggetCreator()
            self._qrels_creator = MinnaQrelsCreator()
            self._leaderboard_judge = MinnaLeaderboardJudge()

        def create_nuggets(self, *args, **kwargs):
            return self._nugget_creator.create_nuggets(*args, **kwargs)

        def create_qrels(self, *args, **kwargs):
            return self._qrels_creator.create_qrels(*args, **kwargs)

        def judge(self, *args, **kwargs):
            return self._leaderboard_judge.judge(*args, **kwargs)

    auto_judge_to_click_command(CompleteMinnaJudge(), "simple_example_judge")()
