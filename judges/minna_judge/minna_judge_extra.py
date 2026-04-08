#!/usr/bin/env python3
# minna_judge_extra.py — kitchen-sink judge that combines every signal and
# formula trick I'd actually try if I had to maximize kendall.
#
# DESIGN PHILOSOPHY (lessons from runs 4-11):
#   1. The trade-off between content kendall and citation kendall is real;
#      no single linear formula dominates. Solution: report MULTIPLE FINAL
#      variants and pick the winner empirically.
#   2. ATTRIBUTION (HHEM) and RETRIEVAL_QUALITY are both content signals;
#      they're nearly redundant. Treat them as one.
#   3. CITATION_ACCURACY (deberta strict 3-class) is orthogonal to content;
#      it's the only signal good at the citation truth cluster.
#   4. The biggest unmined signal is RESPONSE-level nugget grading (asking
#      the LLM "does THIS RESPONSE answer this nugget" instead of "do the
#      retrieved docs answer it"). Predicted to give the largest single
#      boost on correct_nuggets / nugget_coverage / f1 truth measures.
#   5. Cheap text-based discriminators (citation completeness, specificity)
#      can break ties in the middle of the score distribution where
#      kendall vs pearson gap usually opens up.
#   6. Rank-percentile normalization sidesteps the scale-mismatch bug that
#      makes max(cite, attr) collapse to attr in raw values.
#
# CACHING (everything reused from run7 wherever possible):
#   - HHEM attribution NLI:     output-kiddie/minna_judge_run7.nli_scores_newcache.json
#   - Deberta citation NLI:     output-kiddie/minna_judge_run7.citation_deberta_cache.json
#   - Retrieval quality LLM:    output-kiddie/minna_judge_run7.retrieval_quality_cache.json
#   - Claims extraction LLM:    output-kiddie/minna_judge_run7.claims_newcache.json
#   - Response-nugget LLM:      output-kiddie/minna_judge_run7.response_nugget_cache.json  (NEW)
#
# The new response-nugget cache will populate on first run (~3-5k LLM calls
# for the kiddie set, ~$0.50-$2 depending on backend). After that it's free.

import warnings
import logging
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens")
warnings.filterwarnings("ignore", message="You are using the default legacy behaviour")
logging.getLogger("transformers").setLevel(logging.ERROR)

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import asyncio
import json
import re
import dataclasses
import os
import math

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
from autojudge_base.nugget_data import NuggetBank, NuggetQuestion
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm
from transformers import AutoModelForSequenceClassification, T5Tokenizer
from sentence_transformers import CrossEncoder
import torch


# =============================================================================
# NLI models — HHEM for attribution, DeBERTa for citation
# =============================================================================

class _VectaraHHEM:
    """HHEMv2 wrapper with .predict() matching CrossEncoder interface."""

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
                return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            scores.extend(torch.sigmoid(logits[:, 0]).cpu().tolist())
        return scores


nli_model = _VectaraHHEM()
citation_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
print("citation_nli_model: loaded cross-encoder/nli-deberta-v3-base")


# =============================================================================
# Leaderboard spec — many signals, many FINAL variants for empirical picking
# =============================================================================

# All the FINAL variants are defined in one place so it's easy to compare:
#
#   FINAL_SCORE          → V3 from run11 (proven best on f1: kendall 0.678)
#                          0.5*(max(cite,attr)+ret) + 0.10*cite
#   FINAL_RESP_NUG       → V3 with response-level nugget grading mixed in
#                          0.4*resp_nug + 0.6*[V3]
#   FINAL_RANKED         → rank-percentile-normalized V3 (kills scale bias)
#   FINAL_KITCHEN_SINK   → weighted combo of every signal we have
#   FINAL_DUAL_MAX       → max(content_signal, citation_signal) on rank-norm scale
#   FINAL_HARMONIC       → 3-way harmonic mean of (resp_nug, attr, cite_norm)
#                          forces balance — penalizes responses bad at any one
MINIMAL_SPEC = LeaderboardSpec(measures=(
    # Component signals (raw)
    MeasureSpec("COMPLETENESS_SCORE"),
    MeasureSpec("ATTRIBUTION_SCORE"),
    MeasureSpec("CITATION_ACCURACY"),
    MeasureSpec("RETRIEVAL_QUALITY"),
    MeasureSpec("RESPONSE_NUGGET_SCORE"),
    MeasureSpec("CITATION_COMPLETENESS"),
    MeasureSpec("SPECIFICITY"),
    # FINAL variants
    MeasureSpec("FINAL_SCORE"),
    MeasureSpec("FINAL_RESP_NUG"),
    MeasureSpec("FINAL_RANKED"),
    MeasureSpec("FINAL_KITCHEN_SINK"),
    MeasureSpec("FINAL_DUAL_MAX"),
    MeasureSpec("FINAL_HARMONIC"),
))


class GradeRecord:
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
# Helpers — rank-percentile normalization, text-based signals
# =============================================================================

def rank_percentile(values_dict: Dict[Any, float]) -> Dict[Any, float]:
    """Convert {key: value} -> {key: percentile_rank in (0, 1]}.

    Robust to outliers (uses ranks, not min/max). Ties get average rank.
    """
    if not values_dict:
        return {}
    keys = list(values_dict.keys())
    vals = [values_dict[k] for k in keys]
    n = len(vals)
    # Average-rank for ties (matches scipy.stats.rankdata default)
    indexed = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[indexed[j + 1]] == vals[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-indexed average
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank / n
        i = j + 1
    return {keys[i]: ranks[i] for i in range(n)}


_FACT_TOKEN_RE = re.compile(r"\d|[%$€£¥]|\b[A-Z][a-zA-Z]{2,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def is_fact_bearing(text: str) -> bool:
    """Heuristic: contains numbers, currency, percentages, or proper nouns."""
    if not text or len(text.strip()) < 10:
        return False
    return bool(_FACT_TOKEN_RE.search(text))


def compute_citation_completeness(response: Report) -> float:
    """Fraction of fact-bearing sentences/fragments that have any citation.

    Captures the failure mode where a model writes correct claims but
    forgets (or refuses) to cite them. Distinct from CITATION_ACCURACY
    which only checks the *quality* of citations that exist.
    """
    if not response.responses:
        return 0.0
    fact_bearing = 0
    cited = 0
    for fragment in response.responses:
        if not is_fact_bearing(fragment.text):
            continue
        fact_bearing += 1
        if fragment.citations:
            cited += 1
    if fact_bearing == 0:
        return 0.0
    return cited / fact_bearing


def compute_specificity(response: Report) -> float:
    """Density of "specific" tokens (numbers, proper nouns) per word.

    Specific responses tend to be more useful — generic mush is a failure
    mode that other signals miss. Scaled to roughly [0, 1] by clipping.
    """
    text = response.get_report_text() if hasattr(response, "get_report_text") else ""
    if not text:
        return 0.0
    words = text.split()
    if len(words) < 5:
        return 0.0
    specific = 0
    for w in words:
        if _FACT_TOKEN_RE.search(w):
            specific += 1
    raw = specific / len(words)
    # Empirically, specific-token density of 0.20 = highly specific response
    return min(1.0, raw / 0.20)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


def harmonic3(a: float, b: float, c: float, eps: float = 1e-3) -> float:
    """Harmonic mean of three values, robust to zeros."""
    a = max(a, eps)
    b = max(b, eps)
    c = max(c, eps)
    return 3.0 / (1.0 / a + 1.0 / b + 1.0 / c)


# =============================================================================
# Cache helpers
# =============================================================================

def load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(data: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# =============================================================================
# MinnaNuggetCreator — same nugget pipeline as minna_judge.py
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
                    parsed = []
            except (json.JSONDecodeError, AttributeError):
                parsed = []
            if not parsed:
                fallback_q = topic.problem_statement or topic.title or f"What is the answer to topic {topic.request_id}?"
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
# MinnaQrelsCreator — same as minna_judge.py
# =============================================================================

class MinnaQrelsCreator:
    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        grade_range: Tuple[int, int] = (0, 3),
        length_threshold: int = 100,
        **kwargs: Any,
    ) -> Optional[Qrels]:
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)

        responses = list(rag_responses)
        grade_records: List[GradeRecord] = []
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []

        for response in responses:
            topic_id = response.metadata.topic_id
            text = response.get_report_text()
            if topic_id not in nugget_banks.banks:
                continue
            nuggets = nugget_banks.banks[topic_id].nuggets_as_list()
            for nugget in nuggets:
                requests_info.append((
                    response.metadata.run_id,
                    topic_id,
                    MinimaLlmRequest(
                        request_id=f"{response.metadata.run_id}_{topic_id}_{nugget.question_id}",
                        messages=[
                            {"role": "system", "content": (
                                "How well does this response answer this question? "
                                "Reply with a single number:\n"
                                "  2 = fully answers the question\n"
                                "  1 = partially answers the question\n"
                                "  0 = does not answer the question"
                            )},
                            {"role": "user", "content": f"Question: {nugget.question}\n\nResponse: {text}"},
                        ],
                        temperature=0.0,
                    )
                ))

        results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))
        scores: Dict[Tuple[str, str], List[int]] = {}
        for (run_id, topic_id, _), result in zip(requests_info, results):
            score = self._parse_graded(result)
            scores.setdefault((run_id, topic_id), []).append(score)

        max_grade = grade_range[1]
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            text = response.get_report_text()
            topic_id = response.metadata.topic_id
            if key not in scores:
                continue
            tallies = scores[key]
            avg = sum(tallies) / (len(tallies) * 2.0)
            grade = round(max_grade * avg)
            grade_records.append(GradeRecord(topic_id, text, grade))

        qrels = build_qrels(records=grade_records, spec=MINIMAL_QRELS_SPEC)
        print(f"MinnaQrelsCreator: Created qrels for {len(grade_records)} responses")
        return qrels

    def _parse_graded(self, result) -> int:
        try:
            text = result.text.strip()
            for char in text:
                if char in ("0", "1", "2"):
                    return int(char)
            return 0
        except Exception:
            return 0


# =============================================================================
# MinnaLeaderboardJudge — the actual scoring engine
# =============================================================================

class MinnaLeaderboardJudge:

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        keyword_bonus: float = 0.2,
        on_missing_evals: str = "fix_aggregate",
        **kwargs: Any,
    ) -> Leaderboard:
        # Reuse all caches from run7 regardless of current run filebase
        cache_filebase = "output-kiddie/minna_judge_run7"
        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)
        responses = list(rag_responses)

        qrels_dict: Dict[Tuple[str, str], int] = {}
        if qrels:
            for row in qrels.rows:
                qrels_dict[(row.topic_id, row.doc_id)] = row.grade

        # ── Stage A: RETRIEVAL_QUALITY (cached LLM grade of docs vs nugget) ──
        retrieval_cache_path = f"{cache_filebase}.retrieval_quality_cache.json"
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
                doc_texts = [doc.text[:1000] for _, doc in response.documents.items()]
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
                ret_results = asyncio.run(backend.run_batched([req for _, _, _, req in retrieval_requests]))
                for (run_id, topic_id, nug_id, _), result in zip(retrieval_requests, ret_results):
                    cache_key = f"{run_id}_{topic_id}_{nug_id}"
                    score = 0
                    try:
                        for char in result.text.strip():
                            if char in ("0", "1", "2"):
                                score = int(char)
                                break
                    except Exception:
                        score = 0
                    retrieval_cache[cache_key] = score
                save_cache(retrieval_cache, retrieval_cache_path)

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

        # ── Stage B: RESPONSE_NUGGET grading (NEW — biggest predicted impact) ─
        # Same prompt as retrieval quality but feeds the RESPONSE TEXT instead
        # of retrieved docs. Directly measures whether the model used the
        # information to answer each sub-question — much more aligned with
        # nugget_coverage / correct_nuggets truth measures than the existing
        # retrieval-based version.
        response_nug_cache_path = f"{cache_filebase}.response_nugget_cache.json"
        response_nug_cache = load_cache(response_nug_cache_path)
        resp_nug_requests: List[Tuple[str, str, str, MinimaLlmRequest]] = []
        response_nugget_score: Dict[Tuple[str, str], float] = {}

        if nugget_banks:
            for response in responses:
                topic_id = response.metadata.topic_id
                run_id = response.metadata.run_id
                if topic_id not in nugget_banks.banks:
                    continue
                response_text = response.get_report_text()
                if not response_text:
                    continue
                nuggets = nugget_banks.banks[topic_id].nuggets_as_list()
                # Truncate response to keep prompts compact
                resp_for_prompt = response_text[:6000]
                for nugget in nuggets:
                    cache_key = f"{run_id}_{topic_id}_{nugget.question_id}"
                    if cache_key in response_nug_cache:
                        continue
                    resp_nug_requests.append((
                        run_id, topic_id, nugget.question_id,
                        MinimaLlmRequest(
                            request_id=f"resp_nug_{cache_key}",
                            messages=[
                                {"role": "system", "content": (
                                    "Read the response carefully. Does it answer the question with "
                                    "specific, supported information? Reply with a single number:\n"
                                    "  3 = fully answers with specifics and evidence\n"
                                    "  2 = answers with general information\n"
                                    "  1 = mentions the topic but does not answer\n"
                                    "  0 = does not address the question"
                                )},
                                {"role": "user", "content": (
                                    f"Question: {nugget.question}\n\nResponse: {resp_for_prompt}"
                                )},
                            ],
                            temperature=0.0,
                        )
                    ))

            if resp_nug_requests:
                print(f"ResponseNugget: Sending {len(resp_nug_requests)} LLM requests "
                      f"(this populates a new cache; subsequent runs are free)...")
                rn_results = asyncio.run(backend.run_batched([req for _, _, _, req in resp_nug_requests]))
                for (run_id, topic_id, nug_id, _), result in zip(resp_nug_requests, rn_results):
                    cache_key = f"{run_id}_{topic_id}_{nug_id}"
                    score = 0
                    try:
                        for char in result.text.strip():
                            if char in ("0", "1", "2", "3"):
                                score = int(char)
                                break
                    except Exception:
                        score = 0
                    response_nug_cache[cache_key] = score
                save_cache(response_nug_cache, response_nug_cache_path)

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
                    scores_list.append(response_nug_cache.get(cache_key, 0))
                # Normalize by max grade (3) so result is in [0, 1]
                response_nugget_score[(run_id, topic_id)] = sum(scores_list) / (len(scores_list) * 3.0)

        print(f"ResponseNugget: Scored {len(response_nugget_score)} (run, topic) pairs")

        # ── Stage C: Claims extraction (cached) ─────────────────────────────
        claims_cache_path = f"{cache_filebase}.claims_newcache.json"
        claims_cache = load_cache(claims_cache_path)
        claims_requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []

        for response in responses:
            topic_id = response.metadata.topic_id
            key = f"{response.metadata.run_id}_{topic_id}"
            if key in claims_cache:
                continue
            text = response.get_report_text()
            claims_requests_info.append((
                response.metadata.run_id, topic_id,
                MinimaLlmRequest(
                    request_id=f"{response.metadata.run_id}_{topic_id}",
                    messages=[
                        {"role": "system", "content": (
                            "You are a claim extractor. Break this response into specific claims that "
                            "could be verified against a source document. Each claim should be a single, "
                            "self-contained statement. Include all types: facts, opinions, predictions, "
                            "and hedged statements. For hedged claims, preserve the hedging language. "
                            "Return ONLY a JSON array of strings."
                        )},
                        {"role": "user", "content": f"Response: {text}"},
                    ],
                    temperature=0.3,
                )
            ))

        cl_results = asyncio.run(backend.run_batched([req for _, _, req in claims_requests_info]))
        claims: Dict[Tuple[str, str], List[str]] = {}
        for key, value in claims_cache.items():
            r_id, t_id = key.split("_", 1)
            claims[(r_id, t_id)] = value
        for (run_id, topic_id, _), result in zip(claims_requests_info, cl_results):
            key = f"{run_id}_{topic_id}"
            try:
                parsed = json.loads(result.text)
            except (json.JSONDecodeError, AttributeError):
                parsed = []
            parsed = [c for c in parsed if isinstance(c, str) and len(c.strip()) >= 10]
            claims[(run_id, topic_id)] = parsed
            claims_cache[key] = parsed
        save_cache(claims_cache, claims_cache_path)

        # ── Stage D: ATTRIBUTION via HHEM (cached, claim × all docs) ────────
        nli_scores_cache_path = f"{cache_filebase}.nli_scores_newcache.json"
        nli_scores_cache = load_cache(nli_scores_cache_path)
        score_dict: Dict[Tuple[Tuple[str, str], str], int] = {}
        pairs: List[Tuple[str, str]] = []
        pair_index: List[Tuple[Tuple[str, str], str, str]] = []

        for response in responses:
            if not response.documents:
                continue
            key = (response.metadata.run_id, response.metadata.topic_id)
            for claim in claims.get(key, []):
                claim_supported = False
                uncached_docs: List[Tuple[str, Any]] = []
                for doc_id, doc in response.documents.items():
                    key_str = f"{key[0]}_{key[1]}_{doc_id}_{claim}"
                    if key_str in nli_scores_cache:
                        if nli_scores_cache[key_str] == 1:
                            claim_supported = True
                    else:
                        uncached_docs.append((doc_id, doc))
                score_dict[(key, claim)] = 1 if claim_supported else 0
                if not claim_supported:
                    for doc_id, doc in uncached_docs:
                        pairs.append((doc.text, claim))
                        pair_index.append((key, doc_id, claim))

        CHUNK_SIZE = 500
        if pairs:
            print(f"Attribution NLI: {len(pairs)} pairs to score on {nli_model._device}", flush=True)
            for i in range(0, len(pairs), CHUNK_SIZE):
                chunk_scores = nli_model.predict(pairs[i:i + CHUNK_SIZE])
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, doc_id, claim = pair_index[idx]
                    key_str = f"{key[0]}_{key[1]}_{doc_id}_{claim}"
                    raw_nli = 1 if float(score) > 0.5 else 0
                    nli_scores_cache[key_str] = raw_nli
                    if raw_nli == 1:
                        score_dict[(key, claim)] = 1
                if (i // CHUNK_SIZE) % 10 == 9:
                    save_cache(nli_scores_cache, nli_scores_cache_path)
        save_cache(nli_scores_cache, nli_scores_cache_path)

        # ── Stage E: CITATION_ACCURACY via deberta strict (cached) ──────────
        citation_cache_path = f"{cache_filebase}.citation_deberta_cache.json"
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
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, frag_idx = cite_pair_index[idx]
                    cache_key = f"{key[0]}_{key[1]}_{frag_idx}"
                    is_supported = 1 if (score[1] > score[0] and score[1] > score[2]) else 0
                    citation_cache[cache_key] = is_supported
                    if is_supported:
                        prev_sup, prev_total = citation_info[key]
                        citation_info[key] = (prev_sup + 1, prev_total)
        save_cache(citation_cache, citation_cache_path)

        # ── Stage F: text-based signals (citation completeness, specificity) ─
        # Pure heuristics, no LLM calls. Cheap discriminators that can break
        # ties in the middle of the score distribution.
        cite_completeness: Dict[Tuple[str, str], float] = {}
        specificity: Dict[Tuple[str, str], float] = {}
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            cite_completeness[key] = compute_citation_completeness(response)
            specificity[key] = compute_specificity(response)

        # ── Pass 1: collect per-response component values ───────────────────
        per_response: Dict[Tuple[str, str], Dict[str, float]] = {}
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)

            claim_list = claims.get(key, [])
            if claim_list:
                attribution = sum(score_dict.get((key, c), 0) for c in claim_list) / len(claim_list)
            else:
                attribution = 0.0

            cite_sup, cite_total = citation_info.get(key, (0, 0))
            cite_acc = safe_div(cite_sup, cite_total)
            ret_qual = retrieval_quality.get(key, 0.0)
            resp_nug = response_nugget_score.get(key, 0.0)
            cite_compl = cite_completeness.get(key, 0.0)
            spec = specificity.get(key, 0.0)

            per_response[key] = {
                "attribution": attribution,
                "cite_acc": cite_acc,
                "ret_qual": ret_qual,
                "resp_nug": resp_nug,
                "cite_compl": cite_compl,
                "specificity": spec,
            }

        # Compute rank-percentiles for the components used in normalized formulas.
        # Rank-pct kills the scale-mismatch bug: normally cite_acc ≈ 0.10 and
        # attribution ≈ 0.80, so max(cite, attr) always picks attr. After
        # rank-pct both live in (0, 1] with comparable distributions.
        attr_pct = rank_percentile({k: v["attribution"] for k, v in per_response.items()})
        cite_pct = rank_percentile({k: v["cite_acc"] for k, v in per_response.items()})
        ret_pct = rank_percentile({k: v["ret_qual"] for k, v in per_response.items()})
        resp_nug_pct = rank_percentile({k: v["resp_nug"] for k, v in per_response.items()})
        cite_compl_pct = rank_percentile({k: v["cite_compl"] for k, v in per_response.items()})

        # ── Pass 2: build leaderboard with all FINAL variants ───────────────
        builder: LeaderboardBuilder = LeaderboardBuilder(MINIMAL_SPEC)
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            topic_id = response.metadata.topic_id
            text = response.get_report_text()
            text_id = doc_id_md5(text)

            s = per_response[key]
            attribution = s["attribution"]
            cite_acc = s["cite_acc"]
            ret_qual = s["ret_qual"]
            resp_nug = s["resp_nug"]
            cite_compl = s["cite_compl"]
            spec = s["specificity"]

            completeness = qrels_dict.get((topic_id, text_id), 0) / 3.0

            attr_p = attr_pct.get(key, 0.0)
            cite_p = cite_pct.get(key, 0.0)
            ret_p = ret_pct.get(key, 0.0)
            resp_nug_p = resp_nug_pct.get(key, 0.0)
            cite_compl_p = cite_compl_pct.get(key, 0.0)

            # ── FINAL variants ──────────────────────────────────────────────

            # 1. V3 from run11 — proven best on f1 (kendall 0.678)
            final_score = 0.5 * (max(cite_acc, attribution) + ret_qual) + 0.10 * cite_acc
            final_score = min(1.0, final_score)

            # 2. V3 + response-level nugget grading (the new signal)
            #    The response_nugget signal directly measures completeness,
            #    so we let it carry significant weight.
            v3_part = 0.5 * (max(cite_acc, attribution) + ret_qual) + 0.10 * cite_acc
            final_resp_nug = 0.5 * resp_nug + 0.5 * v3_part
            final_resp_nug = min(1.0, final_resp_nug)

            # 3. V3 structure but using rank-percentiles (no scale bias)
            final_ranked = 0.5 * (max(cite_p, attr_p) + ret_p) + 0.10 * cite_p

            # 4. Kitchen sink — every signal weighted by predicted importance
            #    Weights chosen to keep content as the dominant signal while
            #    giving citation, completeness, and the new signals a real say.
            final_kitchen_sink = (
                0.30 * resp_nug          # NEW: response-level nugget grade (biggest bet)
                + 0.20 * attribution      # HHEM claim coverage
                + 0.20 * ret_qual         # retrieval-aware completeness
                + 0.15 * cite_acc         # citation precision
                + 0.08 * cite_compl       # citation coverage (NEW)
                + 0.04 * spec             # specificity (NEW)
                + 0.03 * completeness     # qrels-derived completeness
            )
            final_kitchen_sink = min(1.0, final_kitchen_sink)

            # 5. Dual-max on rank-normalized scale — Proposal B done right.
            #    Now that components are on the same scale, max() actually
            #    picks the dominant cluster fairly per response.
            content_p = 0.5 * resp_nug_p + 0.5 * (0.5 * attr_p + 0.5 * ret_p)
            citation_p = 0.5 * cite_p + 0.5 * cite_compl_p
            final_dual_max = max(content_p, citation_p)

            # 6. Harmonic mean — penalizes responses bad at any one dimension.
            #    Forces balance instead of rewarding "best feature wins."
            #    Uses rank-percentiles so all three are on the same scale.
            final_harmonic = harmonic3(resp_nug_p, attr_p, cite_p)

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=topic_id,
                values={
                    # Component signals
                    "COMPLETENESS_SCORE": completeness,
                    "ATTRIBUTION_SCORE": attribution,
                    "CITATION_ACCURACY": cite_acc,
                    "RETRIEVAL_QUALITY": ret_qual,
                    "RESPONSE_NUGGET_SCORE": resp_nug,
                    "CITATION_COMPLETENESS": cite_compl,
                    "SPECIFICITY": spec,
                    # FINAL variants
                    "FINAL_SCORE": final_score,
                    "FINAL_RESP_NUG": final_resp_nug,
                    "FINAL_RANKED": final_ranked,
                    "FINAL_KITCHEN_SINK": final_kitchen_sink,
                    "FINAL_DUAL_MAX": final_dual_max,
                    "FINAL_HARMONIC": final_harmonic,
                },
            )

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )
        print(f"MinnaLeaderboardJudgeExtra: Built leaderboard with {len(leaderboard.entries)} entries")
        print("  FINAL_SCORE          = run11 V3 (proven best on f1)")
        print("  FINAL_RESP_NUG       = V3 + 50% response-level nugget grade")
        print("  FINAL_RANKED         = V3 with rank-percentile normalization")
        print("  FINAL_KITCHEN_SINK   = weighted combo of all 7 signals")
        print("  FINAL_DUAL_MAX       = max(content_p, citation_p) on rank scale")
        print("  FINAL_HARMONIC       = harmonic mean of (resp_nug, attr, cite)")

        return leaderboard


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    from autojudge_base import AutoJudge, auto_judge_to_click_command

    class CompleteMinnaJudgeExtra(AutoJudge):
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

    auto_judge_to_click_command(CompleteMinnaJudgeExtra(), "minna_judge_extra")()
