#!/usr/bin/env python3
# SUBMISSION 2

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

# ms-marco cross-encoder for discriminative nugget scoring
disc_qa_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("disc_qa_model: loaded cross-encoder/ms-marco-MiniLM-L-6-v2")


# FINAL = 0.5*(max(cite, attr) + ret)
MINIMAL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("ATTRIBUTION_SCORE"),
    MeasureSpec("CITATION_ACCURACY"),
    MeasureSpec("RETRIEVAL_QUALITY"),
    MeasureSpec("RESPONSE_NUGGET_SCORE"),
    MeasureSpec("DISCRIMINATIVE_SCORE"),
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

        # fix2-6: collect diverse response samples to inform nugget generation -- not very effective
        responses = list(rag_responses)
        response_samples: Dict[str, List[str]] = {}
        for resp in responses:
            tid = resp.metadata.topic_id
            text = resp.get_report_text()
            if tid not in response_samples:
                response_samples[tid] = []
            if len(response_samples[tid]) < 3 and len(text) > 50:
                response_samples[tid].append(text[:500])  # first 500 chars

        banks: List[NuggetBank] = []
        requests = []

        for topic in rag_topics:
            if topic.title:
                context = f"Title: {topic.title}\n\nQuestion: {topic.problem_statement}"
            else:
                context = topic.problem_statement

            # include response samples in nugget prompt -- not very effective
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

            # if no nuggets were generated, just grab the question
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
    CACHE_DIR = "output-kiddie"
    CACHE_TAG = "mission3"
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

        # Build LLM requests only for cache misses.
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
            # Process in chunks so progress is saved if crashes.
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
            # each nugget scored 0-5; normalize to [0,1]
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


# =============================================================================
# MinnaLeaderboardJudge
# =============================================================================

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

        # RESPONSE_NUGGET_SCORE. Shares qrels-stage cache 
      
        cache_dir = "output-kiddie"
        cache_tag = "mission3"
        os.makedirs(cache_dir, exist_ok=True)
        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)
        responses = list(rag_responses)

        retrieval_cache_path = f"{cache_dir}/retrieval_quality_cache_{cache_tag}.json"
        retrieval_cache = load_cache(retrieval_cache_path)
        # Each request checks (system, topic, nugget) 
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

                # Build a single "retrieval context",
                # truncate each doc to 1000 chars & cap at 20 docs 
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

            # and normalize to 0-1 (max per nugget is 2)
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
        # get max (claim, doc)
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
                # must check ALL docs to find max.
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
                # save cache every 10 chunks to avoid losing progress
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

        # Run deberta NLI 
        if cite_pairs:
            print(f"Citation NLI (deberta): {len(cite_pairs)} pairs", flush=True)
            for i in range(0, len(cite_pairs), CHUNK_SIZE):
                chunk_scores = citation_nli_model.predict(cite_pairs[i:i + CHUNK_SIZE])
                print(f"  Citation NLI (deberta): {min(i + CHUNK_SIZE, len(cite_pairs))}/{len(cite_pairs)} done", flush=True)
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, frag_idx = cite_pair_index[idx]
                    cache_key = f"{key[0]}_{key[1]}_{frag_idx}"
                    # entailment must be largest
                    is_supported = 1 if (score[1] > score[0] and score[1] > score[2]) else 0
                    citation_cache[cache_key] = is_supported
                    if is_supported:
                        prev_sup, prev_total = citation_info[key]
                        citation_info[key] = (prev_sup + 1, prev_total)

        save_cache(citation_cache, citation_cache_path)

        # ── Stage D1: Extract discriminative nuggets (LLM, 122 calls) ──────
        # Per topic: pick best/worst by retrieval quality, ask LLM what
        # specific facts are in the good docs but missing from the bad.
        disc_nuggets_cache_path = f"{cache_dir}/disc_nuggets_cache_{cache_tag}.json"
        disc_nuggets_cache = load_cache(disc_nuggets_cache_path)
        disc_nuggets: Dict[str, List[str]] = {}  # topic_id → list of nugget strings

        # Build lookup: topic_id → list of (run_id, ret_score, response)
        topic_responses: Dict[str, List[Tuple[str, float, Report]]] = {}
        for response in responses:
            topic_id = response.metadata.topic_id
            run_id = response.metadata.run_id
            ret_score = retrieval_quality.get((run_id, topic_id), 0.0)
            topic_responses.setdefault(topic_id, []).append((run_id, ret_score, response))

        disc_llm_requests: List[Tuple[str, MinimaLlmRequest]] = []

        for topic_id, resp_list in topic_responses.items():
            # Check cache first
            if topic_id in disc_nuggets_cache:
                disc_nuggets[topic_id] = disc_nuggets_cache[topic_id]
                continue

            # Sort by retrieval quality
            resp_list.sort(key=lambda x: x[1])
            _, worst_score, worst_resp = resp_list[0]
            _, best_score, best_resp = resp_list[-1]

            # Skip if gap too small — both retrieved similar quality docs
            if best_score - worst_score < 0.1:
                disc_nuggets[topic_id] = []
                disc_nuggets_cache[topic_id] = []
                continue

            # Build doc text for best and worst
            def _docs_text(resp: Report) -> str:
                if not resp.documents:
                    return "(no documents)"
                texts = [doc.text[:1000] for _, doc in resp.documents.items()]
                return "\n---\n".join(texts[:20])

            best_docs_text = _docs_text(best_resp)
            worst_docs_text = _docs_text(worst_resp)

            # Find topic question
            topic_question = ""
            for t in rag_topics:
                if t.request_id == topic_id:
                    topic_question = t.problem_statement or t.title or topic_id
                    break

            disc_llm_requests.append((
                topic_id,
                MinimaLlmRequest(
                    request_id=f"disc_{topic_id}",
                    messages=[
                        {"role": "system", "content": (
                            "You are an evaluation expert. Compare two sets of retrieved documents "
                            "for the same question. Identify specific facts, details, statistics, "
                            "or pieces of evidence that are present in the GOOD documents but "
                            "missing or poorly covered in the BAD documents. "
                            "Focus on concrete, verifiable information that would make an answer "
                            "more complete and accurate. "
                            "Return ONLY a JSON array of exactly 5 strings."
                        )},
                        {"role": "user", "content": (
                            f"Question: {topic_question}\n\n"
                            f"GOOD retrieval documents:\n{best_docs_text}\n\n"
                            f"BAD retrieval documents:\n{worst_docs_text}"
                        )},
                    ],
                    temperature=0.3,
                )
            ))

        if disc_llm_requests:
            print(f"DiscriminativeNuggets: Sending {len(disc_llm_requests)} LLM requests "
                  f"({len(disc_nuggets_cache)} topics already cached)...")
            try:
                disc_results = asyncio.run(backend.run_batched(
                    [req for _, req in disc_llm_requests]
                ))
                for (topic_id, _), result in zip(disc_llm_requests, disc_results):
                    try:
                        parsed = json.loads(result.text)
                        if not isinstance(parsed, list):
                            parsed = []
                    except (json.JSONDecodeError, AttributeError):
                        parsed = []
                    parsed = [n for n in parsed if isinstance(n, str) and len(n.strip()) >= 10][:5]
                    disc_nuggets[topic_id] = parsed
                    disc_nuggets_cache[topic_id] = parsed
            except Exception as e:
                print(f"WARNING: Discriminative nuggets batch failed: {e}")
                save_cache(disc_nuggets_cache, disc_nuggets_cache_path)
                raise
            save_cache(disc_nuggets_cache, disc_nuggets_cache_path)

        total_disc_nuggets = sum(len(v) for v in disc_nuggets.values())
        print(f"DiscriminativeNuggets: {total_disc_nuggets} nuggets across "
              f"{sum(1 for v in disc_nuggets.values() if v)} topics")

        # ── Stage D2: Score claims against disc. nuggets (GPU, no LLM) ─────
        # For each (response, topic): build (nugget, claim) pairs, score with
        # ms-marco cross-encoder, take mean of max-per-nugget.
        disc_scores_cache_path = f"{cache_dir}/disc_scores_cache_{cache_tag}.json"
        disc_scores_cache = load_cache(disc_scores_cache_path)
        disc_score: Dict[Tuple[str, str], float] = {}

        # Collect all pairs to score in one big batch for GPU efficiency
        disc_pairs: List[Tuple[str, str]] = []
        disc_pair_index: List[Tuple[str, str, int, int]] = []  # run_id, topic_id, nugget_idx, claim_idx

        for response in responses:
            topic_id = response.metadata.topic_id
            run_id = response.metadata.run_id
            cache_key = f"{run_id}_{topic_id}"

            # Check cache
            if cache_key in disc_scores_cache:
                disc_score[(run_id, topic_id)] = float(disc_scores_cache[cache_key])
                continue

            topic_nugs = disc_nuggets.get(topic_id, [])
            response_claims = claims.get((run_id, topic_id), [])

            if not topic_nugs or not response_claims:
                disc_score[(run_id, topic_id)] = 0.0
                disc_scores_cache[cache_key] = 0.0
                continue

            for nug_idx, nugget in enumerate(topic_nugs):
                for claim_idx, claim in enumerate(response_claims):
                    disc_pairs.append((nugget, claim))
                    disc_pair_index.append((run_id, topic_id, nug_idx, claim_idx))

        if disc_pairs:
            print(f"DiscriminativeScore: {len(disc_pairs)} (nugget, claim) pairs "
                  f"to score on GPU...", flush=True)

            # Score all pairs with cross-encoder
            DISC_CHUNK = 512
            all_disc_scores: List[float] = []
            for i in range(0, len(disc_pairs), DISC_CHUNK):
                chunk = disc_pairs[i:i + DISC_CHUNK]
                chunk_scores = disc_qa_model.predict(chunk)
                all_disc_scores.extend(float(s) for s in chunk_scores)
                if (i // DISC_CHUNK) % 50 == 49:
                    print(f"  DiscriminativeScore: {min(i + DISC_CHUNK, len(disc_pairs))}/{len(disc_pairs)} done",
                          flush=True)

            # Aggregate: per (run, topic), build nugget × claim matrix,
            # take max per nugget row, then mean over nuggets.
            # First, collect scores by (run_id, topic_id)
            response_matrices: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
            for idx, (run_id, topic_id, nug_idx, claim_idx) in enumerate(disc_pair_index):
                key = (run_id, topic_id)
                if key not in response_matrices:
                    response_matrices[key] = {}
                if nug_idx not in response_matrices[key]:
                    response_matrices[key][nug_idx] = []
                response_matrices[key][nug_idx].append(all_disc_scores[idx])

            for key, nug_scores in response_matrices.items():
                # per nugget: max score across all claims
                per_nugget_best = [max(scores) for scores in nug_scores.values()]
                final_disc = sum(per_nugget_best) / len(per_nugget_best)
                disc_score[key] = final_disc
                disc_scores_cache[f"{key[0]}_{key[1]}"] = final_disc

            save_cache(disc_scores_cache, disc_scores_cache_path)

        print(f"DiscriminativeScore: Scored {len(disc_score)} (run, topic) pairs")

        # ── Build leaderboard───────────────────────────
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
            disc = disc_score.get(key, 0.0)

            final = 0.5 * (max(cite_acc, attribution) + ret_qual)

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=topic_id,
                values={
                    "ATTRIBUTION_SCORE": attribution,
                    "CITATION_ACCURACY": cite_acc,
                    "RETRIEVAL_QUALITY": ret_qual,
                    "RESPONSE_NUGGET_SCORE": resp_nug,
                    "DISCRIMINATIVE_SCORE": disc,
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
