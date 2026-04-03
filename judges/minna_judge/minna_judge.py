#!/usr/bin/env python3
# RUN7

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import asyncio
import json
import dataclasses
import os
import hashlib
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
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm
from transformers import AutoModelForSequenceClassification, T5Tokenizer
import torch
from .pairwise_judge import PairwisePreferenceJudge, pick_anchors


class _VectaraHHEM:
    """Wrapper around HHEMv2 with .predict() matching CrossEncoder interface."""

    def __init__(self):
        self._tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self._model.eval()

    def predict(self, sentence_pairs, batch_size: int = 32):
        scores = []
        for i in range(0, len(sentence_pairs), batch_size):
            batch = sentence_pairs[i:i + batch_size]
            inputs = self._tokenizer(
                [p[0] for p in batch], [p[1] for p in batch],
                return_tensors="pt", padding=True, truncation=True,
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
            scores.extend(torch.sigmoid(logits[:, 0]).cpu().tolist())
        return scores


nli_model = _VectaraHHEM()


MINIMAL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("COMPLETENESS_SCORE"),    
    MeasureSpec("ATTRIBUTION_SCORE"),    
    MeasureSpec("CITATION_ACCURACY"),     
    MeasureSpec("RETRIEVAL_QUALITY"),     
    # ── FINAL_SCORE variants  ──
    MeasureSpec("FINAL_SCORE"),           # v1: (comp+ret+cite)
    MeasureSpec("FINAL_SCORE_V2"),        # v2: (comp+attr+ret+cite)
    MeasureSpec("FINAL_SCORE_V3"),        # v3: (comp dominates)
    MeasureSpec("FINAL_SCORE_V4"),        # v4: (ret dominates)
    MeasureSpec("FINAL_SCORE_V5"),        # v5: equal weight all 4 components
    # ── Pairwise scores(each uses its own anchors) ──
    MeasureSpec("PAIRWISE_SCORE"),        # pairwise for v1
    MeasureSpec("PAIRWISE_SCORE_V2"),     # pairwise for v2
    MeasureSpec("PAIRWISE_SCORE_V3"),     # pairwise for v3
    MeasureSpec("PAIRWISE_SCORE_V4"),     # pairwise for v4
    MeasureSpec("PAIRWISE_SCORE_V5"),     # pairwise for v5
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

        # fix2-6: collect diverse response samples to inform nugget generation
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

            # include response samples in nugget prompt so nuggets
            # better discriminate between good and bad responses
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
                        # ask LLM to grade 0/1/2 instead of binary 0/1
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
        scores = {}

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
            # each nugget scored 0-2, 1/2 to normalize to 1
            avg = sum(tallies) / (len(tallies) * 2.0)  
            grade = round(max_grade * avg)
            grade_records.append(GradeRecord(topic_id, text, grade))

        qrels = build_qrels(records=grade_records, spec=MINIMAL_QRELS_SPEC)
        print(f"MinnaQrelsCreator: Created qrels for {len(grade_records)} responses")
        return qrels

    def _parse_graded(self, result) -> int:
        """Parse 0/1/2 graded response from LLM."""
        try:
            text = result.text.strip()
            for char in text:
                if char in ("0", "1", "2"):
                    return int(char)
            return 0
        except:
            return 0

    def _parse_binary(self, result) -> int:
        try:
            text = result.text.strip().lower()
            if text.startswith("1") or text.startswith("yes"):
                return 1
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
        qrels: Optional[Qrels] = None,
        keyword_bonus: float = 0.2,
        on_missing_evals: str = "fix_aggregate",
        **kwargs: Any,
    ) -> Leaderboard:
        """Judge RAG responses and produce a leaderboard."""
        filebase = kwargs.get("filebase", "output-kiddie/minna_judge_run7")
        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)
        responses = list(rag_responses)

        qrels_dict: Dict[Tuple[str, str], int] = {}
        if qrels:
            for row in qrels.rows:
                qrels_dict[(row.topic_id, row.doc_id)] = row.grade

        
        retrieval_cache_path = f"{filebase}.retrieval_quality_cache.json"
        retrieval_cache = load_cache(retrieval_cache_path)
        # Each request checks one (system, topic, nugget) triple
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

                # Build a single "retrieval context" string from this system's
                # docs for this topic. We truncate each doc to 1000 chars and
                # cap at 20 docs to stay within LLM context limits.
                doc_texts = []
                for doc_id, doc in response.documents.items():
                    doc_texts.append(doc.text[:1000])
                combined_docs = "\n---\n".join(doc_texts[:20])

                for nugget in nuggets:
                    cache_key = f"{run_id}_{topic_id}_{nugget.question_id}"
                    if cache_key in retrieval_cache:
                        continue

                    # Ask the LLM: do these docs answer this sub-question?
                    # Graded 0/1/2 for consistency with the qrels grading scheme
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
                ret_results = asyncio.run(backend.run_batched(
                    [req for _, _, _, req in retrieval_requests]
                ))
                for (run_id, topic_id, nug_id, _), result in zip(retrieval_requests, ret_results):
                    cache_key = f"{run_id}_{topic_id}_{nug_id}"
                    # Parse graded 0/1/2 response (same logic as qrels grading)
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
                save_cache(retrieval_cache, retrieval_cache_path)

            # Aggregate: for each (run, topic), average the per-nugget scores
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

        # ── Stage 1: Claims extraction ────────────────
        claims_cache_path = f"{filebase}.claims_newcache.json"
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

        results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))

        claims: Dict[Tuple[str, str], List[str]] = {}
        for key, value in claims_cache.items():
            r_id, t_id = key.split("_", 1)
            claims[(r_id, t_id)] = value

        for (run_id, topic_id, _), result in zip(requests_info, results):
            key = f"{run_id}_{topic_id}"
            try:
                parsed = json.loads(result.text)
            except (json.JSONDecodeError, AttributeError):
                parsed = []
            parsed = [c for c in parsed if isinstance(c, str) and len(c.strip()) >= 10]
            claims[(run_id, topic_id)] = parsed
            claims_cache[key] = parsed

        save_cache(claims_cache, claims_cache_path)

        # ── Stage 2a: Attribution score (claim × all docs) ───
        nli_scores_cache_path = f"{filebase}.nli_scores_newcache.json"
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
            for i in range(0, len(pairs), CHUNK_SIZE):
                chunk_scores = nli_model.predict(pairs[i:i + CHUNK_SIZE])
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, doc_id, claim = pair_index[idx]
                    key_str = f"{key[0]}_{key[1]}_{doc_id}_{claim}"
                    # HHEM outputs single float (0-1), threshold at 0.5
                    raw_nli = 1 if float(score) > 0.5 else 0
                    nli_scores_cache[key_str] = raw_nli
                    if raw_nli == 1:
                        score_dict[(key, claim)] = 1

        save_cache(nli_scores_cache, nli_scores_cache_path)

        # ── Stage 2b: Citation accuracy (fragment × cited docs) ──────────
        citation_cache_path = f"{filebase}.citation_newcache.json"
        citation_cache = load_cache(citation_cache_path)

        cite_pairs: List[Tuple[str, str]] = []
        cite_pair_index: List[Tuple[Tuple[str, str], int, str]] = []

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

                # Check ALL cited docs (not just first)
                frag_text = fragment.text
                for cid in cited_ids:
                    if cid in docs:
                        cite_pairs.append((docs[cid].text, frag_text))
                        cite_pair_index.append((key, frag_idx, cid))

            citation_info[key] = (supported, total_cited)

        # Run NLI on citation pairs — a fragment is supported if ANY cited doc entails it
        if cite_pairs:
            already_supported = set()
            for i in range(0, len(cite_pairs), CHUNK_SIZE):
                chunk_scores = nli_model.predict(cite_pairs[i:i + CHUNK_SIZE])
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, frag_idx, cid = cite_pair_index[idx]
                    cache_key = f"{key[0]}_{key[1]}_{frag_idx}"

                    if cache_key in already_supported:
                        continue

                    # HHEM: single float (0-1), threshold at 0.5
                    is_supported = 1 if float(score) > 0.5 else 0
                    if is_supported:
                        already_supported.add(cache_key)
                        citation_cache[cache_key] = 1
                        prev_sup, prev_total = citation_info[key]
                        citation_info[key] = (prev_sup + 1, prev_total)

            # Mark unsupported fragments (checked all docs, none supported)
            for key, frag_idx, cid in cite_pair_index:
                cache_key = f"{key[0]}_{key[1]}_{frag_idx}"
                if cache_key not in citation_cache:
                    citation_cache[cache_key] = 0

        save_cache(citation_cache, citation_cache_path)

        
        # Step 1: Compute all 5 preliminary score variants per (run, topic)
        preliminary: Dict[str, Dict[Tuple[str, str], float]] = {
            "v1": {}, "v2": {}, "v3": {}, "v4": {}, "v5": {},
        }
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            topic_id = response.metadata.topic_id
            text = response.get_report_text()
            text_id = doc_id_md5(text)
            comp = qrels_dict.get((topic_id, text_id), 0) / 3.0
            cite_sup, cite_total = citation_info.get(key, (0, 0))
            ca = cite_sup / cite_total if cite_total > 0 else 0.0
            rq = retrieval_quality.get(key, 0.0)

            # Need claims/attribution for v2-v5
            claim_list = claims.get(key, [])
            attr = sum(score_dict.get((key, c), 0) for c in claim_list) / len(claim_list) if claim_list else 0

            preliminary["v1"][key] = 0.40 * comp + 0.35 * rq + 0.25 * ca
            preliminary["v2"][key] = 0.30 * comp + 0.25 * rq + 0.25 * attr + 0.20 * ca
            preliminary["v3"][key] = 0.55 * comp + 0.20 * rq + 0.15 * attr + 0.10 * ca
            preliminary["v4"][key] = 0.25 * comp + 0.50 * rq + 0.15 * attr + 0.10 * ca
            preliminary["v5"][key] = 0.25 * comp + 0.25 * rq + 0.25 * attr + 0.25 * ca

        # Step 2: Pick anchors and run pairwise for each variant.
        pairwise = PairwisePreferenceJudge()
        all_pairwise: Dict[str, Dict[Tuple[str, str], float]] = {}

        for variant_name, variant_scores in preliminary.items():
            anchors = pick_anchors(variant_scores)
            pw_scores = pairwise.run_pairwise(
                rag_responses=responses,
                rag_topics=rag_topics,
                llm_config=llm_config,
                anchors=anchors,
                filebase=filebase,
            )
            all_pairwise[variant_name] = pw_scores
            print(f"Pairwise {variant_name}: {len(pw_scores)} scored, {len(anchors)} anchors")

        # Build final leaderboard
        builder: LeaderboardBuilder = LeaderboardBuilder(MINIMAL_SPEC)
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            topic_id = response.metadata.topic_id
            text = response.get_report_text()
            text_id = doc_id_md5(text)

            claim_list = claims.get(key, [])
            if claim_list:
                attribution = sum(score_dict.get((key, c), 0) for c in claim_list) / len(claim_list)
            else:
                attribution = 0

            completeness = qrels_dict.get((topic_id, text_id), 0) / 3.0
            pw_v1 = all_pairwise["v1"].get(key, 1.0)
            pw_v2 = all_pairwise["v2"].get(key, 1.0)
            pw_v3 = all_pairwise["v3"].get(key, 1.0)
            pw_v4 = all_pairwise["v4"].get(key, 1.0)
            pw_v5 = all_pairwise["v5"].get(key, 1.0)

            cite_sup, cite_total = citation_info.get(key, (0, 0))
            cite_acc = cite_sup / cite_total if cite_total > 0 else 0.0

            ret_qual = retrieval_quality.get(key, 0.0)

           
            final_v1 = 0.40 * completeness + 0.35 * ret_qual + 0.25 * cite_acc
            final_v2 = 0.30 * completeness + 0.25 * ret_qual + 0.25 * attribution + 0.20 * cite_acc
            final_v3 = 0.55 * completeness + 0.20 * ret_qual + 0.15 * attribution + 0.10 * cite_acc
            final_v4 = 0.25 * completeness + 0.50 * ret_qual + 0.15 * attribution + 0.10 * cite_acc
            final_v5 = 0.25 * completeness + 0.25 * ret_qual + 0.25 * attribution + 0.25 * cite_acc

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=topic_id,
                values={
                    # ── Component measures ──
                    "COMPLETENESS_SCORE": completeness,
                    "ATTRIBUTION_SCORE": attribution,
                    "CITATION_ACCURACY": cite_acc,
                    "RETRIEVAL_QUALITY": ret_qual,
                    "PAIRWISE_SCORE": pw_v1,
                    "PAIRWISE_SCORE_V2": pw_v2,
                    "PAIRWISE_SCORE_V3": pw_v3,
                    "PAIRWISE_SCORE_V4": pw_v4,
                    "PAIRWISE_SCORE_V5": pw_v5,
                    # ── FINAL_SCORE variants ──
                    "FINAL_SCORE": final_v1,
                    "FINAL_SCORE_V2": final_v2,
                    "FINAL_SCORE_V3": final_v3,
                    "FINAL_SCORE_V4": final_v4,
                    "FINAL_SCORE_V5": final_v5,
                },
            )

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )
        print(f"MinnaLeaderboardJudge: Built leaderboard with {len(leaderboard.entries)} entries (includes PAIRWISE_SCORE)")

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
