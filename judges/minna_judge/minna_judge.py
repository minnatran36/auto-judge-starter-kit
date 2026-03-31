#!/usr/bin/env python3
# VERSION 4

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import asyncio
import json
import dataclasses
import os
import hashlib
# fix2-4: import numpy for cosine similarity in claim deduplication
import numpy as np
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
from sentence_transformers import CrossEncoder, SentenceTransformer
from .pairwise_judge import PairwisePreferenceJudge, pick_anchors

# Reverted to nli-deberta: vectara/hallucination_evaluation_model has persistent
# compatibility issues with current transformers/sentence-transformers versions.
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# fix2-4: sentence embedding model for claim deduplication
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


MINIMAL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("COMPLETENESS_SCORE"),
    MeasureSpec("ATTRIBUTION_SCORE"),
    MeasureSpec("CITATION_ACCURACY"),
    MeasureSpec("FINAL_SCORE"),
    MeasureSpec("PAIRWISE_SCORE"),
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


# fix2-4: deduplicate claims by cosine similarity
def _deduplicate_claims(claims: List[str], threshold: float = 0.85) -> List[str]:
    """Remove near-duplicate claims using cosine similarity.
    If two claims are > threshold similar, keep only the first one."""
    if len(claims) <= 1:
        return claims
    embeddings = _embed_model.encode(claims, convert_to_numpy=True)
    # normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    keep = []
    for i, claim in enumerate(claims):
        is_dup = False
        for j in keep:
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return [claims[i] for i in keep]


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

            # fix2-6: include response samples in nugget prompt so nuggets
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

            # Fallback: if no nuggets were generated, just grab the question
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
                        # fix2-2: ask LLM to grade 0/1/2 instead of binary 0/1
                        # for finer-grained completeness assessment
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
            # fix2-2: parse 0/1/2 graded response instead of binary
            score = self._parse_graded(result)
            scores.setdefault((run_id, topic_id), []).append(score)

        # fix2-2: use grade_range (0, 5) for finer granularity
        # with 0-2 per nugget and up to 10 nuggets, we get much smoother grades
        max_grade = grade_range[1]
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            text = response.get_report_text()
            topic_id = response.metadata.topic_id
            if key not in scores:
                continue

            tallies = scores[key]
            # fix2-2: average of 0-2 scores, scaled to grade range
            # e.g. avg=1.5 out of 2.0 → grade = round(3 * 0.75) = 2
            avg = sum(tallies) / (len(tallies) * 2.0)  # normalize to 0-1
            grade = round(max_grade * avg)
            grade_records.append(GradeRecord(topic_id, text, grade))

        qrels = build_qrels(records=grade_records, spec=MINIMAL_QRELS_SPEC)
        print(f"MinnaQrelsCreator: Created qrels for {len(grade_records)} responses")
        return qrels

    # fix2-2: new graded parser replacing old binary parser
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
        filebase = kwargs.get("filebase", "output-kiddie/minna_judge")
        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)

        backend = OpenAIMinimaLlm(full_config)
        responses = list(rag_responses)
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []

        # ── Check cache
        claims_cache_path = f"{filebase}.claims_cache.json"
        claims_cache = load_cache(claims_cache_path)

        for response in responses:
            topic_id = response.metadata.topic_id
            key = f"{response.metadata.run_id}_{topic_id}"

            if key in claims_cache:
                continue

            text = response.get_report_text()
            requests_info.append((
                response.metadata.run_id,
                topic_id,
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

        # Load cached claims into the working dict
        claims: Dict[Tuple[str, str], List[str]] = {}
        for key, value in claims_cache.items():
            r_id, t_id = key.split("_", 1)
            claims[(r_id, t_id)] = value

        qrels_dict: Dict[Tuple[str, str], int] = {}
        if qrels:
            for row in qrels.rows:
                qrels_dict[(row.topic_id, row.doc_id)] = row.grade

        for (run_id, topic_id, _), result in zip(requests_info, results):
            key = f"{run_id}_{topic_id}"
            try:
                parsed = json.loads(result.text)
            except (json.JSONDecodeError, AttributeError):
                parsed = []
            # filter out very short claims (< 10 chars)
            parsed = [c for c in parsed if isinstance(c, str) and len(c.strip()) >= 10]
            # fix2-4: deduplicate near-identical claims using cosine similarity
            parsed = _deduplicate_claims(parsed)
            claims[(run_id, topic_id)] = parsed
            claims_cache[key] = parsed

        save_cache(claims_cache, claims_cache_path)

        pairs: List[Tuple[str, str]] = []
        pair_index: List[Tuple[Tuple[str, str], str, str]] = []
        # (run_id, topic_id), doc_id, claim

        nli_scores_cache_path = f"{filebase}.nli_scores_cache.json"
        nli_scores_cache = load_cache(nli_scores_cache_path)
        score_dict: Dict[Tuple[Tuple[str, str], str], int] = {}
        # Dict[Tuple[Tuple[run_id, topic_id], claim], 0/1]

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
        all_nli_scores = []
        if pairs:
            for i in range(0, len(pairs), CHUNK_SIZE):
                chunk_scores = nli_model.predict(pairs[i:i + CHUNK_SIZE])
                all_nli_scores.extend(chunk_scores)

            for (key, doc_id, claim), score in zip(pair_index, all_nli_scores):
                key_str = f"{key[0]}_{key[1]}_{doc_id}_{claim}"
                # nli-deberta outputs [contradiction, entailment, neutral] logits
                raw_nli = 1 if score[1] > 0.0 else 0
                nli_scores_cache[key_str] = raw_nli
                if raw_nli == 1:
                    score_dict[(key, claim)] = 1

        save_cache(nli_scores_cache, nli_scores_cache_path)

        # ── Stage 2b: Citation accuracy ──────────────────────────────────
        # For each response fragment, check if the cited docs actually
        # support the fragment text (not just any doc in the collection).
        citation_cache_path = f"{filebase}.citation_cache.json"
        citation_cache = load_cache(citation_cache_path)

        cite_pairs: List[Tuple[str, str]] = []
        cite_pair_index: List[Tuple[Tuple[str, str], int]] = []
        # (run_id, topic_id), fragment_index

        # Track per-response citation info: {key: (supported_count, total_cited_fragments)}
        citation_info: Dict[Tuple[str, str], Tuple[int, int]] = {}

        for response in responses:
            if not response.responses:
                continue
            key = (response.metadata.run_id, response.metadata.topic_id)
            docs = response.documents or {}

            supported = 0
            total_cited = 0

            for frag_idx, fragment in enumerate(response.responses):
                # Get cited doc IDs from this fragment
                cited_ids = []
                if fragment.citations:
                    if isinstance(fragment.citations, dict):
                        cited_ids = list(fragment.citations.keys())
                    elif isinstance(fragment.citations, list):
                        cited_ids = [str(c) for c in fragment.citations]

                if not cited_ids:
                    continue  # fragment has no citations, skip

                total_cited += 1
                cache_key = f"{key[0]}_{key[1]}_{frag_idx}"

                if cache_key in citation_cache:
                    if citation_cache[cache_key] == 1:
                        supported += 1
                    continue

                # Check if any of the cited docs support this fragment
                frag_text = fragment.text
                for cid in cited_ids:
                    if cid in docs:
                        cite_pairs.append((docs[cid].text, frag_text))
                        cite_pair_index.append((key, frag_idx))
                        break  # one NLI check per fragment (against first matching cited doc)

            citation_info[key] = (supported, total_cited)

        # Run NLI on citation pairs
        if cite_pairs:
            for i in range(0, len(cite_pairs), CHUNK_SIZE):
                chunk_scores = nli_model.predict(cite_pairs[i:i + CHUNK_SIZE])
                for j, score in enumerate(chunk_scores):
                    idx = i + j
                    key, frag_idx = cite_pair_index[idx]
                    cache_key = f"{key[0]}_{key[1]}_{frag_idx}"
                    # nli-deberta: [contradiction, entailment, neutral]
                    # Use stricter threshold: entailment must be the dominant class
                    is_supported = 1 if (score[1] > score[0] and score[1] > score[2]) else 0
                    citation_cache[cache_key] = is_supported
                    if is_supported:
                        prev_sup, prev_total = citation_info[key]
                        citation_info[key] = (prev_sup + 1, prev_total)

        save_cache(citation_cache, citation_cache_path)

        # ── Stage 3: Pairwise preference evaluation + score aggregation ──

        # Collect FINAL_SCOREs for anchor selection
        original_scores: Dict[Tuple[str, str], float] = {}
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            topic_id = response.metadata.topic_id
            text = response.get_report_text()
            text_id = doc_id_md5(text)
            comp = qrels_dict.get((topic_id, text_id), 0) / 3.0
            cite_sup, cite_total = citation_info.get(key, (0, 0))
            ca = cite_sup / cite_total if cite_total > 0 else 0.0
            original_scores[key] = comp * ca

        anchors = pick_anchors(original_scores)

        pairwise = PairwisePreferenceJudge()
        pairwise_scores = pairwise.run_pairwise(
            rag_responses=responses,
            rag_topics=rag_topics,
            llm_config=llm_config,
            anchors=anchors,
            filebase=filebase,
        )

        # Build final leaderboard with all 4 measures
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
            pairwise_val = pairwise_scores.get(key, 0.0)

            # Citation accuracy: fraction of cited fragments supported by their cited doc
            cite_sup, cite_total = citation_info.get(key, (0, 0))
            cite_acc = cite_sup / cite_total if cite_total > 0 else 0.0

            final = completeness * cite_acc

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=topic_id,
                values={
                    "COMPLETENESS_SCORE": completeness,
                    "ATTRIBUTION_SCORE": attribution,
                    "CITATION_ACCURACY": cite_acc,
                    "FINAL_SCORE": final,
                    "PAIRWISE_SCORE": pairwise_val,
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
