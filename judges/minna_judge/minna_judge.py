#!/usr/bin/env python3

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
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")


MINIMAL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("COMPLETENESS_SCORE"),
    MeasureSpec("ATTRIBUTION_SCORE"),
    MeasureSpec("FINAL_SCORE"),
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

        banks: List[NuggetBank] = []
        requests = []

        for topic in rag_topics:
            # FIX: include title alongside problem_statement, with guard for missing title
            if topic.title:
                context = f"Title: {topic.title}\n\nQuestion: {topic.problem_statement}"
            else:
                context = topic.problem_statement
            
            # Note1
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
            except (json.JSONDecodeError, AttributeError):
                parsed = []

            questions = []
            for item in parsed[:limit]:
                questions.append(NuggetQuestion.from_lazy(
                    query_id=topic.request_id,
                    question=item,
                    gold_answers=[], #no gold answer here
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
                        request_id=f"{response.metadata.run_id}_{topic_id}_{nugget.question_id}",
                        messages=[
                            {"role": "system", "content": "Does this response answer this question? Reply 1 for yes, 0 for no."},
                            {"role": "user", "content": f"Question: {nugget.question}\n\nResponse: {text}"},
                        ],
                        temperature=0.0,
                    )
                ))

        results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))
        scores = {}

        for (run_id, topic_id, _), result in zip(requests_info, results):
            score = self._parse_binary(result)
            scores.setdefault((run_id, topic_id), []).append(score)

        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            text = response.get_report_text()
            topic_id = response.metadata.topic_id
            if key not in scores:
                continue

            tallies = scores[key]
            grade = round(3 * (sum(tallies) / len(tallies)))
            grade_records.append(GradeRecord(topic_id, text, grade))

        qrels = build_qrels(records=grade_records, spec=MINIMAL_QRELS_SPEC)
        print(f"MinnaQrelsCreator: Created qrels for {len(grade_records)} responses")
        return qrels

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
        builder: LeaderboardBuilder = LeaderboardBuilder(MINIMAL_SPEC)
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
            # FIX: filter out very short claims (< 10 chars)
            parsed = [c for c in parsed if isinstance(c, str) and len(c.strip()) >= 10]
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
        """
        response.documents:
        {
            "doc_001": Document(text="The treaty was signed in..."),
            "doc_002": Document(text="Carbon emissions data shows..."),
            ...
        }
        documents has many doc, 
        each doc = doc_id & text
        """

        for response in responses:
            if not response.documents:
                continue

            key = (response.metadata.run_id, response.metadata.topic_id)

            # claims.get(key, []) = a list of claims
            for claim in claims.get(key, []):
                claim_supported = False
                uncached_docs: List[Tuple[str, Any]] = []
                #[doc_id, doc]
                for doc_id, doc in response.documents.items():
                    key_str = f"{key[0]}_{key[1]}_{doc_id}_{claim}"
                    if key_str in nli_scores_cache:
                        if nli_scores_cache[key_str] == 1:
                            claim_supported = True
                            # Keep iterating to collect any remaining uncached docs
                            # but we won't use them if claim is already supported.
                    else:
                        uncached_docs.append((doc_id, doc))

                score_dict[(key, claim)] = 1 if claim_supported else 0

                if not claim_supported:
                    # Only queue NLI for docs whose result we don't know yet
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
                # score layout from nli-deberta: [contradiction, entailment, neutral] logits
                raw_nli = 1 if score[1] > 0.0 else 0
                nli_scores_cache[key_str] = raw_nli
                if raw_nli == 1:
                    score_dict[(key, claim)] = 1
                # score_dict default already set to 0 above

        save_cache(nli_scores_cache, nli_scores_cache_path)

        # ── Stage 3: Score aggregation ────────────────────────────────────────

        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            topic_id = response.metadata.topic_id
            text = response.get_report_text()
            text_id = doc_id_md5(text)

            claim_list = claims.get(key, [])

            if claim_list:
                attr_score = sum(score_dict.get((key, claim), 0) for claim in claim_list)
                attribution = attr_score / len(claim_list)
            else:
                attribution = 0

            completeness = qrels_dict.get((topic_id, text_id), 0) / 3.0

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=topic_id,
                values={
                    "COMPLETENESS_SCORE": completeness,
                    "ATTRIBUTION_SCORE": attribution,
                    "FINAL_SCORE": 0.5 * (completeness + attribution),
                },
            )

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )
        print(f"MinnaLeaderboardJudge: Built leaderboard with {len(leaderboard.entries)} entries")
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
