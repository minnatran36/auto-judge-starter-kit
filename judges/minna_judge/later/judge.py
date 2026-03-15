from .cache import load_cache, save_cache
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
    MeasureSpec("ATTRIBUTION_SCORE_LENIENT"),
    MeasureSpec("ATTRIBUTION_SCORE_STRICT"),
    MeasureSpec("FINAL_SCORE"),
))


class MinnaLeaderboardJudge:

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        # Settings from workflow.yml judge_settings
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
        responses = list(rag_responses) #convert from Iterable to list to reuse
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []

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
                        "You are a fact extractor. Extract only specific factual assertions from this response that are: "
                        "verifiable against a source document, not common knowledge, specific enough to be true or false. "
                        "Return ONLY a JSON array of strings with no other text. "
                        'Example: [\"claim1\", \"claim2\"]'
                    )},                                                                                       
                    {"role": "user", "content": f"Response: {text}"},
                ],
                temperature=0.0,
                )
            ))

        results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))

        claims = {}
        for key, value in claims_cache.items():
            r_id, t_id = key.split("_", 1)
            claims[((r_id, t_id))] = value

        pairs = [] #(doc, claim)
        pair_index = [] #(which ans the doc from)

        qrels_dict= {}
        if qrels:
            for row in qrels.rows:
                qrels_dict[(row.topic_id, row.doc_id)] = row.grade


        for(run_id, topic_id, _), result in zip(requests_info, results):
            key = f"{run_id}_{topic_id}"
            try: 
                parsed = json.loads(result.text)
            except (json.JSONDecodeError, AttributeError):
                parsed = []
            claims[(run_id, topic_id)] = parsed
            claims_cache[key] = parsed
        save_cache(claims_cache, claims_cache_path)

    # LENIENT ATTRIBUTION -- write function to extract nli scores
        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)  

            for claim_id, claim in enumerate(claims.get(key,[])):
                for doc_id, doc in response.documents.items():
                    pairs.append((doc.text, claim))
                    pair_index.append((key, claim_id))


        CHUNK_SIZE = 500
        all_scores = []
        if pairs:
            for i in range(0, len(pairs), CHUNK_SIZE):
                chunk_scores = nli_model.predict(pairs[i:i+CHUNK_SIZE])
                all_scores.extend(chunk_scores)


        score_dict = {}

        #test_scores[0] = [-4.96, 4.71, -1.23] -- score[1], logit
        for (key, claim_id), score in zip(pair_index, all_scores):
            if score[1] > 0.0:
                score_dict[(key, claim_id)] = True
            elif (key, claim_id) not in score_dict:
                score_dict[(key, claim_id)] = False
            

        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            topic_id = response.metadata.topic_id
            text = response.get_report_text()
            text_id = doc_id_md5(text)

            attr_score = 0
            claim_list = claims.get(key,[])

            if claim_list:
                for i in range(len(claim_list)):
                    attr_score += score_dict.get((key, i), False)

                attribution_lenient = attr_score/len(claim_list)  
            else:
                attribution_lenient = 0
            
            completeness = (qrels_dict.get((topic_id, text_id), 0) / 3.0)
            builder.add(
                run_id = response.metadata.run_id,
                topic_id = topic_id,               
                values={
                   "COMPLETENESS_SCORE": completeness,
                   "ATTRIBUTION_SCORE": attribution_lenient, 
                   "FINAL_SCORE": 0.5 * (completeness + attribution_lenient),
                },
            )
      

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )
        print(f"MinnaLeaderboardJudge: Built leaderboard with {len(leaderboard.entries)} entries")
        return leaderboard