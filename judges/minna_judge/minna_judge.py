#!/usr/bin/env python3

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import asyncio
import json
import dataclasses
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
    doc_id=lambda r: doc_id_md5(r.text),  # Hash response text as doc_id
    grade=lambda r: r.grade,
    on_duplicate="keep_max",  # Keep highest grade if duplicates
)



class MinnaNuggetCreator:
    # Declare the nugget format this creator produces
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        # Settings from workflow.yml nugget_settings
        max_nuggets: int = 10,
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)

        backend = OpenAIMinimaLlm(full_config)
        
        banks: List[NuggetBank] = []
        requests = []

        for topic in rag_topics:            
            requests.append(MinimaLlmRequest(
                request_id=topic.request_id,
                messages=[
                    {"role": "system", "content": "Extract distinct sub-questions. "
                "Return ONLY a JSON array with no other text. Each element should have 'question' and 'answer' fields. "
                "Example: [{\"question\": \"Why is the puppy so cute?\", \"answer\": \"Because he's chubby.\"}]"},
                    {"role": "user", "content": topic.problem_statement},],
                   temperature=0.3,))
            
        results = asyncio.run(backend.run_batched(requests))

        for topic, result in zip(rag_topics, results):
            bank = NuggetBank(
                query_id = topic.request_id,
                title_query = topic.title or topic.request_id,
            )

            parsed = json.loads(result.text)
            questions = []
            for item in parsed[:max_nuggets]:
                questions.append(NuggetQuestion.from_lazy(
                    query_id = topic.request_id,
                    question = item["question"],
                    gold_answers = [item["answer"]],
                ))
            bank.add_nuggets(questions)
            banks.append(bank)
                
           
        return NuggetBanks.from_banks_list(banks)



class MinnaQrelsCreator:
    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        # Settings from workflow.yml qrels_settings
        grade_range: Tuple[int, int] = (0, 3),
        length_threshold: int = 100,
        **kwargs: Any,
    ) -> Optional[Qrels]:
        

        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        backend = OpenAIMinimaLlm(full_config)

        responses = list(rag_responses) #convert from Iterable to list to reuse
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
                    request_id=f"{response.metadata.run_id}_{topic_id}",
                    messages=[
                        {"role": "system", "content": "Does this response answer this question? Reply 1 for yes, 0 for no."},
                        {"role": "user", "content": f"Question: {nugget.question}\n\nResponse: {text}"},
                    ],
                    temperature=0.0,
                    )
                ))
        results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))
        scores = {}

        for(run_id, topic_id, _), result in zip(requests_info, results):
            score = self._parse_binary(result)
            scores.setdefault((run_id, topic_id), []).append(score)

        for response in responses:
            key = (response.metadata.run_id, response.metadata.topic_id)
            text = response.get_report_text()
            topic_id = response.metadata.topic_id
            if key not in scores:
                continue
            
            tallies = scores[key]
            grade = round(3*(sum(tallies)/len(tallies)))
            grade_records.append(GradeRecord(topic_id, text, grade))
        
        qrels = build_qrels(records=grade_records, spec=MINIMAL_QRELS_SPEC)
        print(f"MinnaQrelsCreator: Created qrels for {len(grade_records)} responses")
        return qrels

    def _parse_binary(self, result) -> int:
        try:
            text = result.text.strip().lower()
            if(text.startswith("1") or text.startswith("yes")):
                return 1
            return 0
        except:
            return 0
        
# =============================================================================
# ExampleLeaderboardJudge - LeaderboardJudgeProtocol
# =============================================================================

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
        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        full_config = dataclasses.replace(full_config, rpm=300, max_attempts=100, max_outstanding=8)
        
        backend = OpenAIMinimaLlm(full_config)

        builder: LeaderboardBuilder = LeaderboardBuilder(MINIMAL_SPEC)
        responses = list(rag_responses) #convert from Iterable to list to reuse
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []

   
    
        for response in responses:
            topic_id = response.metadata.topic_id
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

        qrels_dict= {}
        if qrels:
            for row in qrels.rows:
                qrels_dict[(row.topic_id, row.doc_id)] = row.grade

        for(run_id, topic_id, _), result in zip(requests_info, results):
            try: 
                parsed = json.loads(result.text)
            except json.JSONDecodeError:
                parsed = []
            claims[(run_id, topic_id)] = parsed


       
        for response in responses:
            scores = 0
            key = (response.metadata.run_id, response.metadata.topic_id)  
            text = response.get_report_text()    
            text_id = doc_id_md5(text) 
            topic_id = response.metadata.topic_id

            for claim in claims[(key)]:
                supported = False
                for doc_id, doc in response.documents.items():
                    score = nli_model.predict([(doc.text, claim)])
                    if score[0][1] > 0.5:
                        supported = True
                        break
                if supported: scores += 1
            attribution = scores/len(claims[key]) if claims[key] else 0.0
           
            completeness = (qrels_dict.get((topic_id, text_id), 0) / 3.0)

            builder.add(
                run_id = response.metadata.run_id,
                topic_id = topic_id,               
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
# CLI Entry Point (optional - for direct execution)
# =============================================================================
# Note: With modular classes, prefer running via:
#   auto-judge run --workflow workflow.yml
#
# For backwards compatibility, we still support direct CLI execution
# by combining all three protocols into a single object.

if __name__ == "__main__":
  
 
    from autojudge_base import AutoJudge, auto_judge_to_click_command
    class CompleteMinnaJudge(AutoJudge):
        #Combined class for CLI compatibility.
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


"""  
    testing data
    import json
    with open("data/kiddie/runs/repgen/beet") as f:
        data = json.loads(f.readline())
    print("run id: ", data["metadata"]["run_id"])
    print("topic id: ", data["metadata"]["topic_id"])
    print("no. documents: ", len(data["documents"]))
    

    response = data["responses"][0]
    print(type(response))
    print(dir(response))
    print(hasattr(response, 'documents'))
    print(hasattr(response, 'raw'))
    print(hasattr(response, 'data'))
    print("first citation: ", first_sentence["citations"])

    first_doc_id = list(data["documents"].keys())[0]
    print("first doc id: ", data["documents"][first_doc_id])
  
"""  
