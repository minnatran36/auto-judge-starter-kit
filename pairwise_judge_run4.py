#!/usr/bin/env python3
"""
Anchor-based pairwise preference judge.

For each topic, uses a provided anchor (best system), then asks an LLM
to compare every other system's response against the anchor.
Results are mapped to numeric scores.

fix2-5: each comparison is run twice with swapped positions (A/B) to
counteract LLM positional bias. Final score = average of both directions.
"""

import asyncio
import dataclasses
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from autojudge_base import Report, Request, LlmConfigProtocol
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, OpenAIMinimaLlm

# Letter → score mapping
CATEGORY_SCORES = {
    "A": 2.0,
    "B": 1.5,
    "C": 1.0,
    "D": 0.5,
    "E": 0.0,
}

# fix2-5: reverse mapping for swapped position
# when anchor is in position B, the meaning of letters flips
CATEGORY_SCORES_REVERSED = {
    "A": 0.0,   # "B much better" means anchor much better → system much worse
    "B": 0.5,   # "B slightly better" means anchor slightly better → system slightly worse
    "C": 1.0,   # equal stays equal
    "D": 1.5,   # "B slightly worse" means anchor slightly worse → system slightly better
    "E": 2.0,   # "B much worse" means anchor much worse → system much better
}

PAIRWISE_SYSTEM_PROMPT = (
    "You are an evaluation expert. Given a question and two responses, "
    "compare Response B to Response A (the reference).\n\n"
    "Pick ONE:\n"
    "  A: Response B is much better than Response A\n"
    "  B: Response B is slightly better than Response A\n"
    "  C: Response B is about equal to Response A\n"
    "  D: Response B is slightly worse than Response A\n"
    "  E: Response B is much worse than Response A\n\n"
    "Reply with ONLY the letter (A, B, C, D, or E)."
)

#---------------------- UTILITIES ----------------------

def _parse_letter(result) -> str:
    """Extract a category letter from LLM response."""
    try:
        text = result.text.strip().upper()
        for letter in CATEGORY_SCORES:
            if text.startswith(letter):
                return letter
    except (AttributeError, TypeError):
        pass
    return "E"


def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_original_scores(path: str, measure: str = "FINAL_SCORE") -> Dict[Tuple[str, str], float]:
    """Parse eval.txt → {(run_id, topic_id): score}"""
    scores: Dict[Tuple[str, str], float] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            run_id, meas, topic_id, val = parts
            if meas != measure or topic_id == "all":
                continue
            scores[(run_id, topic_id)] = float(val)
    return scores


def pick_anchors(original_scores: Dict[Tuple[str, str], float]) -> Dict[str, str]:
    """Pick best system per topic → {topic_id: run_id}"""
    by_topic: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (run_id, topic_id), score in original_scores.items():
        by_topic[topic_id][run_id] = score
    return {
        topic_id: max(runs, key=runs.get)
        for topic_id, runs in by_topic.items()
    }


#---------------------- PAIRWISE JUDGE ----------------------

class PairwisePreferenceJudge:

    def run_pairwise(
        self,
        rag_responses: List[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        anchors: Dict[str, str],
        **kwargs: Any,
    ) -> Dict[Tuple[str, str], float]:

        filebase = kwargs.get("filebase", "output-kiddie/minna_judge")
        cache_path = f"{filebase}.pairwise_cache.json"
        cache = load_cache(cache_path)

        full_config = (
            MinimaLlmConfig.from_dict(llm_config.raw)
            if llm_config.raw
            else MinimaLlmConfig.from_env()
        )
        full_config = dataclasses.replace(
            full_config, rpm=300, max_attempts=100, max_outstanding=8
        )

        backend = OpenAIMinimaLlm(full_config)
        topic_questions = {t.request_id: t.problem_statement for t in rag_topics}

        # ── 1. text_dict: all response texts ──────────────────────────
        text_dict: Dict[Tuple[str, str], str] = {}
        for response in rag_responses:
            text_dict[(response.metadata.run_id, response.metadata.topic_id)] = response.get_report_text()

        # ── 2. anchor_dict: anchor text per topic ─────────────────────
        anchor_dict: Dict[str, str] = {}
        for topic_id, run_id in anchors.items():
            if (run_id, topic_id) in text_dict:
                anchor_dict[topic_id] = text_dict[(run_id, topic_id)]

        # ── 3. score_dict: prefill from anchors + cache ───────────────
        score_dict: Dict[Tuple[str, str], float] = {}

        # anchors get 1.0
        for topic_id, anchor_id in anchors.items():
            if (anchor_id, topic_id) in text_dict:
                score_dict[(anchor_id, topic_id)] = 1.0

        # fix2-5: cache now stores both directions: "fwd" and "rev"
        # a fully cached pair has both "{rid}_{tid}_fwd" and "{rid}_{tid}_rev"
        # average them to get the final score
        for (rid, tid) in text_dict:
            fwd_key = f"{rid}_{tid}_fwd"
            rev_key = f"{rid}_{tid}_rev"
            if fwd_key in cache and rev_key in cache:
                fwd_score = CATEGORY_SCORES.get(cache[fwd_key], 0.0)
                rev_score = CATEGORY_SCORES_REVERSED.get(cache[rev_key], 0.0)
                score_dict[(rid, tid)] = (fwd_score + rev_score) / 2.0

        # ── 4. build requests for uncached pairs ──────────────────────
        # fix2-5: build two requests per pair (forward + reverse position)
        requests_info: List[Tuple[str, str, str]] = []  # (run_id, topic_id, direction)

        for (rid, tid) in sorted(text_dict):
            if (rid, tid) in score_dict:
                continue
            if tid not in anchor_dict:
                continue

            # forward: anchor=A, system=B (need if not cached)
            if f"{rid}_{tid}_fwd" not in cache:
                requests_info.append((rid, tid, "fwd"))

            # reverse: system=A, anchor=B (need if not cached)
            if f"{rid}_{tid}_rev" not in cache:
                requests_info.append((rid, tid, "rev"))

        # ── 5. send to LLM ───────────────────────────────────────────
        if requests_info:
            llm_requests = []
            for rid, tid, direction in requests_info:
                question = topic_questions.get(tid, "")
                if direction == "fwd":
                    # fix2-5: forward — anchor in position A, system in position B
                    resp_a = anchor_dict[tid]
                    resp_b = text_dict[(rid, tid)]
                else:
                    # fix2-5: reverse — system in position A, anchor in position B
                    resp_a = text_dict[(rid, tid)]
                    resp_b = anchor_dict[tid]

                llm_requests.append(MinimaLlmRequest(
                    request_id=f"pw_{tid}_{rid}_{direction}",
                    messages=[
                        {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Question: {question}\n\n"
                            f"Response A (reference):\n{resp_a}\n\n"
                            f"Response B:\n{resp_b}"
                        )},
                    ],
                    temperature=0.0,
                ))

            print(f"PairwiseJudge: Sending {len(llm_requests)} LLM requests (fix2-5: includes swapped pairs)...")
            results = asyncio.run(backend.run_batched(llm_requests))

            # ── 6. zip results back into cache ────────────────────────
            for (rid, tid, direction), result in zip(requests_info, results):
                letter = _parse_letter(result)
                cache[f"{rid}_{tid}_{direction}"] = letter

            save_cache(cache, cache_path)

        # ── 7. build final scores from cache (both directions) ────────
        # fix2-5: average forward and reverse scores for each pair
        for (rid, tid) in sorted(text_dict):
            if (rid, tid) in score_dict:
                continue
            if tid not in anchor_dict:
                continue

            fwd_letter = cache.get(f"{rid}_{tid}_fwd", "E")
            rev_letter = cache.get(f"{rid}_{tid}_rev", "E")
            fwd_score = CATEGORY_SCORES.get(fwd_letter, 0.0)
            rev_score = CATEGORY_SCORES_REVERSED.get(rev_letter, 0.0)
            score_dict[(rid, tid)] = (fwd_score + rev_score) / 2.0

        print(f"PairwiseJudge: Scored {len(score_dict)} (run, topic) pairs")
        return score_dict
