"""
DeepSeek-as-Judge: scores and ranks SFT model responses to create preference pairs.
Used in on-policy mode where the SFT model generates candidate responses.
"""

import json
import logging
from tqdm import tqdm

from src.data.function_pool import format_functions_for_prompt

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are an expert judge evaluating function calling quality.
Given a user query and available functions, score each candidate response.

Available functions:
{functions_text}

User query: {query}

Expected correct call:
{expected}

Candidate responses to evaluate:
{candidates}

Score each candidate on these 4 dimensions (0-5 each):
1. json_validity: Is the response valid JSON with "name" and "arguments" fields?
2. function_correctness: Is the correct function called?
3. argument_accuracy: Are argument names and values correct?
4. completeness: Are all required arguments present with reasonable values?

Return a JSON object:
{{
  "scores": [
    {{"index": 0, "json_validity": N, "function_correctness": N, "argument_accuracy": N, "completeness": N, "total": N, "reasoning": "..."}},
    ...
  ]
}}"""


class ResponseJudge:
    """Use DeepSeek to judge and rank SFT model responses."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def judge_responses(self, scenario, responses, functions_text):
        """Score multiple responses for a single scenario."""
        expected = json.dumps({
            "name": scenario["expected_function"],
            "arguments": scenario["expected_arguments"],
        }, indent=2)

        candidate_parts = []
        for i, resp in enumerate(responses):
            text = resp["raw"] if isinstance(resp, dict) else resp
            candidate_parts.append(f"Candidate {i}: {text}")

        prompt = JUDGE_PROMPT.format(
            functions_text=functions_text,
            query=scenario["query"],
            expected=expected,
            candidates="\n".join(candidate_parts),
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat(messages, temperature=0.1, max_tokens=2048, json_mode=True)
            parsed = self.llm.parse_json_response(result)
            scores = parsed.get("scores", parsed) if isinstance(parsed, dict) else parsed
            for s in scores:
                if "total" not in s:
                    s["total"] = sum(s.get(k, 0) for k in
                                     ["json_validity", "function_correctness", "argument_accuracy", "completeness"])
            return scores
        except Exception as e:
            logger.warning("Judging failed: %s", e)
            return None

    def build_pairs_from_samples(self, sampled_results, min_score_gap=3, progress=True):
        """Build preference pairs from judged SFT model samples.

        Selects (best, worst) from each scenario's candidates.
        Only keeps pairs where score gap >= min_score_gap for clear signal.
        """
        pairs = []
        iterator = tqdm(sampled_results, desc="Judging responses") if progress else sampled_results

        for item in iterator:
            scenario = item["scenario"]
            system_prompt = item["system_prompt"]
            responses = item["responses"]

            valid_responses = [r for r in responses if r["parsed"] is not None]
            if len(valid_responses) < 2:
                continue

            functions_text = format_functions_for_prompt(scenario["available_functions"])
            scores = self.judge_responses(scenario, valid_responses, functions_text)
            if scores is None or len(scores) < 2:
                continue

            scored = list(zip(valid_responses, scores))
            scored.sort(key=lambda x: x[1].get("total", 0), reverse=True)
            best_resp, best_score = scored[0]
            worst_resp, worst_score = scored[-1]

            gap = best_score.get("total", 0) - worst_score.get("total", 0)
            if gap < min_score_gap:
                continue

            pairs.append({
                "system": system_prompt,
                "query": scenario["query"],
                "chosen": best_resp["raw"],
                "rejected": worst_resp["raw"],
                "chosen_score": best_score.get("total", 0),
                "rejected_score": worst_score.get("total", 0),
                "score_gap": gap,
                "error_type": "on_policy",
                "expected_function": scenario["expected_function"],
            })

        logger.info("Built %d on-policy preference pairs (min gap=%d)", len(pairs), min_score_gap)
        return pairs
