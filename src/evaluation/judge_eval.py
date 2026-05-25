"""
LLM-as-Judge evaluation: head-to-head comparison between models.
DeepSeek judges which model's output is better for each test case.
Computes win rate, tie rate, and per-dimension analysis.
"""

import json
import logging
from tqdm import tqdm

from src.data.function_pool import format_functions_for_prompt

logger = logging.getLogger(__name__)

HEAD_TO_HEAD_PROMPT = """\
You are an expert judge comparing two function calling responses.

User query: {query}
Available functions:
{functions_text}

Expected correct call:
{expected}

Response A:
{response_a}

Response B:
{response_b}

Compare the two responses on:
1. Correctness: Is the right function called with correct arguments?
2. Completeness: Are all required arguments present?
3. Format quality: Is the JSON well-formed and clean?
4. Instruction following: Does it strictly output only the function call?

Return a JSON object:
{{
  "winner": "A" or "B" or "tie",
  "correctness_winner": "A" or "B" or "tie",
  "reasoning": "brief explanation",
  "confidence": "high" or "medium" or "low"
}}

Be objective. If both are equally good or bad, say "tie"."""


class JudgeEvaluator:
    """Head-to-head evaluation using DeepSeek as judge."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def judge_pair(self, query, functions_text, expected, response_a, response_b):
        prompt = HEAD_TO_HEAD_PROMPT.format(
            query=query,
            functions_text=functions_text,
            expected=json.dumps(expected, indent=2),
            response_a=response_a,
            response_b=response_b,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat(messages, temperature=0.1, max_tokens=512, json_mode=True)
            return self.llm.parse_json_response(result)
        except Exception as e:
            logger.warning("Judge call failed: %s", e)
            return None

    def evaluate_head_to_head(
        self, test_scenarios, predictions_a, predictions_b,
        model_a_name="Model A", model_b_name="Model B",
        swap_order=True,
    ):
        """Run head-to-head evaluation with optional position debiasing.

        Args:
            swap_order: If True, run each comparison twice with swapped positions
                        to cancel position bias. Winner must be consistent.
        """
        results = {"a_wins": 0, "b_wins": 0, "ties": 0, "errors": 0, "details": []}

        for i, scenario in enumerate(tqdm(test_scenarios, desc="Head-to-head judging")):
            functions_text = format_functions_for_prompt(scenario["available_functions"])
            expected = {
                "name": scenario["expected_function"],
                "arguments": scenario["expected_arguments"],
            }

            pred_a = predictions_a[i]
            pred_b = predictions_b[i]

            verdict_1 = self.judge_pair(scenario["query"], functions_text, expected, pred_a, pred_b)
            if verdict_1 is None:
                results["errors"] += 1
                continue

            winner = verdict_1.get("winner", "tie")

            if swap_order:
                verdict_2 = self.judge_pair(scenario["query"], functions_text, expected, pred_b, pred_a)
                if verdict_2 is None:
                    results["errors"] += 1
                    continue

                winner_2 = verdict_2.get("winner", "tie")
                swapped_winner_2 = "B" if winner_2 == "A" else ("A" if winner_2 == "B" else "tie")

                if winner != swapped_winner_2:
                    winner = "tie"

            if winner == "A":
                results["a_wins"] += 1
            elif winner == "B":
                results["b_wins"] += 1
            else:
                results["ties"] += 1

            results["details"].append({
                "index": i,
                "query": scenario["query"],
                "winner": winner,
                "reasoning": verdict_1.get("reasoning", ""),
            })

        total_valid = results["a_wins"] + results["b_wins"] + results["ties"]
        results["summary"] = {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "a_win_rate": results["a_wins"] / total_valid if total_valid else 0,
            "b_win_rate": results["b_wins"] / total_valid if total_valid else 0,
            "tie_rate": results["ties"] / total_valid if total_valid else 0,
            "total_comparisons": total_valid,
            "errors": results["errors"],
        }

        return results

    @staticmethod
    def print_summary(results):
        s = results["summary"]
        print(f"\n{'='*50}")
        print(f"Head-to-Head: {s['model_a']} vs {s['model_b']}")
        print(f"{'='*50}")
        print(f"  {s['model_a']} wins: {results['a_wins']:>4} ({s['a_win_rate']:.1%})")
        print(f"  {s['model_b']} wins: {results['b_wins']:>4} ({s['b_win_rate']:.1%})")
        print(f"  Ties:             {results['ties']:>4} ({s['tie_rate']:.1%})")
        print(f"  Total:            {s['total_comparisons']:>4}")
        if s["errors"]:
            print(f"  Errors:           {s['errors']:>4}")
        print()
