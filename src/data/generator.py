"""
Scenario generator: uses DeepSeek API to create diverse function calling scenarios
and construct preference pairs (chosen / rejected) for DPO training.

Two modes:
  - synthetic: DeepSeek generates both chosen and rejected responses
  - on_policy: SFT model samples responses, DeepSeek judges and ranks them
"""

import json
import random
import logging
from tqdm import tqdm

from src.data.function_pool import FUNCTION_POOL, DOMAIN_MAP, format_functions_for_prompt
from src.data.prompt_templates import build_fc_system_prompt

logger = logging.getLogger(__name__)

SCENARIO_GENERATION_PROMPT = """\
You are a data generator for function calling training.
Given the available functions below, generate {batch_size} diverse and realistic user queries.
Each query should naturally require calling exactly ONE of the available functions.

Available functions:
{functions_text}

Requirements:
- Queries must be natural language (as a real user would type)
- Cover different phrasings: formal, casual, short, detailed
- Include queries in English
- Each query should clearly map to one function with specific argument values
- Vary the complexity: some simple (1-2 args), some complex (3+ args)

Return a JSON array where each element has:
- "query": the user's natural language query
- "expected_function": the correct function name
- "expected_arguments": dict of correct argument name-value pairs

Return ONLY the JSON array, no other text."""

ERROR_TYPES = [
    "wrong_function",
    "missing_required_arg",
    "hallucinated_arg_value",
    "extra_nonexistent_arg",
    "wrong_arg_type",
    "malformed_json_value",
]

REJECTION_GENERATION_PROMPT = """\
You are generating INTENTIONALLY INCORRECT function call responses for training.
Given the user query and the correct function call, generate a plausible but WRONG response.

User query: {query}
Available functions:
{functions_text}
Correct response: {correct_response}

Generate a wrong response with this specific error type: {error_type}

Error type explanations:
- wrong_function: Call a different function that doesn't match the query
- missing_required_arg: Omit a required argument
- hallucinated_arg_value: Use an incorrect value that doesn't match the query
- extra_nonexistent_arg: Add an argument that doesn't exist in the function definition
- wrong_arg_type: Use wrong type (string instead of number, etc.)
- malformed_json_value: Include a subtly malformed value (e.g. nested quote issues)

Return ONLY a JSON object with "name" and "arguments" fields. Make the error subtle and plausible — the kind of mistake a language model would actually make."""


class ScenarioGenerator:
    def __init__(self, llm_client, seed=42):
        self.llm = llm_client
        self.rng = random.Random(seed)

    def _create_malformed_value(self, correct_dict):
        """Programmatically corrupt a correct function call to create a malformed_json_value error.

        json_mode=True prevents the LLM from producing truly malformed values,
        so we apply deterministic corruption instead.
        """
        corrupted = json.loads(json.dumps(correct_dict))
        args = corrupted.get("arguments", {})
        if not args:
            return corrupted

        key = self.rng.choice(list(args.keys()))
        value = args[key]

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            args[key] = str(value)
        elif isinstance(value, str):
            if len(value) > 2:
                strategy = self.rng.choice(["extra_quote", "truncate", "case_swap"])
                if strategy == "extra_quote":
                    args[key] = value + '"'
                elif strategy == "truncate":
                    args[key] = value[: len(value) // 2]
                else:
                    args[key] = value.upper() if value != value.upper() else value.lower()
            else:
                strategy = self.rng.choice(["extra_quote", "duplicate"])
                if strategy == "extra_quote":
                    args[key] = value + '"'
                else:
                    args[key] = value + value
        elif isinstance(value, bool):
            args[key] = str(value).lower()
        else:
            args[key] = str(value)

        return corrupted

    def _select_function_subset(self, min_funcs=3, max_funcs=6):
        """Select a random subset of functions, ensuring domain diversity."""
        domains = list(DOMAIN_MAP.keys())
        self.rng.shuffle(domains)
        selected_names = []
        for domain in domains[:self.rng.randint(2, len(domains))]:
            funcs_in_domain = DOMAIN_MAP[domain]
            k = min(self.rng.randint(1, 2), len(funcs_in_domain))
            selected_names.extend(self.rng.sample(funcs_in_domain, k))
        if len(selected_names) < min_funcs:
            remaining = [f["name"] for f in FUNCTION_POOL if f["name"] not in selected_names]
            extra = self.rng.sample(remaining, min(min_funcs - len(selected_names), len(remaining)))
            selected_names.extend(extra)
        selected_names = selected_names[:max_funcs]
        pool = {f["name"]: f for f in FUNCTION_POOL}
        return [pool[n] for n in selected_names]

    def generate_scenarios(self, num_scenarios, batch_size=10):
        """Generate diverse function calling scenarios using DeepSeek."""
        all_scenarios = []
        num_batches = (num_scenarios + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating scenarios"):
            remaining = min(batch_size, num_scenarios - len(all_scenarios))
            functions = self._select_function_subset()
            functions_text = format_functions_for_prompt(functions)

            prompt = SCENARIO_GENERATION_PROMPT.format(
                batch_size=remaining,
                functions_text=functions_text,
            )
            messages = [{"role": "user", "content": prompt}]

            try:
                response = self.llm.chat(messages, temperature=0.9, max_tokens=4096, json_mode=True)
                parsed = self.llm.parse_json_response(response)
                if isinstance(parsed, dict) and "scenarios" in parsed:
                    parsed = parsed["scenarios"]
                if not isinstance(parsed, list):
                    parsed = [parsed]

                for item in parsed:
                    if not all(k in item for k in ("query", "expected_function", "expected_arguments")):
                        continue
                    item["available_functions"] = functions
                    all_scenarios.append(item)
            except Exception as e:
                logger.warning("Batch %d failed: %s", batch_idx, e)
                continue

            if len(all_scenarios) >= num_scenarios:
                break

        return all_scenarios[:num_scenarios]

    def generate_rejected_response(self, scenario, error_type=None):
        """Generate a plausible but incorrect response for a scenario."""
        if error_type is None:
            error_type = self.rng.choice(ERROR_TYPES)

        correct_dict = {
            "name": scenario["expected_function"],
            "arguments": scenario["expected_arguments"],
        }

        if error_type == "malformed_json_value":
            corrupted = self._create_malformed_value(correct_dict)
            return {"response": corrupted, "error_type": error_type}

        correct_response = json.dumps(correct_dict)
        functions_text = format_functions_for_prompt(scenario["available_functions"])

        prompt = REJECTION_GENERATION_PROMPT.format(
            query=scenario["query"],
            functions_text=functions_text,
            correct_response=correct_response,
            error_type=error_type,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.chat(messages, temperature=0.7, max_tokens=512, json_mode=True)
            rejected = self.llm.parse_json_response(response)
            if "name" in rejected and "arguments" in rejected:
                return {"response": rejected, "error_type": error_type}
        except Exception as e:
            logger.warning("Rejection generation failed: %s", e)

        return None

    def build_preference_pairs(self, scenarios, progress=True):
        """Build (chosen, rejected) pairs from scenarios using DeepSeek."""
        pairs = []
        iterator = tqdm(scenarios, desc="Building preference pairs") if progress else scenarios

        for scenario in iterator:
            chosen = {
                "name": scenario["expected_function"],
                "arguments": scenario["expected_arguments"],
            }

            error_type = self.rng.choice(ERROR_TYPES)
            rejected_result = self.generate_rejected_response(scenario, error_type)
            if rejected_result is None:
                continue

            system_prompt = build_fc_system_prompt(scenario["available_functions"])

            pairs.append({
                "system": system_prompt,
                "query": scenario["query"],
                "chosen": json.dumps(chosen, ensure_ascii=False),
                "rejected": json.dumps(rejected_result["response"], ensure_ascii=False),
                "error_type": rejected_result["error_type"],
                "expected_function": scenario["expected_function"],
            })

        return pairs
