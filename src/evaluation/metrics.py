"""
Standard function calling evaluation metrics.
Reuses the same 5-metric framework from Phase 1 for direct comparison.
"""

import json
import logging

from src.data.prompt_templates import build_fc_system_prompt, build_chat_messages

logger = logging.getLogger(__name__)


def parse_function_call(text):
    """Parse a function call JSON from model output.
    Uses first-object-only strategy (Phase 1 lesson: prevents redundant generation errors).
    """
    text = text.strip()
    if not text:
        return None

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end_marker = text.find("```", start)
            block = text[start:end_marker].strip() if end_marker > start else text[start:].strip()
            try:
                obj = json.loads(block)
                if isinstance(obj, dict) and "name" in obj:
                    return obj
            except json.JSONDecodeError:
                pass

    # First-object-only: find first complete { ... }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            try:
                obj = json.loads(text[start:i + 1])
                if isinstance(obj, dict) and "name" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            break

    return None


def compute_fc_metrics(predictions, references):
    """Compute the 5 standard function calling metrics.

    Args:
        predictions: list of model output strings
        references: list of dicts with 'name' and 'arguments'

    Returns:
        dict with json_valid_rate, name_accuracy, arg_names_accuracy,
             arg_values_accuracy, exact_match
    """
    n = len(predictions)
    json_valid = 0
    name_correct = 0
    arg_names_scores = []
    arg_values_scores = []
    exact_matches = 0

    for pred_text, ref in zip(predictions, references):
        pred = parse_function_call(pred_text)

        if pred is None:
            arg_names_scores.append(0.0)
            arg_values_scores.append(0.0)
            continue

        json_valid += 1

        ref_name = ref["name"]
        pred_name = pred.get("name", "")
        if pred_name == ref_name:
            name_correct += 1

        ref_args = ref.get("arguments", {})
        pred_args = pred.get("arguments", {})
        ref_keys = set(ref_args.keys())
        pred_keys = set(pred_args.keys())

        # Arg names: Jaccard similarity
        if ref_keys or pred_keys:
            intersection = ref_keys & pred_keys
            union = ref_keys | pred_keys
            arg_names_scores.append(len(intersection) / len(union))
        else:
            arg_names_scores.append(1.0)

        # Arg values: exact match per shared key
        shared_keys = ref_keys & pred_keys
        if shared_keys:
            correct_values = sum(
                1 for k in shared_keys
                if str(ref_args[k]).lower().strip() == str(pred_args.get(k, "")).lower().strip()
            )
            arg_values_scores.append(correct_values / len(ref_keys))
        else:
            arg_values_scores.append(0.0 if ref_keys else 1.0)

        # Exact match: name + all args
        if pred_name == ref_name and ref_keys == pred_keys:
            all_values_match = all(
                str(ref_args[k]).lower().strip() == str(pred_args.get(k, "")).lower().strip()
                for k in ref_keys
            )
            if all_values_match:
                exact_matches += 1

    return {
        "json_valid_rate": json_valid / n if n else 0,
        "name_accuracy": name_correct / n if n else 0,
        "arg_names_accuracy": sum(arg_names_scores) / n if n else 0,
        "arg_values_accuracy": sum(arg_values_scores) / n if n else 0,
        "exact_match": exact_matches / n if n else 0,
        "num_samples": n,
    }


def load_model_for_eval(model_path, base_model=None, adapter_path=None, sft_adapter=None):
    """Load a model for evaluation. Supports base, SFT, and DPO models.

    For DPO models trained on top of a merged SFT adapter, pass sft_adapter
    so the base weights are merged before the DPO adapter is applied.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = base_model or model_path
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if sft_adapter:
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def evaluate_model(model, tokenizer, test_scenarios, max_new_tokens=256):
    """Run evaluation on test scenarios and compute metrics."""
    import torch
    from tqdm import tqdm

    predictions = []
    references = []

    with torch.no_grad():
        for scenario in tqdm(test_scenarios, desc="Evaluating"):
            system_prompt = build_fc_system_prompt(scenario["available_functions"])
            messages = build_chat_messages(system_prompt, scenario["query"])
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            predictions.append(pred_text)
            references.append({
                "name": scenario["expected_function"],
                "arguments": scenario["expected_arguments"],
            })

    metrics = compute_fc_metrics(predictions, references)
    return metrics, predictions, references


def compare_models(results_dict):
    """Pretty-print comparison of multiple model evaluation results."""
    header = f"{'Model':<25} {'JSON%':>7} {'Name%':>7} {'ArgN%':>7} {'ArgV%':>7} {'EM%':>7}"
    print(header)
    print("-" * len(header))
    for name, metrics in results_dict.items():
        print(
            f"{name:<25} "
            f"{metrics['json_valid_rate']*100:>6.1f}% "
            f"{metrics['name_accuracy']*100:>6.1f}% "
            f"{metrics['arg_names_accuracy']*100:>6.1f}% "
            f"{metrics['arg_values_accuracy']*100:>6.1f}% "
            f"{metrics['exact_match']*100:>6.1f}%"
        )
