"""
Format preference pairs into HuggingFace Dataset for TRL DPOTrainer.

Uses TRL's conversational format: each sample contains 'chosen' and 'rejected'
as lists of chat messages. TRL applies the tokenizer's chat template internally,
ensuring train/inference format consistency.
"""

import json
import logging
from datasets import Dataset

logger = logging.getLogger(__name__)


def pairs_to_dpo_format(pairs):
    """Convert preference pairs to TRL conversational format.

    TRL DPOTrainer (>=0.9) accepts:
        chosen:   [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
        rejected: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

    This ensures the tokenizer's chat template is applied automatically,
    avoiding any train/inference format mismatch.
    """
    records = []
    for pair in pairs:
        prefix = [
            {"role": "system", "content": pair["system"]},
            {"role": "user", "content": pair["query"]},
        ]
        records.append({
            "chosen": prefix + [{"role": "assistant", "content": pair["chosen"]}],
            "rejected": prefix + [{"role": "assistant", "content": pair["rejected"]}],
        })
    return records


def pairs_to_dataset(pairs, train_ratio=0.9, seed=42):
    """Convert preference pairs to train/test HuggingFace Datasets."""
    records = pairs_to_dpo_format(pairs)

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)

    logger.info("Dataset created: %d train, %d test", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


def save_pairs_jsonl(pairs, path):
    """Save preference pairs as JSONL for inspection and reproducibility."""
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Saved %d pairs to %s", len(pairs), path)


def load_pairs_jsonl(path):
    """Load preference pairs from JSONL."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info("Loaded %d pairs from %s", len(pairs), path)
    return pairs


def merge_pair_sources(*pair_lists):
    """Merge preference pairs from multiple sources (synthetic + on-policy)."""
    merged = []
    for pairs in pair_lists:
        merged.extend(pairs)
    logger.info("Merged %d total pairs from %d sources", len(merged), len(pair_lists))
    return merged


def dataset_statistics(pairs):
    """Compute dataset statistics for inspection."""
    error_types = {}
    functions = {}
    for pair in pairs:
        et = pair.get("error_type", "unknown")
        error_types[et] = error_types.get(et, 0) + 1
        fn = pair.get("expected_function", "unknown")
        functions[fn] = functions.get(fn, 0) + 1

    return {
        "total_pairs": len(pairs),
        "error_type_distribution": dict(sorted(error_types.items(), key=lambda x: -x[1])),
        "function_distribution": dict(sorted(functions.items(), key=lambda x: -x[1])),
        "avg_chosen_length": sum(len(p["chosen"]) for p in pairs) / max(len(pairs), 1),
        "avg_rejected_length": sum(len(p["rejected"]) for p in pairs) / max(len(pairs), 1),
    }
