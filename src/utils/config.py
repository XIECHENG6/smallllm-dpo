"""Load project configuration from settings.yaml."""

import os
import yaml

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")


def load_config(path=None):
    """Load YAML config. Falls back to config/settings.yaml relative to project root."""
    path = path or _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dpo_args_from_config(config=None):
    """Build DPOTrainingArgs from config dict, allowing notebook overrides."""
    from src.training.dpo_train import DPOTrainingArgs

    if config is None:
        config = load_config()

    dpo_cfg = config.get("dpo", {})
    qlora_cfg = config.get("qlora", {})

    return DPOTrainingArgs(
        base_model=config.get("sft_model", {}).get("base", "Qwen/Qwen2.5-3B-Instruct"),
        sft_adapter=config.get("sft_model", {}).get("adapter"),
        output_dir=dpo_cfg.get("output_dir", "./dpo_output"),
        beta=dpo_cfg.get("beta", 0.5),
        learning_rate=dpo_cfg.get("learning_rate", 5e-6),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=dpo_cfg.get("num_train_epochs", 1),
        max_length=dpo_cfg.get("max_length", 512),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 384),
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=dpo_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=dpo_cfg.get("logging_steps", 10),
        save_steps=dpo_cfg.get("save_steps", 200),
        lora_rank=qlora_cfg.get("lora_rank", 16),
        lora_alpha=qlora_cfg.get("lora_alpha", 32),
        lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        seed=config.get("data", {}).get("seed", 42),
    )
