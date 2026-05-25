"""
DPO training pipeline with QLoRA on small language models.

Supports three training configurations:
  1. base_only:  DPO directly on base model (Qwen2.5-3B-Instruct)
  2. sft_plus_dpo: Load SFT adapter → merge → DPO with new LoRA
  3. Custom: any HF model + optional adapter path

The reference model is handled implicitly by TRL when peft_config is provided:
the model *before* LoRA adaptation serves as the reference.
"""

import copy
import os
import torch
import logging
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingArgs:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    sft_adapter: Optional[str] = None
    output_dir: str = "./dpo_output"
    beta: float = 0.5
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    max_length: int = 512
    max_prompt_length: int = 384
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bf16: bool = True
    seed: int = 42


def load_model_and_tokenizer(args: DPOTrainingArgs):
    """Load base model with 4-bit quantization, optionally merge SFT adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading base model: %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.sft_adapter:
        logger.info("Loading and merging SFT adapter: %s", args.sft_adapter)
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged — model is now the SFT checkpoint")

    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def get_lora_config(args: DPOTrainingArgs):
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_dpo_config(args: DPOTrainingArgs):
    kwargs = dict(
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        bf16=args.bf16,
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
    )
    import inspect
    dpo_params = inspect.signature(DPOConfig.__init__).parameters
    if "max_length" in dpo_params:
        kwargs["max_length"] = args.max_length
    if "max_prompt_length" in dpo_params:
        kwargs["max_prompt_length"] = args.max_prompt_length
    return DPOConfig(**kwargs)


def train_dpo(train_dataset, eval_dataset, args: DPOTrainingArgs):
    """Full DPO training pipeline."""
    model, tokenizer = load_model_and_tokenizer(args)
    lora_config = get_lora_config(args)
    dpo_config = get_dpo_config(args)

    logger.info(
        "Starting DPO training — beta=%.2f, lr=%.1e, epochs=%d, rank=%d",
        args.beta, args.learning_rate, args.num_train_epochs, args.lora_rank,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    train_result = trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset) if eval_dataset else 0
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete. Model saved to %s", args.output_dir)
    return trainer, metrics


def run_dpo_sweep(train_dataset, eval_dataset, base_args: DPOTrainingArgs, sweep_param, sweep_values):
    """Run DPO training across multiple values of a hyperparameter for ablation.

    Automatically resumes interrupted sweeps by skipping values whose
    output directories already contain saved metrics.
    """
    import gc
    import json as _json

    results = []

    for value in sweep_values:
        args = copy.deepcopy(base_args)
        setattr(args, sweep_param, value)
        args.output_dir = os.path.join(base_args.output_dir, f"{sweep_param}_{value}")

        metrics_file = os.path.join(args.output_dir, "train_results.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = _json.load(f)
            results.append({"param": sweep_param, "value": value, "metrics": metrics})
            logger.info("Skipping %s=%s (already completed, loaded from %s)", sweep_param, value, metrics_file)
            continue

        logger.info("=== Sweep: %s = %s ===", sweep_param, value)
        trainer, metrics = train_dpo(train_dataset, eval_dataset, args)
        results.append({
            "param": sweep_param,
            "value": value,
            "metrics": metrics,
        })

        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GPU memory released after %s=%s", sweep_param, value)

    return results
