"""
On-policy sampler: loads an SFT model and generates multiple candidate responses
per scenario via temperature sampling. Used for on-policy DPO data generation.
"""

import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.data.function_pool import format_functions_for_prompt
from src.data.prompt_templates import build_fc_system_prompt
from src.evaluation.metrics import parse_function_call

logger = logging.getLogger(__name__)


class SFTSampler:
    """Sample multiple responses from an SFT model for each scenario."""

    def __init__(self, base_model_name, adapter_path=None, device_map="auto"):
        logger.info("Loading base model: %s", base_model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )

        if adapter_path:
            logger.info("Loading SFT adapter: %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
            logger.info("SFT adapter merged successfully")

        self.model.eval()

    def _build_prompt(self, system_prompt, query):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def sample_responses(self, system_prompt, query, n=4, temperature=0.8, max_new_tokens=256):
        """Generate n responses for a single query."""
        prompt = self._build_prompt(system_prompt, query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        responses = []
        for output in outputs:
            new_tokens = output[inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(text)

        return responses

    def sample_batch(self, scenarios, n_per_scenario=4, temperature=0.8):
        """Sample responses for a batch of scenarios."""
        results = []

        for scenario in tqdm(scenarios, desc="Sampling from SFT model"):
            system_prompt = build_fc_system_prompt(scenario["available_functions"])

            responses = self.sample_responses(
                system_prompt, scenario["query"],
                n=n_per_scenario, temperature=temperature,
            )

            parsed_responses = []
            for resp in responses:
                parsed_responses.append({"raw": resp, "parsed": parse_function_call(resp)})

            results.append({
                "scenario": scenario,
                "system_prompt": system_prompt,
                "responses": parsed_responses,
            })

        return results
