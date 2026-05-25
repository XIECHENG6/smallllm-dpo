"""
Gradio demo: compare Base / SFT / DPO models on function calling.
Designed for HuggingFace Spaces deployment.
"""

import os
import json
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.data.function_pool import FUNCTION_POOL, get_functions_by_names
from src.data.prompt_templates import build_fc_system_prompt, build_chat_messages
from src.evaluation.metrics import parse_function_call

MODELS = {}


def load_models():
    """Load all model variants for comparison, reusing base weights where possible."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    sft_adapter = os.environ.get("SFT_ADAPTER", "")
    dpo_adapter = os.environ.get("DPO_ADAPTER", "")

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    MODELS["base"] = base_model

    if sft_adapter:
        sft_model = PeftModel.from_pretrained(base_model, sft_adapter)
        MODELS["sft"] = sft_model

    if dpo_adapter:
        if sft_adapter:
            dpo_base = AutoModelForCausalLM.from_pretrained(
                base_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
            )
            dpo_base = PeftModel.from_pretrained(dpo_base, sft_adapter)
            dpo_base = dpo_base.merge_and_unload()
            dpo_model = PeftModel.from_pretrained(dpo_base, dpo_adapter)
        else:
            dpo_model = PeftModel.from_pretrained(base_model, dpo_adapter)
        MODELS["dpo"] = dpo_model

    MODELS["tokenizer"] = tokenizer
    return tokenizer


def generate_response(model_key, query, selected_functions):
    if model_key not in MODELS:
        return f"Model '{model_key}' not loaded"

    model = MODELS[model_key]
    tokenizer = MODELS["tokenizer"]

    func_names = [f.strip() for f in selected_functions.split(",") if f.strip()]
    functions = get_functions_by_names(func_names)
    if not functions:
        functions = FUNCTION_POOL[:5]

    system_prompt = build_fc_system_prompt(functions)
    messages = build_chat_messages(system_prompt, query)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    parsed = parse_function_call(text)
    if parsed:
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        return f"**Raw output:**\n```\n{text}\n```\n\n**Parsed:**\n```json\n{formatted}\n```"
    return f"**Raw output:**\n```\n{text}\n```\n\n**Parsed:** Failed to parse as function call"


def compare_all(query, selected_functions):
    outputs = {}
    for key in ["base", "sft", "dpo"]:
        if key in MODELS:
            outputs[key] = generate_response(key, query, selected_functions)
        else:
            outputs[key] = "Model not loaded"
    return outputs.get("base", ""), outputs.get("sft", ""), outputs.get("dpo", "")


EXAMPLE_QUERIES = [
    ["What's the weather like in Shanghai?", "get_weather, search_web, get_news"],
    ["Book a flight from Beijing to Tokyo on 2026-07-01", "search_flights, book_hotel, get_directions"],
    ["Send an email to alice@example.com about the meeting tomorrow", "send_email, create_calendar_event, set_reminder"],
    ["How much is 500 USD in Japanese Yen?", "convert_currency, get_exchange_rate, get_stock_price"],
    ["Translate 'Hello World' to Chinese", "translate_text, summarize_text, search_web"],
]

ALL_FUNCTION_NAMES = ", ".join(f["name"] for f in FUNCTION_POOL)


def create_demo():
    with gr.Blocks(title="SmallLLM-DPO: Function Calling Alignment") as demo:
        gr.Markdown("# SmallLLM-DPO: Function Calling Alignment with DPO")
        gr.Markdown(
            "Compare **Base** (Qwen2.5-3B-Instruct) vs **SFT** (Phase 1 fine-tuned) vs "
            "**SFT+DPO** (preference-aligned) on function calling tasks."
        )

        with gr.Row():
            query_input = gr.Textbox(label="User Query", placeholder="e.g., What's the weather in Tokyo?", lines=2)
            func_input = gr.Textbox(label="Available Functions (comma-separated)", value="get_weather, search_web, get_news", lines=2)

        compare_btn = gr.Button("Compare All Models", variant="primary")

        with gr.Row():
            base_output = gr.Markdown(label="Base Model")
            sft_output = gr.Markdown(label="SFT Model")
            dpo_output = gr.Markdown(label="SFT + DPO Model")

        compare_btn.click(compare_all, inputs=[query_input, func_input], outputs=[base_output, sft_output, dpo_output])

        gr.Examples(examples=EXAMPLE_QUERIES, inputs=[query_input, func_input])

        gr.Markdown(f"**All available functions:** {ALL_FUNCTION_NAMES}")

    return demo


if __name__ == "__main__":
    print("Loading models...")
    load_models()
    print("Starting demo...")
    demo = create_demo()
    demo.launch(share=True)
