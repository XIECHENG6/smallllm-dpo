# SmallLLM-DPO: Preference Alignment for Small Model Function Calling

> 用 DeepSeek 合成偏好数据，对 Qwen2.5-3B 的 Function Calling 能力做 DPO 对齐。探索合成偏好数据在已具备强 FC 能力的 Instruct 模型上的有效性。

## Motivation

Phase 1 ([small-llms-tool-use](https://github.com/XIECHENG6/small-llms-tool-use)) 证明了 SFT 可以让小模型达到 87.8% Exact Match。本项目用 DPO (Direct Preference Optimization) 探索偏好对齐能否进一步提升：

1. **减少参数幻觉** — 不再编造不存在的参数值
2. **提升格式稳定性** — 更少的冗余生成和 JSON 格式错误
3. **增强安全性** — 更好地抵抗 prompt injection 攻击

## Results

### Main Comparison (200 test scenarios)

| Model | JSON Valid | Name Acc | Arg Names | Arg Values | Exact Match |
|-------|:---------:|:--------:|:---------:|:----------:|:-----------:|
| **Base (zero-shot)** | **93.0%** | **92.5%** | **86.1%** | **77.7%** | **56.0%** |
| Base + DPO (β=0.5) | 89.0% | 88.5% | 82.1% | 74.0% | 52.5% |

### LLM-as-Judge Head-to-Head (50 comparisons)

| Base Win | DPO Win | Tie |
|:--------:|:-------:|:---:|
| 10% | 0% | **90%** |

### Safety Evaluation (14 tests)

| Category | Base | DPO |
|----------|:----:|:---:|
| Prompt Injection (5) | 3/5 | 3/5 |
| Unauthorized Call (3) | 3/3 | 3/3 |
| Edge Case (4) | 1/4 | 1/4 |
| Ambiguity (2) | 2/2 | 2/2 |
| **Overall** | **64.3%** | **64.3%** |

### Ablation: DPO Beta Sweep

| β | JSON Valid | Exact Match |
|---|:---------:|:-----------:|
| 0.05 | 82% | 48% |
| 0.1 | 85% | 50% |
| **0.5** | **88%** | **53%** |

Higher β (stronger KL penalty) = better. Model benefits from staying closer to the reference.

### Ablation: Data Size Scaling

| Train Size | JSON Valid | Exact Match |
|:----------:|:---------:|:-----------:|
| **200** | **92%** | **56%** |
| 500 | 91% | 56% |
| 893 (full) | 83% | 48% |

Less data = better. More training data introduces more noise than signal.

## Key Findings

**DPO did not improve over the base Qwen2.5-3B-Instruct model.** Analysis:

1. **Base model is already strong** — 93% JSON Valid / 56% Exact Match zero-shot, leaving little room for improvement
2. **Synthetic preference data lacks signal** — programmatically injected errors (wrong values, malformed JSON) are too artificial; the base model already avoids these mistakes
3. **More data hurts** — data scaling shows diminishing returns, suggesting noise in the preference pairs outweighs learning signal
4. **Higher β helps** — the model performs best when constrained to stay near the reference, confirming DPO is pushing the model in an unhelpful direction

This is a common finding when applying DPO to already-capable instruction-tuned models with synthetic data ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Data Synthesis Pipeline                │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   DeepSeek    │───→│  Scenario    │───→│ Preference │  │
│  │   API         │    │  Generation  │    │ Pairs      │  │
│  │              │    │  (1000)      │    │ (893)      │  │
│  └──────────────┘    └──────────────┘    └─────┬─────┘  │
└─────────────────────────────────┬───────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────┐
│                   DPO Training (QLoRA)                  │
│                                                         │
│  Qwen2.5-3B-Instruct ──→ DPO (β=0.5)                   │
│                                                         │
│  Config: rank=16, 4-bit NF4, lr=5e-6, 1 epoch          │
│  Reference model = base (implicit via PEFT)             │
└─────────────────────────────────┬───────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────┐
│                   Evaluation Suite                       │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Standard  │  │ LLM-as-Judge │  │ Safety           │  │
│  │ FC Metrics│  │ Head-to-Head │  │ Prompt Injection │  │
│  │ (5 dims) │  │ Win Rate     │  │ Unauthorized     │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Data Generation

DeepSeek generates both chosen (correct) and rejected (with specific error types):
- `wrong_function` — called the wrong function
- `missing_required_arg` — missing required argument
- `hallucinated_arg_value` — argument value doesn't match query
- `extra_nonexistent_arg` — added non-existent argument
- `wrong_arg_type` — wrong argument type
- `malformed_json_value` — malformed JSON value (programmatic injection)

## Key Components

| Module | Description |
|--------|-------------|
| `src/data/generator.py` | DeepSeek scenario generation + preference pair synthesis |
| `src/data/formatter.py` | Convert to TRL DPOTrainer conversational format |
| `src/training/dpo_train.py` | QLoRA + DPO training with hyperparameter sweep support |
| `src/evaluation/metrics.py` | 5-dim FC metrics (JSON/Name/ArgN/ArgV/EM) |
| `src/evaluation/judge_eval.py` | Model A vs B head-to-head + position debiasing |
| `src/evaluation/safety.py` | Prompt injection / unauthorized call tests (14 cases) |

## Notebooks

| Notebook | Content | Requirements |
|----------|---------|-------------|
| `01_Generate_Preference_Data.ipynb` | Scenario generation + preference pairs | DeepSeek API key |
| `02_DPO_Training_and_Eval.ipynb` | DPO training + 3-way evaluation + ablation | Colab L4 GPU |
| `03_Upload_and_Demo.ipynb` | HuggingFace upload + Gradio Demo | Colab |

## Future Work

The most promising improvement is **on-policy data generation**:

1. Have the base Qwen model generate multiple responses per scenario (high temperature sampling)
2. Use DeepSeek as judge to rank responses
3. Best → chosen, worst → rejected

This creates preference pairs from the model's **actual mistakes** rather than artificial error injection, providing a stronger learning signal. The infrastructure (`src/data/sampler.py`, `src/data/judge.py`) already supports this.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set DeepSeek API key
export DEEPSEEK_API_KEY="your-key-here"

# 3. Run notebooks in order on Google Colab
#    01 → Generate data (API calls, no GPU needed)
#    02 → Train + Evaluate (needs L4 GPU)
#    03 → Upload + Demo
```

## Tech Stack

- **Training**: QLoRA 4-bit NF4 + TRL DPOTrainer + PEFT
- **Data**: DeepSeek API (synthesis + judging) + 25 functions × 5 domains
- **Evaluation**: 5-dim FC metrics + LLM-as-Judge win rate + Safety suite (14 tests)
- **Base model**: Qwen2.5-3B-Instruct
- **Platform**: Google Colab Pro (L4 GPU)

## HuggingFace

- Model: [Cheng-1/qwen2.5-3b-fc-dpo](https://huggingface.co/Cheng-1/qwen2.5-3b-fc-dpo)
- Dataset: [Cheng-1/fc-preference-data](https://huggingface.co/datasets/Cheng-1/fc-preference-data)

## Project Context

This is Phase 6 of a research series on small LLM capabilities:

| Phase | Project | Focus |
|-------|---------|-------|
| 1 | [small-llms-tool-use](https://github.com/XIECHENG6/small-llms-tool-use) | SFT for Function Calling |
| 2 | [agenttune](https://github.com/XIECHENG6/agenttune) | SFT for ReAct Agent |
| 3 | [smallrag](https://github.com/XIECHENG6/smallrag) | RAG Pipeline |
| 4 | [CodeAgent-MCP](https://github.com/XIECHENG6/CodeAgent-MCP) | Multi-Agent Code Gen |
| 5 | [kg-agent](https://github.com/XIECHENG6/kg-agent) | KG-Enhanced Agent |
| **6** | **smallllm-dpo** | **DPO Preference Alignment** |

**Progression**: SFT (P1-2) → Retrieval (P3) → System Design (P4-5) → **Alignment (P6)**

## License

Apache-2.0
