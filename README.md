<div align="center">

# SmallLLM-DPO

**DPO 对齐能否提升小模型的 Function Calling 能力？**

*剧透：合成偏好数据不行 —— 消融实验解释了原因。*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow.svg)](https://huggingface.co/Cheng-1/qwen2.5-3b-fc-dpo)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg)](https://huggingface.co/datasets/Cheng-1/fc-preference-data)

</div>

---

## TL;DR

使用 QLoRA + DPO (Direct Preference Optimization) 对 Qwen2.5-3B-Instruct 进行 Function Calling 偏好对齐。通过系统性消融实验发现，**合成偏好数据无法为已经很强的 Base 模型提供有效提升** —— 并通过 beta 和数据量两组实验精确解释了原因。

> **核心发现**：当 Base 模型已经很强（93% JSON 有效率）时，DPO 的瓶颈在于**数据质量**而非训练方法。程序化注入的错误太过人工 —— 模型本身不会犯这类错误，因此无法从中学到有用信号。

---

## 实验结果

<table>
<tr>
<td width="50%">

### 主模型对比
| 模型 | JSON Valid | Exact Match |
|:-----|:---------:|:-----------:|
| **Base (zero-shot)** | **93.0%** | **56.0%** |
| + DPO (ours) | 89.0% | 52.5% |

*200 条测试场景，5 维 FC 指标*

</td>
<td width="50%">

### LLM-as-Judge 对比（50 对）
| 判定结果 | 比例 |
|:---------|:----:|
| 平局 | **90%** |
| Base 胜 | 10% |
| DPO 胜 | 0% |

*DeepSeek 评审 + 位置去偏*

</td>
</tr>
</table>

### 消融实验

<table>
<tr>
<td width="50%">

#### Beta 消融（KL 惩罚强度）
| Beta | JSON Valid | Exact Match |
|:----:|:---------:|:-----------:|
| 0.05 | 82% | 48% |
| 0.1 | 85% | 50% |
| **0.5** | **88%** | **53%** |

Beta 越大越好 —— 模型需要**更强的约束来贴近参考分布**，说明 DPO 训练方向有害。

</td>
<td width="50%">

#### 数据量消融
| 训练量 | JSON Valid | Exact Match |
|:------:|:---------:|:-----------:|
| **200** | **92%** | **56%** |
| 500 | 91% | 56% |
| 893（全量） | 83% | 48% |

数据越少越好 —— 偏好对中**噪声大于信号**，训练越多退化越严重。

</td>
</tr>
</table>

### 安全性评估（14 项测试）

| 类别 | Base | DPO |
|:-----|:----:|:---:|
| Prompt Injection (5) | 3/5 | 3/5 |
| Unauthorized Call (3) | 3/3 | 3/3 |
| Edge Case (4) | 1/4 | 1/4 |
| Ambiguity (2) | 2/2 | 2/2 |
| **总计** | **64.3%** | **64.3%** |

---

## 为什么 DPO 没有效果（分析）

```
                    预期效果                      实际情况
                    ┌──────────────┐             ┌──────────────────────┐
偏好数据        ──> │ DPO 学会     │       ──>   │ Base 模型本来就      │
(chosen/rejected)   │ 避免这些错误 │             │ 不会犯这些错误       │
                    └──────────────┘             │ → 没有可学的信号     │
                                                └──────────────────────┘
```

三组证据相互印证：

1. **Beta 分析**：beta 越大（KL 约束越强）→ 性能越好。说明 DPO loss 在把模型往*错误方向*拉，KL 项是唯一的制动力。

2. **数据量分析**：数据越多性能越差（92% → 83%）。偏好对的噪声大于信号 —— 程序化注入的错误模式过于人工，与模型实际会犯的错误不匹配。

3. **Judge 评估**：90% 平局率说明 Base 和 DPO 在输出质量上几乎无差异，DPO 训练实际上是一个 no-op。

**根本原因**：rejected 样本通过 `_create_malformed_value()` 程序化注入错误（损坏的 JSON 值、错误的参数类型等）。这些人工错误太简单 —— Qwen2.5-3B-Instruct 已经在大量指令数据上训练过，不会犯这类错误。DPO 没有有用的信号可学。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      数据合成流水线                          │
│                                                             │
│  DeepSeek API ──→ 1000 个 FC 场景 ──→ 893 个偏好对          │
│                  （5 领域 × 25 函数）  （6 种错误类型）       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    DPO 训练（QLoRA）                         │
│                                                             │
│  Qwen2.5-3B-Instruct ──→ DPO（β=0.5, lr=5e-6, 1 epoch）    │
│  4-bit NF4 量化            LoRA rank=16                     │
│  参考模型 = Base（通过 PEFT 隐式实现）                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       评估体系                               │
│                                                             │
│  5 维 FC 指标  ·  LLM-as-Judge 对比  ·  14 项安全测试        │
│  Beta 消融     ·  数据量消融         ·  三方模型对比          │
└─────────────────────────────────────────────────────────────┘
```

## 偏好数据中的错误类型

| 类型 | 描述 | 数量 |
|:-----|:----|:----:|
| `wrong_function` | 调用了错误的函数 | ~165 |
| `missing_required_arg` | 缺少必需参数 | ~165 |
| `hallucinated_arg_value` | 参数值与 query 不符 | ~165 |
| `extra_nonexistent_arg` | 添加了不存在的参数 | ~165 |
| `wrong_arg_type` | 参数类型错误 | ~165 |
| `malformed_json_value` | JSON 值格式错误（程序化注入） | ~68 |

---

## 项目结构

```
smallllm-dpo/
├── config/settings.yaml              # 全局配置（beta, lr, LoRA 参数）
├── src/
│   ├── data/
│   │   ├── function_pool.py          # 25 函数 × 5 领域
│   │   ├── prompt_templates.py       # 系统 prompt 集中管理
│   │   ├── generator.py              # DeepSeek 场景生成 + 偏好对合成
│   │   ├── sampler.py                # On-policy 温度采样
│   │   ├── judge.py                  # DeepSeek-as-Judge 评分
│   │   └── formatter.py              # → TRL 对话格式
│   ├── training/
│   │   └── dpo_train.py              # DPOTrainer + 超参扫描
│   └── evaluation/
│       ├── metrics.py                # 5 维 FC 指标 + JSON 解析
│       ├── judge_eval.py             # Head-to-head 对比 + 位置去偏
│       └── safety.py                 # 14 项安全测试（注入/越权/边界）
├── demo/app.py                       # Gradio 模型对比 demo
├── notebooks/
│   ├── 01_Generate_Preference_Data   # 数据流水线（仅需 API，无需 GPU）
│   ├── 02_DPO_Training_and_Eval      # 训练 + 评估 + 消融（L4 GPU）
│   └── 03_Upload_and_Demo            # HF 上传 + Gradio demo
└── tests/                            # 25 个单元测试（pytest）
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 DeepSeek API key
export DEEPSEEK_API_KEY="your-key-here"

# 本地运行测试（无需 GPU）
pytest tests/ -v

# 在 Google Colab 上按顺序运行 Notebook：
#   01 → 生成数据（仅 API 调用）
#   02 → 训练 + 评估（需 L4 GPU）
#   03 → 上传 + Demo
```

## 未来改进方向：On-Policy 数据生成

目前最有希望让 DPO 真正生效的改进路径：

```
当前方案（off-policy）：                 改进方案（on-policy）：

DeepSeek 同时生成 ──→ 人工错误          Qwen 模型生成  ──→ 真实错误
chosen 和 rejected    模型根本           多个候选回答       模型实际
                      不会犯的                              会犯的
                                        DeepSeek 排序 ───┘
                                        最好 → chosen
                                        最差 → rejected
```

所需基础设施（`sampler.py` 和 `judge.py`）已经就绪，改动集中在数据生成流水线（Notebook 01）。

---

## HuggingFace

| 资源 | 链接 |
|:-----|:-----|
| DPO Adapter | [Cheng-1/qwen2.5-3b-fc-dpo](https://huggingface.co/Cheng-1/qwen2.5-3b-fc-dpo) |
| 偏好数据集 | [Cheng-1/fc-preference-data](https://huggingface.co/datasets/Cheng-1/fc-preference-data) |

## 项目背景

本项目是小模型能力研究系列的**第 6 阶段**：

| 阶段 | 项目 | 方向 | 核心成果 |
|:----:|:-----|:-----|:---------|
| 1 | [small-llms-tool-use](https://github.com/XIECHENG6/small-llms-tool-use) | SFT Function Calling | 87.8% Exact Match |
| 2 | [agenttune](https://github.com/XIECHENG6/agenttune) | SFT ReAct Agent | 多步推理 |
| 3 | [smallrag](https://github.com/XIECHENG6/smallrag) | RAG Pipeline | 端到端检索 |
| 4 | [CodeAgent-MCP](https://github.com/XIECHENG6/CodeAgent-MCP) | Multi-Agent Code Gen | MCP 集成 |
| 5 | [kg-agent](https://github.com/XIECHENG6/kg-agent) | KG-Enhanced Agent | 知识增强 |
| **6** | **smallllm-dpo** | **DPO 偏好对齐** | **负面结果 + 深度分析** |

**递进路线**：SFT (P1-2) → 检索增强 (P3) → 系统设计 (P4-5) → **对齐 (P6)**

---

<div align="center">

**License**: Apache-2.0

</div>
