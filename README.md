# ChartVerse: Scaling Chart Reasoning via Reliable Programmatic Synthesis from Scratch

<div align="center">
  <strong>Anonymous Authors</strong>
  <br>
  (Affiliations Placeholder)
</div>

<br>

<div align="center">
  <a href="https://arxiv.org/abs/YOUR_ARXIV_ID" target="_blank">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-ChartVerse-red?logo=arxiv" height="25" />
  </a>
  <a href="https://github.com/YOUR_USERNAME/ChartVerse" target="_blank">
      <img alt="Github Star" src="https://img.shields.io/github/stars/YOUR_USERNAME/ChartVerse?style=social" height="25" />
  </a>
  <a href="https://huggingface.co/collections/YOUR_USERNAME/chartverse-models" target="_blank">
      <img alt="HF Collections" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Collection-ChartVerse-ffc107?color=ffc107&logoColor=white" height="25" />
  </a>
  <a href="https://YOUR_USERNAME.github.io/ChartVerse/" target="_blank">
      <img alt="Homepage" src="https://img.shields.io/badge/🌐_Homepage-Project_Page-blue?color=blue&logoColor=white" height="25" />
  </a>
</div>

<br>
<div align="center">
  <img src="assets/complex_images.png" width="100%" alt="ChartVerse Pipeline">
</div>

## 🔥 News
* **[2026-01-07]** 🚀 **Project Launch:** The full **ChartVerse ecosystem** is now available, including the **paper**, **Complexity-Aware Chart Coder**, **models** (2B/4B/8B), and **datasets** (SFT-600K & RL-40K).

* **[2026-01-07]** 🏆 **SOTA Performance:** **ChartVerse-8B** achieves **64.1%** average score on 6 benchmarks, surpassing its teacher **Qwen3-VL-30B-Thinking** (62.9%) and approaching **Qwen3-32B** (67.0%). **ChartVerse-4B** (61.9%) significantly outperforms **Qwen3-VL-8B-Thinking** (60.0%).

## 📖 Abstract


Chart reasoning is a critical capability for Vision Language Models (VLMs). However, the development of open-source models is severely hindered by the lack of high-quality training data. Existing datasets suffer from a dual challenge: synthetic charts are often simplistic and repetitive, while the associated QA pairs are prone to hallucinations and lack the reasoning depth required for complex tasks.
To bridge this gap, we propose **ChartVerse**, a scalable framework designed to synthesize complex charts and reliable reasoning data from scratch. (1) To address the bottleneck of simple patterns, we first introduce **Rollout Posterior Entropy (RPE)**, a novel metric that quantifies chart complexity. Guided by RPE, we develop **complexity-aware chart coder** to autonomously synthesize diverse, high-complexity charts via executable programs. (2) To guarantee reasoning rigor, we develop **truth-anchored inverse QA synthesis**. Diverging from standard generation, we adopt an answer-first paradigm: we extract deterministic answers directly from the source code, generate questions conditional on these anchors, and enforce strict consistency verification. To further elevate difficulty and reasoning depth, we filter samples based on model fail-rate and distill high-quality Chain-of-Thought (CoT) reasoning.
We curate ChartVerse-SFT-600K and ChartVerse-RL-40K using Qwen3-VL-30B-A3B-Thinking as the teacher. Experimental results demonstrate that ChartVerse-8B achieves state-of-the-art performance, notably surpassing its teacher and rivaling the stronger Qwen3-32B-Thinking.

## ⚡ Method Highlights

<div align="center">
  <img src="assets/rpe_illustration.png" width="100%" alt="RPE Illustration">
</div>
<div align="center">
  <img src="assets/pipeline.png" width="100%" alt="RPE Illustration">
</div>

### 1. Metric: Rollout Posterior Entropy (RPE)

We quantify intrinsic complexity via **generative stability**.

**As shown above**, the calculation pipeline is:
* **🔄 Rollout:** Generate executable code multiple times using a VLM.
* **🧮 Entropy:** Compute spectral entropy from the CLIP embedding Gram matrix through singular values
* **🛡️ Filter:** Retain only challenging samples with high entropy.

### 2. Complexity-Aware Chart Coder
An autonomous coder trained to synthesize diverse charts from scratch.
* **❄️ Cold Start:** Inferred code from high-RPE real-world charts using **Claude-4-Sonnet**.
* **🔄 Iterative Self-Enhancement:**
    1. **Generate:** Previous coder generates candidates.
    2. **Filter:** Keep samples with **High RPE** and **Low Similarity**.
    3. **Retrain:** Update the coder with the boosted dataset to train the next round.

### 3. Truth-Anchored Inverse QA Pipeline
We adopt an **Answer-First Paradigm ($A \rightarrow Q$)** to eliminate hallucinations.

#### Phase I: Inverse Logic Construction (Text-Only)
Using **Qwen3-30B-A3B-Thinking**, we synthesize the logic chain based on code ground truth:
1.  **Script ($S$):** Code $\rightarrow$ Python Script $\rightarrow$ Deterministic Answer $A_{py}$.
2.  **Reverse ($Q$):** Synthesize Question $Q$ strictly leading to Script $S$.
3.  **Verify:** Retain pairs where inferred answer $\hat{A} == A_{py}$.

#### Phase II: Visual Distillation
* **CoT Distillation:** Generate reasoning traces using **Qwen3-VL-30B-A3B-Thinking**.
* **Fail-Rate Filter:** Retain "hard but solvable" samples ($0 < r(Q) < 1$) to ensure difficulty.

## 💾 Datasets

### 📥 Download Datasets

Our synthesis pipeline produced two high-quality datasets available on HuggingFace:

| Dataset Name | Scale | Composition | Download |
| :--- | :---: | :--- | :---: |
| **ChartVerse-SFT-600K** | **600K** | High-Complexity Charts + CoT Reasoning | [🤗 Download](https://huggingface.co/datasets/ChartVerse-SFT) |
| **ChartVerse-RL-40K** | **40K** | Hard Reasoning Samples | [🤗 Download](https://huggingface.co/datasets/ChartVerse-RL) |

### 🆚 Comparison with Existing Datasets

We compare **ChartVerse-SFT-600K** with mainstream chart reasoning datasets. As shown below, ChartVerse achieves superior diversity and complexity while ensuring reasoning rigor.

<div align="center">
  <img src="assets/chart_cmp.png" width="100%" alt="Dataset Comparison">
</div>

**Key Advantages:**
* **Higher Complexity (RPE):** Our dataset achieves the highest **Rollout Posterior Entropy (0.44)** compared to baselines (e.g., CoSyn 0.35, ChartQA 0.26), proving that our charts possess greater intrinsic structural difficulty.
* **Superior Diversity:** ChartVerse records the highest **Color Entropy (3.17)** and **Semantic Embedding Spread (0.51)**, indicating a much broader coverage of visual styles and chart topics than previous synthetic engines.
* **Rigorous Reliability:** Unlike datasets reliant on raw LLM generation (which suffer from hallucinations), ChartVerse guarantees **Answer Accuracy** through our *Truth-Anchored Inverse QA* pipeline, backed by **3.9B** tokens of high-quality Chain-of-Thought reasoning data.

## 🦁 Model Zoo & Performance

We release the **Complexity-Aware Chart Coder** and the full **ChartVerse Reasoning Model Series** (including SFT and RL aligned versions).

### 📥 Download Models

| Model Name | Type | Base Model | Param | Avg Score | HuggingFace Link |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Complexity-Aware Chart Coder** | Coder | Qwen2.5-Coder | 7B | - | [🤗 Download](https://huggingface.co/YOUR_USERNAME/Complexity-Aware-Chart-Coder) |
| **ChartVerse-2B-SFT** | VLM | Qwen3-VL-2B-Instruct | 2B | 49.8 | [🤗 Download](https://huggingface.co/YOUR_USERNAME/ChartVerse-2B-SFT) |
| **ChartVerse-2B-RL** | VLM | ChartVerse-2B-SFT | 2B | **54.3** | [🤗 Download](https://huggingface.co/YOUR_USERNAME/ChartVerse-2B-RL) |
| **ChartVerse-4B-SFT** | VLM | Qwen3-VL-4B-Instruct | 4B | 59.7 | [🤗 Download](https://huggingface.co/YOUR_USERNAME/ChartVerse-4B-SFT) |
| **ChartVerse-4B-RL** | VLM | ChartVerse-4B-SFT | 4B | **61.9** | [🤗 Download](https://huggingface.co/YOUR_USERNAME/ChartVerse-4B-RL) |
| **ChartVerse-8B-SFT** | VLM | Qwen3-VL-8B-Instruct | 8B | 62.5 | [🤗 Download](https://huggingface.co/YOUR_USERNAME/ChartVerse-8B-SFT) |
| **ChartVerse-8B-RL** | VLM | ChartVerse-8B-SFT | 8B | **64.1** | [🤗 Download](https://huggingface.co/YOUR_USERNAME/ChartVerse-8B-RL) |

### 🏆 Detailed Performance

We evaluate ChartVerse on 6 benchmarks requiring complex chart understanding and reasoning. The results demonstrate that **data quality triumphs over model scale**.

<div align="center">
  <img src="assets/overall_result.png" width="100%" alt="Performance Comparison">
</div>

**Key Findings:**
* **Small > Large:** Despite having half the parameters, **ChartVerse-4B** significantly outperforms **Qwen3-VL-8B-Thinking**.
* **Student > Teacher:** **ChartVerse-8B** breaks the distillation ceiling, surpassing its teacher model, **Qwen3-VL-30B-Thinking.
* **Top-Tier Performance:** Our 8B model rivals the performance of the much larger **Qwen3-VL-32B-Thinking**, proving the effectiveness of our pipeline.

## 🛠️ Quick Start

Since ChartVerse models are initialized from **Qwen3-VL**, usage is straightforward with the `transformers` library.

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import torch

# 1. Load Model
model_path = "YOUR_USERNAME/ChartVerse-8B-RL"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# 2. Prepare Input
image_path = "assets/demo_chart.png"
query = "Which region demonstrates the greatest proportional variation in annual revenue compared to its typical revenue level?
"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": query},
        ],
    }
]

# 3. Inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=16384)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])