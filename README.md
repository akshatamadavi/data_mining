# 🦥 Unsloth AI – Modern LLM Fine-Tuning Experiments

**Author:** [Akshata Madavi]  
**Course:** Data Mining / CMPE 255  
**Demo Video:** 🎥 [https://youtu.be/pq_o0b3v3rk](https://youtu.be/pq_o0b3v3rk)

---

## 📘 Overview

This project explores **five modern fine-tuning and reinforcement learning methods** using the **[Unsloth.ai](https://unsloth.ai)** framework.  
Each notebook demonstrates a distinct training paradigm — from full finetuning to reinforcement learning for reasoning — on lightweight open models (e.g., **SmolLM**, **Gemma-3-1B**) that run efficiently on free Colab T4 GPUs.

---

## 🚀 Notebooks Summary

| # | Notebook | Technique | Core Idea | Output |
|:-:|:--|:--|:--|:--|
| **1️⃣** | `01_unslothai_full_finetuning.ipynb` | **Full Fine-Tuning (SFT)** | Train all model weights on chat/task data for highest quality | Full finetuned model |
| **2️⃣** | `02_unslothai_LoRA_parameter.ipynb` | **Parameter-Efficient Fine-Tuning (LoRA)** | Train small adapter layers instead of full weights; faster & lighter | LoRA adapters (`.safetensors`) |
| **3️⃣** | `03_rl_prefs.ipynb` | **Direct Preference Optimization (DPO)** | Align model responses toward preferred outputs (`prompt, chosen, rejected`) | LoRA + preference-aligned model |
| **4️⃣** | `04_grpo_reasoning.ipynb` | **Reinforcement Learning (GRPO)** | Improve reasoning (math/logic) using custom reward functions for format + correctness | LoRA adapters with reasoning skills |
| **5️⃣** | `05_continued_pretraining.ipynb` | **Continued Pretraining** | Teach model new domain/language from unlabeled text (Causal LM objective) | Domain-adapted model |

---

