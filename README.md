# Multilingual Toxicity & Safety Classification — UnityAI-Guard 2.0


This repository contains the complete implementation for **Assignment-3 of CS613 (NLP)**, IIT Gandhinagar.  
Our objective is to extend the **UnityAI-Guard** multilingual toxicity detection framework by:

- Adding 4 new Indic languages (Bn, Od, Ml, Kn)  
- Introducing a **binary toxicity head** and a **fine-grained 17-category safety distribution head**  
- Training LoRA-based lightweight adapters on Llama-3.2-1B  
- Building a fully functional **web interface** showcasing inference and results  
- Documenting experiments, ablations, and evaluation

The project integrates dataset curation, supervised finetuning, evaluation, and deployment into one cohesive pipeline.

---

## Quick Links  
- [Dataset on HuggingFace](https://huggingface.co/datasets/advaitIITGN/unity_AI_guard_v2_dataset)
- [Website](https://saikarna913.github.io/AI-Guard/)

---

## Features

- **Multilingual support:** 10 low-resource Indic languages  
- **Two-head architecture:**  
  - Binary SAFE / HARMFUL classifier  
  - 17-category safety distribution generator  
- **Efficient training:** 4-bit QLoRA adapters  
- **Web portal:**  
  - Real-time inference  
  - Language selection  
  - Distribution visualisation  
- **Evaluation scripts:** zero-shot, few-shot, SFT, adapter comparison


---

## Summary of Approach

### 1. Dataset Curation
- Aggregated multilingual raw-text datasets from public Indic sources  
- Applied a **regex-based safety category extraction** pipeline  
- Propagated English labels to translated samples where applicable  
- Generated **soft label distributions** proportional to detected category frequencies  

Only small dataset samples are included in this repo; full datasets are referenced externally.

---

### 2. Model Architecture
We use **Llama-3.2-1B** as the base model and attach two LoRA heads:

1. **Binary Classification Head**  
   - Trained on 14k curated samples  
   - Outputs SAFE / HARMFUL  

2. **Distribution Head**  
   - Trained on ~68k samples  
   - Outputs probability distribution over 17 safety categories  

Both are trained via text-to-text generation.

---

### 3. Training
- Parameter-efficient finetuning using **4-bit QLoRA**  
- Separate adapters for binary and distribution tasks  
- Training commands located in `training/*.sh`  

---

### 4. Evaluation
We evaluate:

- Zero-shot baselines (Llama-3.2-1B-Instruct, Aya-8B)  
- Few-shot prompt-based evaluations  
- Post-SFT adapter performance  

Metrics include:

- F1-score  
- Accuracy  
- KL-divergence  
- mAP  
- Top-k recall  

---



