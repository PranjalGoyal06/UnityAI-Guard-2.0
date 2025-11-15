#!/bin/bash
set -x

# --- 1. Variables ---
export MODEL_TO_FINETUNE="meta-llama/Llama-3.2-1B"
# --- UPDATED: Pointed to the new balanced training file ---
export DATASET_PATH="./train_balanced.jsonl"
export OUTPUT_SAVE_PATH="./checkpoint/category_head_adapter_BALANCED"

mkdir -p "$OUTPUT_SAVE_PATH"

export INSTRUCTION_TEMPLATE="You are a multilingual text-moderation model. \
You are given a harmful piece of text, for which you must output a probability distribution over the following safety categories (S1 … S17). Each category is defined as: \
S1: Violent Crimes \
S2: Non-Violent Crimes \
S3: Sex Crimes \
S4: Child Exploitation \
S5: Defamation \
S6: Specialized Advice \
S7: Privacy \
S8: Intellectual Property \
S9: Indiscriminate Weapons \
S10: Hate \
S11: Self-Harm \
S12: Sexual Content \
S13: Elections \
S14: Code Interpreter Abuse \
S15: Religious Intolerance \
S16: Gender-based Abuse \
S17: Trolling/Cyberbullying \
Text: {} \
Harm Distribution: "

# --- 2. Training Command ---
# This command trains on the balanced dataset from 'train_balanced.jsonl'
# and saves the new, clean adapter.
#
deepspeed --module openrlhf.cli.train_sft \
  --pretrain "$MODEL_TO_FINETUNE" \
  --dataset "$DATASET_PATH" \
  --input_key "text" \
  --output_key "safety_distribution" \
  --input_template "$INSTRUCTION_TEMPLATE" \
  --save_path "$OUTPUT_SAVE_PATH" \
  --save_hf_ckpt \
  --max_len 2048 \
  --max_epochs 1 \
  --zero_stage 2 \
  --bf16 \
  --learning_rate 5e-6 \
  --train_batch_size 8 \
  --micro_train_batch_size 1 \
  --logging_steps 1 \
  --save_steps -1 \
  --eval_steps -1 \
  --attn_implementation "flash_attention_2" \
  --adam_offload \
  --packing_samples \
  --load_in_4bit \
  --lora_rank 64 \
  --lora_alpha 64 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj