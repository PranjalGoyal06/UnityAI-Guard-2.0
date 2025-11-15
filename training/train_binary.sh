#!/bin/bash
set -x

# --- 1. Variables ---
export MODEL_TO_FINETUNE="meta-llama/Llama-3.2-1B"
# --- CHANGED: Pointed to the new binary dataset ---
export DATASET_PATH="./binary_train_1.jsonl"
# --- CHANGED: Saving to a new directory for the new binary model ---
export OUTPUT_SAVE_PATH="./checkpoint/binary_head_adapter_1"

mkdir -p "$OUTPUT_SAVE_PATH"

# --- CHANGED: New template for binary SAFE/UNSAFE classification ---
export INSTRUCTION_TEMPLATE="You are a multilingual text moderation model.
Classify each input as either HARMFUL or SAFE.

HARMFUL means the text contains or promotes any form of violence, crime, sexual or child exploitation, hate, defamation, self-harm, harassment, privacy violation, or other abusive or unsafe behavior.
All other text is SAFE.

Always output only one word:
UNSAFE
or
SAFE

Text: {}
Label: "

# --- 2. Training Command ---
#
# 1. Pointed --dataset to $DATASET_PATH
# 2. Pointed --save_path to $OUTPUT_SAVE_PATH
# 3. --- CHANGED --output_key to "label" ---
# 4. --- REMOVED --max_samples ---
#
deepspeed --module openrlhf.cli.train_sft \
  --pretrain "$MODEL_TO_FINETUNE" \
  --dataset "$DATASET_PATH" \
  --input_key "text" \
  --output_key "label" \
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