#!/usr/bin/env python3
"""
Evaluate a binary SAFE / UNSAFE LoRA classifier saved in ADAPTER_PATH on a jsonl file.

Usage:
  python eval_binary.py --file binary_test_1.jsonl --n 500 --batch-size 16 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# metrics
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    balanced_accuracy_score
)

INSTRUCTION_TEMPLATE = """You are a multilingual text moderation model.
Classify each input as either HARMFUL or SAFE.

HARMFUL means the text contains or promotes any form of violence, crime, sexual or child exploitation, hate, defamation, self-harm, harassment, privacy violation, or other abusive or unsafe behavior.
All other text is SAFE.

Always output only one word:
UNSAFE
or
SAFE

Text: {}
Label:"""


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def prepare_prompts(items):
    return [INSTRUCTION_TEMPLATE.format(it["text"]) for it in items]


def normalize_pred(text):
    if text is None:
        return "UNKNOWN"
    t = text.strip().upper()
    if "UNSAFE" in t:
        return "UNSAFE"
    if "SAFE" in t:
        return "SAFE"
    if t in ("0", "HARMFUL", "DANGEROUS", "ABUSIVE"):
        return "UNSAFE"
    if t in ("1", "BENIGN", "OK"):
        return "SAFE"
    return "UNKNOWN"


def batch_generate(model, tokenizer, prompts, device, max_new_tokens=3, batch_size=8, eos_ids=None, pbar=None):
    all_preds = []

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id

    for i in range(0, len(prompts), batch_size):
        if pbar:
            pbar.update(min(batch_size, len(prompts) - i))

        batch_prompts = prompts[i:i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        input_lens = (input_ids != pad_id).sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
            )

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs

        for b in range(sequences.shape[0]):
            seq = sequences[b]
            start = int(input_lens[b].item())
            gen_tokens = seq[start:]
            pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            all_preds.append(pred_text)

    return all_preds


def main(args):
    data = read_jsonl(args.file)
    if len(data) == 0:
        raise SystemExit("No data found in file.")

    # sample N
    if args.n is None or args.n >= len(data):
        sampled = data
    else:
        random.seed(args.seed)
        sampled = random.sample(data, args.n)

    texts = [d["text"] for d in sampled]
    golds = [d["label"].strip().upper() for d in sampled]

    print("Loading model...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    try:
        model.to(device)
    except:
        pass

    eos_ids = [tokenizer.eos_token_id, 128009]

    print("Preparing prompts...")
    prompts = prepare_prompts(sampled)

    print(f"\nRunning inference on {len(prompts)} examples (batch_size={args.batch_size})...\n")
    pbar = tqdm(total=len(prompts), desc="Evaluating", unit="samples")

    raw_preds = batch_generate(
        model, tokenizer, prompts, device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        eos_ids=eos_ids,
        pbar=pbar
    )
    pbar.close()

    preds = [normalize_pred(p) for p in raw_preds]

    y_true = [g if g in ("SAFE", "UNSAFE") else "UNKNOWN" for g in golds]
    y_pred = preds

    df = pd.DataFrame({
        "text": texts,
        "gold": y_true,
        "pred_raw": raw_preds,
        "pred": y_pred
    })
    df.to_csv("predictions.csv", index=False, encoding="utf-8")

    valid = [i for i, (gt, pr) in enumerate(zip(y_true, y_pred)) if gt in ("SAFE", "UNSAFE")]
    if not valid:
        print("No valid labeled samples.")
        return

    y_true_valid = [y_true[i] for i in valid]
    y_pred_valid = [y_pred[i] for i in valid]

    acc = accuracy_score(y_true_valid, y_pred_valid)
    bal_acc = balanced_accuracy_score(y_true_valid, y_pred_valid)
    kappa = cohen_kappa_score(y_true_valid, y_pred_valid)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, labels=["SAFE", "UNSAFE"], zero_division=0
    )
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=["SAFE", "UNSAFE"])

    print("\n===== SUMMARY =====")
    print(f"Evaluated {len(y_true_valid)} valid samples out of {len(prompts)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}\n")

    print("Per-class metrics (SAFE, UNSAFE):")
    for lbl, p, r, f, s in zip(["SAFE", "UNSAFE"], precisions, recalls, f1s, supports):
        print(f"  {lbl:6s} precision={p:.4f} recall={r:.4f} f1={f:.4f} support={s}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true_valid, y_pred_valid, labels=["SAFE", "UNSAFE"], zero_division=0))

    print("\nSaved predictions to predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="./binary_test_1.jsonl")
    parser.add_argument("--data", type=str, dest="file_alias")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--adapter", type=str, default="./checkpoint/binary_head_adapter_1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if getattr(args, "file_alias", None):
        args.file = args.file_alias

    main(args)
