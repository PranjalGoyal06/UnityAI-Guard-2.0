import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from scipy.special import kl_div
from tqdm import tqdm
import sys
import warnings
import argparse
from collections import defaultdict, Counter  # --- ADDED: For language-specific results ---
from langdetect import detect, LangDetectException # --- ADDED: For language detection ---

# --- 1. Argument Parser ---
# (This section is unchanged)
parser = argparse.ArgumentParser(description="Evaluate a multi-label classification model.")
parser.add_argument(
    "--num_samples",
    type=int,
    default=None,  # Default is None, which means we process all samples
    help="Number of samples to evaluate from the test file. (Default: All)"
)
args = parser.parse_args()

# --- 2. Configuration ---
# (This section is unchanged)
MODEL_TO_FINETUNE = "meta-llama/Llama-3.2-1B"
ADAPTER_PATH = "./checkpoint/category_head_adapter_BALANCED"
DATA_FILE_PATH = "./clean_test_set.jsonl"
INSTRUCTION_TEMPLATE = """You are a multilingual text-moderation model. \
You are given a harmful piece of text, for which you must output a probability distribution over the following safety categories (S1 … S17). Each category is defined as: \
S1: Violent Crimes \
S2: Non-Violinent Crimes \
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
Harm Distribution: """
CATEGORY_NAMES = [f"S{i+1}" for i in range(17)]
epsilon = 1e-9

# --- 3. Helper Function to Parse Distributions ---
# (This function is unchanged)
def parse_distribution(dist_string: str) -> np.ndarray | None:
    try:
        matches = re.findall(r"S\d+:\s*([\d\.]+)", dist_string)
        if len(matches) == 17:
            return np.array([float(m) for m in matches])
        matches = re.findall(r"([\d\.]+)", dist_string)
        if len(matches) == 17:
            return np.array([float(m) for m in matches])
        return None
    except Exception:
        return None

# --- 4. Load Model and Tokenizer (Same as inference) ---
# (This section is unchanged)
print(f"Loading base model: {MODEL_TO_FINETUNE}")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_TO_FINETUNE,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_FINETUNE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model and adapter loaded successfully.")
eos_token_ids = [tokenizer.eos_token_id, 128009]

# --- 5. Load Evaluation Data ---
# (This section is unchanged)
print(f"Loading evaluation data from {DATA_FILE_PATH}...")
eval_data = []
try:
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading clean test set"):
            try:
                eval_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line in test set", file=sys.stderr)
            if args.num_samples and len(eval_data) >= args.num_samples:
                print(f"\nLimiting evaluation to first {args.num_samples} samples as requested.")
                break
except FileNotFoundError:
    print(f"ERROR: Clean test file not found: {DATA_FILE_PATH}", file=sys.stderr)
    print("Please run 'create_clean_test_set.py' first.", file=sys.stderr)
    sys.exit(1)
print(f"Loaded {len(eval_data)} samples for evaluation from {DATA_FILE_PATH}.")

# --- 6. Run Evaluation Loop ---
# --- ADDED: Dictionaries to store language-specific results ---
lang_y_true = defaultdict(list)
lang_y_pred = defaultdict(list)
lang_kl_scores = defaultdict(list)
lang_top1_recalls = defaultdict(list)
lang_top3_recalls = defaultdict(list)
lang_ap_scores = defaultdict(list)
lang_format_errors = Counter()
lang_sample_count = Counter()
# --- END ADDED ---

# Lists for overall results
y_true_top1 = []
y_pred_top1 = []
all_kl_scores = []
all_top1_recalls = []
all_top3_recalls = []
all_ap_scores = []
format_errors = 0

print("Running evaluation...")
warnings.filterwarnings('ignore', category=UserWarning)

for sample in tqdm(eval_data, desc="Evaluating samples"):
    input_text = sample.get("text")
    truth_string = sample.get("safety_distribution")
    
    if not input_text or not truth_string:
        continue
        
    # --- ADDED: Language Detection ---
    try:
        # Detect language (e.g., 'en', 'or', 'bn')
        lang = detect(input_text)
    except LangDetectException:
        lang = 'unknown' # Handle short or ambiguous text
    lang_sample_count[lang] += 1
    # --- END ADDED ---

    truth_vector = parse_distribution(truth_string)
    if truth_vector is None:
        print(f"Skipping sample, could not parse ground truth: {truth_string}", file=sys.stderr)
        continue
        
    prompt = INSTRUCTION_TEMPLATE.format(input_text)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=eos_token_ids,
            # --- FIX: This silences the pad_token_id warning ---
            pad_token_id=tokenizer.eos_token_id
        )
        
    prediction_string = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    prediction_vector = parse_distribution(prediction_string)
    
    if prediction_vector is None:
        format_errors += 1
        lang_format_errors[lang] += 1 # --- ADDED ---
        continue
        
    # --- Store Overall Results (Unchanged) ---
    y_true_top1.append(np.argmax(truth_vector))
    y_pred_top1.append(np.argmax(prediction_vector))
    truth_norm = (truth_vector + epsilon) / np.sum(truth_vector + epsilon)
    pred_norm = (prediction_vector + epsilon) / np.sum(prediction_vector + epsilon)
    kl_score = np.sum(kl_div(truth_norm, pred_norm))
    all_kl_scores.append(kl_score)
    true_positive_indices = set(np.where(truth_vector > 0)[0])
    if len(true_positive_indices) == 0:
        # --- ADDED: Store lang-specific results even if no positive labels for KL/AP ---
        lang_y_true[lang].append(np.argmax(truth_vector))
        lang_y_pred[lang].append(np.argmax(prediction_vector))
        lang_kl_scores[lang].append(kl_score)
        # We can't calculate AP/Recall, so we'll skip them for this sample
        continue 
    pred_top1_indices = set(np.argsort(prediction_vector)[-1:])
    pred_top3_indices = set(np.argsort(prediction_vector)[-3:])
    hits_top1 = len(true_positive_indices.intersection(pred_top1_indices))
    hits_top3 = len(true_positive_indices.intersection(pred_top3_indices))
    top1_recall = hits_top1 / len(true_positive_indices)
    top3_recall = hits_top3 / len(true_positive_indices)
    all_top1_recalls.append(top1_recall)
    all_top3_recalls.append(top3_recall)
    y_true_binary = (truth_vector > 0).astype(int)
    ap_score = average_precision_score(y_true_binary, prediction_vector)
    all_ap_scores.append(ap_score)
    
    # --- ADDED: Store Language-Specific Results ---
    lang_y_true[lang].append(np.argmax(truth_vector))
    lang_y_pred[lang].append(np.argmax(prediction_vector))
    lang_kl_scores[lang].append(kl_score)
    lang_top1_recalls[lang].append(top1_recall)
    lang_top3_recalls[lang].append(top3_recall)
    lang_ap_scores[lang].append(ap_score)
    # --- END ADDED ---

# --- 7. Print Results ---
# (This section is unchanged, it prints all metrics)
print("\n" + "="*70)
print("--- Overall Evaluation Report ---")
print("="*70)
if not y_true_top1:
    print("Evaluation failed. No valid samples were processed.")
    sys.exit()
print(f"Total samples processed: {len(eval_data)}")
print(f"Valid predictions (parseable): {len(y_true_top1)}")
print(f"Model format errors (unparseable): {format_errors}")
print(f"Success Rate: {len(y_true_top1) / len(eval_data) * 100 :.2f}%")
# --- Report Section 1: Distribution & Ranking Metrics ---
print("\n" + "="*70)
print("--- Overall Distribution & Ranking Metrics (Multi-Label) ---")
print("="*70)
if all_kl_scores:
    print(f"Mean KL-Divergence:  {np.mean(all_kl_scores):.4f}  (Lower is better)")
    print(f"Mean Average Precision (mAP): {np.mean(all_ap_scores):.4f}  (Higher is better)")
    print(f"Mean Top-1 Recall:     {np.mean(all_top1_recalls):.4f}  (Higher is better)")
    print(f"Mean Top-3 Recall:     {np.mean(all_top3_recalls):.4f}  (Higher is better)")
else:
    print("No valid samples were found for distribution metrics.")
# --- Report Section 2: Top-1 Classification Metrics ---
print("\n" + "="*70)
print("--- Overall Top-1 Classification Metrics (Multi-Class) ---")
print("="*70)
print("\n--- Classification Report (Top-1) ---")
print(classification_report(y_true_top1, y_pred_top1, target_names=CATEGORY_NAMES, labels=range(17), zero_division=0))
print("\n--- Confusion Matrix (Top-1) ---")
print("(Rows = True Labels, Columns = Predicted Labels)")
cm = confusion_matrix(y_true_top1, y_pred_top1, labels=range(17))
print(f"     { ' '.join([f'{i:3}' for i in range(17)]) }")
print("    " + "-"*53)
for i, row in enumerate(cm):
    print(f"S{i+1:<2} | { ' '.join([f'{x:3}' for x in row]) }")
    
# --- ADDED: Language-Specific Reports ---
print("\n" + "="*70)
print("--- Language-Specific Reports ---")
print("="*70)

detected_languages = sorted(lang_sample_count.keys())
if not detected_languages:
    print("No languages were detected.")
    sys.exit()

for lang in detected_languages:
    print(f"\n--- Language: '{lang}' ---")
    total_lang_samples = lang_sample_count[lang]
    valid_lang_samples = total_lang_samples - lang_format_errors[lang]
    
    print(f"Total Samples:     {total_lang_samples}")
    if valid_lang_samples == 0:
        print("No valid predictions for this language.")
        continue
        
    print(f"Valid Predictions: {valid_lang_samples}")
    print(f"Format Errors:     {lang_format_errors[lang]}")
    print(f"Success Rate:      {(valid_lang_samples / total_lang_samples * 100):.2f}%")
    
    # --- Language Distribution Metrics ---
    if lang_kl_scores[lang]:
        print(f"\nDistribution Metrics (for '{lang}'):")
        print(f"  Mean KL-Divergence:  {np.mean(lang_kl_scores[lang]):.4f}")
        print(f"  Mean Average Precision (mAP): {np.mean(lang_ap_scores[lang]):.4f}")
        print(f"  Mean Top-1 Recall:     {np.mean(lang_top1_recalls[lang]):.4f}")
        print(f"  Mean Top-3 Recall:     {np.mean(lang_top3_recalls[lang]):.4f}")
    
    # --- Language Classification Report (Top-1) ---
    if lang_y_true[lang]:
        print(f"\nClassification Report (Top-1 for '{lang}'):")
        # We must filter out labels that don't appear in this language's subset
        present_labels = sorted(list(set(lang_y_true[lang]) | set(lang_y_pred[lang])))
        present_target_names = [CATEGORY_NAMES[i] for i in present_labels]
        
        print(classification_report(
            lang_y_true[lang], 
            lang_y_pred[lang], 
            labels=present_labels, 
            target_names=present_target_names, 
            zero_division=0
        ))
    print("-" * (len(lang) + 18))
# --- END ADDED ---