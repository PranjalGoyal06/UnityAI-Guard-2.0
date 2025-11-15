import json
import re
import numpy as np
from collections import Counter
from tqdm import tqdm
import sys

# --- 1. Configuration ---
DATA_FILE_PATH = "./category_data_balanced.jsonl"
SAMPLES_TO_ANALYZE = 64000
CATEGORY_NAMES = [f"S{i+1}" for i in range(17)]

# --- 2. Helper Function to Parse Distributions ---

def parse_distribution(dist_string: str) -> list[float] | None:
    """
    Parses the model's output string (e.g., "{S1: 0.4, S2: 0.4...}")
    into a list of 17 floats.
    """
    try:
        # This regex is specific to the format: {S1: 0.4, S2: 0.4, ...}
        # It finds all floating-point/integer numbers that follow "S<number>:"
        matches = re.findall(r"S\d+:\s*([\d\.]+)", dist_string)
        
        if len(matches) == 17:
            return [float(m) for m in matches]
        
        # Fallback for other formats (less likely for the data file)
        matches = re.findall(r"([\d\.]+)", dist_string)
        if len(matches) == 17:
            return [float(m) for m in matches]
            
        # If we're here, the format is wrong
        return None
        
    except Exception:
        return None

# --- 3. Main Analysis Loop ---

print(f"Starting analysis of the first {SAMPLES_TO_ANALYZE} samples from {DATA_FILE_PATH}...")

# This Counter will store the "winner" for each sample
# e.g., {'S10': 30000, 'S1': 5000, ...}
category_counts = Counter()
malformed_lines = 0

try:
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        
        # Use tqdm for a progress bar
        for i, line in enumerate(tqdm(f, total=SAMPLES_TO_ANALYZE, desc="Analyzing data")):
            
            # Stop after we've analyzed the target number of samples
            if i >= SAMPLES_TO_ANALYZE:
                break
            
            try:
                data = json.loads(line)
                dist_string = data.get("safety_distribution")

                if not dist_string:
                    malformed_lines += 1
                    continue
                
                # Parse the {S1: 0.4, ...} string
                vector = parse_distribution(dist_string)
                
                if vector is None:
                    malformed_lines += 1
                    continue
                
                # Find the index of the highest probability
                # e.g., if S10 has the max value, winner_index will be 9
                winner_index = np.argmax(vector)
                
                # Get the category name (e.g., "S10")
                winner_category = CATEGORY_NAMES[winner_index]
                
                # Add to our count
                category_counts[winner_category] += 1
                
            except Exception:
                # This catches JSON errors or other unexpected issues
                malformed_lines += 1

except FileNotFoundError:
    print(f"ERROR: Could not find the file: {DATA_FILE_PATH}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)

# --- 4. Print The Report ---

total_valid_samples = SAMPLES_TO_ANALYZE - malformed_lines

print("\n" + "="*50)
print("--- Training Data Analysis Report ---")
print("="*50)

print(f"Total lines processed: {SAMPLES_TO_ANALYZE}")
print(f"Valid samples (parseable): {total_valid_samples}")
print(f"Malformed lines (skipped): {malformed_lines}")
print("\n--- 'Winning' Category Distribution ---")

if total_valid_samples == 0:
    print("No valid data was found.")
    sys.exit()

# Sort the results from most common to least common
sorted_counts = category_counts.most_common()

for category, count in sorted_counts:
    percentage = (count / total_valid_samples) * 100
    print(f"{category:<5}: {count:8d} samples ({percentage:5.2f}%)")

print("="*50)
print("\n--- Conclusion ---")
s10_count = category_counts.get('S10', 0)
s10_percentage = (s10_count / total_valid_samples) * 100

if s10_percentage > 50:
    print(f"HYPOTHESIS CONFIRMED: The dataset is SEVERELY imbalanced.")
    print(f"S10 (Hate) is the 'winning' category in {s10_percentage:.2f}% of your training data.")
    print("This explains why the model collapsed and only predicts 'S10'.")
elif s10_percentage > 25:
    print(f"HYPOTHESIS LIKELY: The dataset is imbalanced.")
    print(f"S10 (Hate) is the 'winning' category in {s10_percentage:.2f}% of your training data.")
    print("This is likely the cause of the model collapse.")
else:
    print("HYPOTHESIS DENIED: The dataset does not appear to be imbalanced towards S10.")
    print("The model collapse might be due to a different issue (e.g., learning rate, data quality).")
