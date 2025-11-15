import json
import re
import numpy as np
import random
from tqdm import tqdm
import sys
from collections import defaultdict

# --- 1. Configuration ---
ORIGINAL_DATA_FILE = "./category_data.jsonl"
BALANCED_DATA_FILE = "./category_data_balanced.jsonl"

# --- Balancing Target ---
# We will over/undersample to get this many samples for ALL 17 categories.
# 4,000 is a good starting point based on your data.
TARGET_SAMPLES_PER_CATEGORY = 4000

CATEGORY_NAMES = [f"S{i+1}" for i in range(17)]

# --- 2. Helper Function to Parse Distributions ---
def parse_distribution(dist_string: str) -> list[float] | None:
    """
    Parses the distribution string (e.g., "{S1: 0.4, S2: 0.4...}")
    into a list of 17 floats.
    """
    try:
        matches = re.findall(r"S\d+:\s*([\d\.]+)", dist_string)
        if len(matches) == 17:
            return [float(m) for m in matches]
        
        # Fallback for other formats
        matches = re.findall(r"([\d\.]+)", dist_string)
        if len(matches) == 17:
            return [float(m) for m in matches]
            
        return None
    except Exception:
        return None

# --- 3. Main Balancing Logic ---

def create_balanced_dataset():
    
    # --- First Pass: Read and Sort All Samples ---
    # We will store all samples in memory, sorted by their winning category.
    # e.g., category_pools['S1'] = [ {..json1..}, {..json5..}, ... ]
    category_pools = defaultdict(list)
    malformed_lines = 0

    print(f"--- Pass 1: Reading and sorting all samples from {ORIGINAL_DATA_FILE} ---")
    print("This may take a few minutes...")
    
    try:
        with open(ORIGINAL_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading file"):
                try:
                    data = json.loads(line)
                    dist_string = data.get("safety_distribution")

                    if not dist_string:
                        malformed_lines += 1
                        continue
                    
                    vector = parse_distribution(dist_string)
                    if vector is None:
                        malformed_lines += 1
                        continue
                    
                    # Find the winning category for this sample
                    winner_index = np.argmax(vector)
                    winner_category = CATEGORY_NAMES[winner_index]
                    
                    # Add the original JSON data (as a dict) to the correct pool
                    category_pools[winner_category].append(data)
                    
                except Exception:
                    malformed_lines += 1

    except FileNotFoundError:
        print(f"ERROR: File not found: {ORIGINAL_DATA_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Pass 1 Complete ---")
    print(f"Total malformed lines skipped: {malformed_lines}")
    print("Original Distribution:")
    for category in CATEGORY_NAMES:
        count = len(category_pools[category])
        print(f"{category:<5}: {count:8d} samples")

    # --- Second Pass: Balance by Over/Undersampling ---
    print("\n--- Pass 2: Balancing dataset ---")
    print(f"Target is {TARGET_SAMPLES_PER_CATEGORY} samples per category.")
    
    balanced_dataset = []
    
    for category in tqdm(CATEGORY_NAMES, desc="Balancing categories"):
        pool = category_pools[category]
        pool_size = len(pool)
        
        if pool_size == 0:
            print(f"Warning: No samples found for category {category}. Skipping.", file=sys.stderr)
            continue
        
        if pool_size > TARGET_SAMPLES_PER_CATEGORY:
            # --- Undersampling ---
            # Randomly select N samples from the pool
            balanced_samples = random.sample(pool, TARGET_SAMPLES_PER_CATEGORY)
            balanced_dataset.extend(balanced_samples)
            
        elif pool_size < TARGET_SAMPLES_PER_CATEGORY:
            # --- Oversampling ---
            # Select N samples *with replacement* (allows duplicates)
            balanced_samples = random.choices(pool, k=TARGET_SAMPLES_PER_CATEGORY)
            balanced_dataset.extend(balanced_samples)
            
        else:
            # --- Just right ---
            # Add all samples
            balanced_dataset.extend(pool)

    print(f"--- Pass 2 Complete ---")
    print(f"Total samples *before* balancing: {sum(len(p) for p in category_pools.values())}")
    print(f"Total samples *after* balancing:  {len(balanced_dataset)}")
    print(f"(Expected: {17 * TARGET_SAMPLES_PER_CATEGORY})")

    # --- Third Pass: Shuffle and Save ---
    print("\n--- Pass 3: Shuffling and saving new dataset ---")
    
    # Shuffle the final dataset to mix categories
    random.shuffle(balanced_dataset)
    
    try:
        with open(BALANCED_DATA_FILE, 'w', encoding='utf-8') as f:
            for item in tqdm(balanced_dataset, desc="Writing file"):
                # Write each item as a JSON line
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    except Exception as e:
        print(f"ERROR: Could not write to file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- All Complete! ---")
    print(f"New balanced dataset saved to: {BALANCED_DATA_FILE}")

# --- Run the main function ---
if __name__ == "__main__":
    create_balanced_dataset()
