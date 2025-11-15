import json
import random
from collections import defaultdict


CONFIG = {
	"input_path": "binary_data_final.jsonl",
	"output_train_path": "binary_train_big.jsonl",
	"output_test_path": "binary_test_big.jsonl",
	"split": 0.90,
	"seed": 42,
	"hhrlhf_counts": {"bn": 7500, "kn": 7500, "ml": 7500, "or": 12500},
	"indiccorp_counts": {"bn": 7500, "kn": 7500, "ml": 7500, "or": 12500},
	"macd_counts": {"ml": 12000, "kn": 12000},
	"comm_bn_count": 6000,
	"bad_bn_count": 6000,
}


def read_buckets(path):
    buckets = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"skipping bad jsonl line {i}: {e}")
                continue
            key = (row.get("source"), row.get("language"))
            if None in key:
                print(f"skipping line {i}: missing source/language")
                continue
            buckets[key].append(row)
    return buckets


def pick(buckets, source, lang, num):
	if int(num) <= 0:
		return []
	key = (source, lang)
	pool = buckets.get(key, [])
	if len(pool) < int(num):
		raise RuntimeError(f"need {num} rows for {source}/{lang} but only have {len(pool)}")
	random.shuffle(pool)
	grab = pool[: int(num)]
	buckets[key] = pool[int(num) :]
	return grab


def label(row):
	value = row.get("safe(0)/harmful(1)")
	try:
		value = int(float(value))
	except Exception:
		value = 0
	return "UNSAFE" if value == 1 else "SAFE"


def write_jsonl(path, rows):
	with open(path, "w", encoding="utf-8", newline="\n") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(cfg):
	random.seed(cfg.get("seed", 0))
	buckets = read_buckets(cfg["input_path"])
	grabbed = []
	for lang, num in cfg["hhrlhf_counts"].items():
		grabbed.extend(pick(buckets, "ai4bharat/indic-align", lang, num))
	for lang, num in cfg["indiccorp_counts"].items():
		grabbed.extend(pick(buckets, "ai4bharat/IndicCorpV2", lang, num))
	grabbed.extend(pick(buckets, "ShareChatAI/MACD", "ml", cfg["macd_counts"].get("ml", 0)))
	grabbed.extend(pick(buckets, "ShareChatAI/MACD", "kn", cfg["macd_counts"].get("kn", 0)))
	grabbed.extend(pick(buckets, "Multi Labeled Bengali Toxic Comments", "bn", cfg.get("comm_bn_count", 0)))
	grabbed.extend(pick(buckets, "BAD-Bangla-Aggressive-Text-Dataset", "bn", cfg.get("bad_bn_count", 0)))
	if not grabbed:
		raise RuntimeError("nothing was sampled")
	rows = [{"text": str(row.get("text", "")), "label": label(row)} for row in grabbed]
	random.shuffle(rows)
	split = float(cfg.get("split", 0.8))
	cut = int(len(rows) * split)
	cut = max(1, min(cut, len(rows) - 1))
	train_rows = rows[:cut]
	test_rows = rows[cut:]
	write_jsonl(cfg["output_train_path"], train_rows)
	write_jsonl(cfg["output_test_path"], test_rows)
	print(f"train {len(train_rows)} -> {cfg['output_train_path']}")
	print(f"test {len(test_rows)} -> {cfg['output_test_path']}")


if __name__ == "__main__":
	main(CONFIG)

