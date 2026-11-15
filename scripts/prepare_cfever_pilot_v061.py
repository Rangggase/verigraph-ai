
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/datasets/cfever/train.jsonl"
OUTPUT = ROOT / "data/pilots/cfever_zh_train_50.jsonl"

SEED = 42
SAMPLE_SIZE = 50

records = []
seen = set()

with SOURCE.open("r", encoding="utf-8-sig") as f:
    for row_number, line in enumerate(f, start=1):
        if not line.strip():
            continue

        row = json.loads(line)
        claim = row.get("claim", "").strip()

        if not claim or claim in seen:
            continue

        seen.add(claim)

        records.append({
            "claim_id": f"cfever_train_{row_number:06d}",
            "claim": claim,
            "language": "zh",
            "source_dataset": "cfever",
            "source_split": "train",
            "source_row": row_number
        })

if len(records) < SAMPLE_SIZE:
    raise ValueError("Jumlah klaim valid kurang dari 50.")

rng = random.Random(SEED)
selected = rng.sample(records, SAMPLE_SIZE)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for record in selected:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("=== VERIGRAPH AI v0.6.1 ===")
print("Pilot dataset berhasil dibuat.")
print("Jumlah klaim:", len(selected))
print("Seed:", SEED)
print("Sumber:", SOURCE)
print("Output:", OUTPUT)

print("\n=== CONTOH 3 KLAIM ===")
for item in selected[:3]:
    print(item["claim_id"], ":", item["claim"])
