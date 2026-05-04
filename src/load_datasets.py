"""Download and save the 3 datasets to /workspace/datasets/."""

import json
import random
import csv
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download

random.seed(42)
OUT = "/workspace/datasets"


def download_harmful():
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    entries = [{"prompt": r["Goal"], "category": r.get("Category", "")} for r in ds]
    path = f"{OUT}/harmful_real.json"
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"harmful: {len(entries)} entries saved")
    return entries


def download_medical():
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    indices = random.sample(range(len(ds)), 100)
    entries = [{"prompt": ds[i]["Question"], "answer": ds[i]["Answer"]} for i in sorted(indices)]
    path = f"{OUT}/medical_real.json"
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"medical: {len(entries)} entries saved")
    return entries


def download_legal():
    files = list_repo_files("nguha/legalbench", repo_type="dataset")
    tsv_test = [f for f in files if f.startswith("data/") and f.endswith("/test.tsv")]
    random.shuffle(tsv_test)

    entries = []
    tasks_used = set()
    for tf in tsv_test:
        if len(tasks_used) >= 10:
            break
        task = tf.split("/")[1]
        p = hf_hub_download("nguha/legalbench", tf, repo_type="dataset")
        with open(p) as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        if len(rows) < 5:
            continue
        cols = list(rows[0].keys())
        text_col = next((c for c in cols if c.lower() in ("question", "text")), cols[0])
        for row in random.sample(rows, min(10, len(rows))):
            entries.append({"prompt": row[text_col], "task": task})
        tasks_used.add(task)

    entries = entries[:100]
    path = f"{OUT}/legal_real.json"
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"legal: {len(entries)} entries across {len(tasks_used)} tasks")
    return entries


if __name__ == "__main__":
    download_harmful()
    download_medical()
    download_legal()

    # Verify
    for name in ["harmful_real", "medical_real", "legal_real"]:
        data = json.load(open(f"{OUT}/{name}.json"))
        print(f"\n{name}: {len(data)} entries")
        for e in data[:2]:
            print(f"  - {e['prompt'][:80]}...")
