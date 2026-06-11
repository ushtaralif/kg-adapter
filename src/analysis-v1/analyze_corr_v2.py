

import json, math, numpy as np
from scipy.stats import spearmanr

PATH = "evaluated_final_cleaned_data-v2.json"
ALPHA = 0.05

def calculate_entity_entropy(wikidata_metadata):
    total = sum(e.get("num_statements", 0) for e in wikidata_metadata.values())
    if total == 0: return 0.0
    return -sum(
        (e.get("num_statements", 0) / total) * math.log2(e.get("num_statements", 0) / total)
        for e in wikidata_metadata.values() if e.get("num_statements", 0) > 0
    )

def safe_sum(meta, key):
    return sum(v.get(key, 0) for v in meta.values())

with open(PATH) as f:
    data = json.load(f)

rows = []
for ex in data:
    meta = ex.get("wikidata_metadata", {}) or {}
    rows.append({
        "entropy":        ex.get("entropy", 0.0),
        "entropy_risk":   ex.get("entropy_risk", 0.0),
        "entity_entropy": calculate_entity_entropy(meta),
        "num_statements": safe_sum(meta, "num_statements"),
        "num_references": safe_sum(meta, "num_references"),
        "num_properties": safe_sum(meta, "num_properties"),
        "num_sitelinks":  safe_sum(meta, "num_sitelinks"),
        # binary correct: 1 if label_id <= 3, else 0
        "llama_base_correct":    int(ex.get("llama_base_single_eval",    {}).get("label_id", 99) <= 3),
        "llama_adapter_correct": int(ex.get("llama_adapter_single_eval", {}).get("label_id", 99) <= 3),
        "mistral_base_correct":  int(ex.get("mistral_base_single_eval",  {}).get("label_id", 99) <= 3),
        "mistral_adapter_correct":int(ex.get("mistral_adapter_single_eval",{}).get("label_id", 99) <= 3),
    })

features = ["entropy", "entropy_risk", "entity_entropy",
            "num_statements", "num_references", "num_properties", "num_sitelinks"]

models = ["llama_base_correct", "llama_adapter_correct",
          "mistral_base_correct", "mistral_adapter_correct"]

print(f"n={len(rows)}\n")
print(f"{'Feature':<16} {'LLaMA Base':>14} {'LLaMA KG':>14} {'Mistral Base':>14} {'Mistral KG':>14}")
print("-" * 75)

for feat in features:
    row_str = f"{feat:<16}"
    for model in models:
        x = np.array([r[feat] for r in rows])
        y = np.array([r[model] for r in rows])
        rho, p = spearmanr(x, y)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        row_str += f"  {rho:+.4f}{sig:<2} (p={p:.4f})"
    print(row_str)