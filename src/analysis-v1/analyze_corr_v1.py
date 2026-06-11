import json
import math
import sys
from collections import Counter

import numpy as np
from scipy.stats import pearsonr, spearmanr

PATH = sys.argv[1] if len(sys.argv) > 1 else "../analysis-v1/gpt_evaluated_combined-v2.json"
ALPHA = 0.05


def calculate_entity_entropy(wikidata_metadata):
    total_statements = sum(entity.get("num_statements", 0) for entity in wikidata_metadata.values())
    if total_statements == 0:
        return 0.0

    entropy = -sum(
        (entity.get("num_statements", 0) / total_statements)
        * math.log2(entity.get("num_statements", 0) / total_statements)
        for entity in wikidata_metadata.values()
        if entity.get("num_statements", 0) > 0
    )
    return entropy


def score_to_label(score):
    if score is None:
        return "unknown"
    if 1 <= score <= 3:
        return "incorrect"
    elif 4 <= score <= 7:
        return "partial_correct"
    elif 8 <= score <= 10:
        return "correct"
    return "unknown"



def safe_sum_metadata(wikidata_metadata, key):
    return sum(v.get(key, 0) for v in wikidata_metadata.values())


def corr_with_significance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {
            "n": int(len(x)),
            "pearson_r": None,
            "pearson_p": None,
            "pearson_sig": False,
            "spearman_rho": None,
            "spearman_p": None,
            "spearman_sig": False,
        }

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    return {
        "n": int(len(x)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "pearson_sig": bool(pearson_p < ALPHA),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "spearman_sig": bool(spearman_p < ALPHA),
    }

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and "examples" in data:
    examples = data["examples"]
else:
    examples = data

rows = []

for ex in examples:
    wikidata_metadata = ex.get("wikidata_metadata", {}) or {}

    entity_entropy = calculate_entity_entropy(wikidata_metadata)
    num_statements = safe_sum_metadata(wikidata_metadata, "num_statements")
    num_references = safe_sum_metadata(wikidata_metadata, "num_references")
    num_properties = safe_sum_metadata(wikidata_metadata, "num_properties")
    num_sitelinks = safe_sum_metadata(wikidata_metadata, "num_sitelinks")

    rows.append({
        "id": ex.get("id"),
        "entropy": ex.get("entropy", 0.0),
        "entropy_risk": ex.get("entropy_risk", 0.0),
        "entity_entropy": entity_entropy,
        "num_statements": num_statements,
        "num_references": num_references,
        "num_properties": num_properties,
        "num_sitelinks": num_sitelinks,
        "llama_base_score": ex.get("llama_base_score"),
        "llama_adapter_score": ex.get("llama_adapter_score"),
        "mistral_base_score": ex.get("mistral_base_score"),
        "mistral_adapter_score": ex.get("mistral_adapter_score"),
    })

print(f"\nTotal examples loaded: {len(rows)}")

score_fields = [
    "llama_base_score",
    "llama_adapter_score",
    "mistral_base_score",
    "mistral_adapter_score",
]

feature_fields = [
    "entropy",
    "entropy_risk",
    "entity_entropy",
    "num_statements",
    "num_references",
    "num_properties",
    "num_sitelinks",
]

# optional: score-bin summaries
print("\n=== Score Bin Distribution ===")
# for score_field in score_fields:
#     cnt = Counter(score_to_label(r.get(score_field)) for r in rows if r.get(score_field) is not None)
#     print(f"\n{score_field}:")
#     for k in ["incorrect", "partial_correct", "correct"]:
#         print(f"  {k}: {cnt.get(k, 0)}")

# score summaries
print("\n=== Score Distribution ===")
for score_field in score_fields:
    raw_cnt = Counter()
    label_cnt = Counter()

    for r in rows:
        score = r.get(score_field)
        if score is None:
            continue

        try:
            score = int(score)
        except (TypeError, ValueError):
            continue

        raw_cnt[score] += 1
        label_cnt[score_to_label(score)] += 1

    print(f"\n{score_field}:")
    print("  Raw scores:")
    for s in range(1, 11):
        print(f"    {s}: {raw_cnt.get(s, 0)}")

    print("  Labels:")
    for k in ["incorrect", "partial_correct", "correct"]:
        print(f"    {k}: {label_cnt.get(k, 0)}")


print("\n=== Correlation Analysis (alpha = 0.05) ===")

all_results = {}

for score_field in score_fields:
    print(f"\n\n########## {score_field} ##########")
    all_results[score_field] = {}

    valid_rows = [r for r in rows if r.get(score_field) is not None]
    scores = [r[score_field] for r in valid_rows]

    for feat in feature_fields:
        values = [r[feat] for r in valid_rows]
        res = corr_with_significance(scores, values)
        all_results[score_field][feat] = res

        print(
            f"{feat:16s} | "
            f"Pearson r={res['pearson_r']:.4f}, p={res['pearson_p']:.6f}, sig={res['pearson_sig']} | "
            f"Spearman rho={res['spearman_rho']:.4f}, p={res['spearman_p']:.6f}, sig={res['spearman_sig']}"
        )

out_file = "../analysis-v1/correlation_results_10point.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved correlation results to: {out_file}")