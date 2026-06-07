"""
Usage:
  python analyze-v3.py \
    --input logs/gpt_evaluated_combined.json
"""
import json
import math
import argparse
import numpy as np
from itertools import product


# ── Features ──────────────────────────────────────────────────────────────────

def compute_entity_entropy(wikidata_metadata):
    total = sum(e.get("num_statements", 0) for e in wikidata_metadata.values())
    if total == 0: return 0.0
    return -sum(
        (e["num_statements"]/total) * math.log2(e["num_statements"]/total)
        for e in wikidata_metadata.values()
        if e.get("num_statements", 0) > 0
    )

def extract(entry):
    meta = entry.get("wikidata_metadata", {})
    entities = list(meta.values())
    return {
        "entropy_risk":   entry.get("entropy_risk", 0.0),
        "num_sitelinks":  sum(e.get("num_sitelinks",  0) for e in entities),
        "num_references": sum(e.get("num_references", 0) for e in entities),
        "num_statements": sum(e.get("num_statements", 0) for e in entities),
        "entity_entropy": compute_entity_entropy(meta),
        "entropy":        entry.get("entropy", 0.0),
    }

def is_correct(score, t=3):
    return score >= t


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(y_base, y_adapter, inject_mask):
    result = np.where(inject_mask, y_adapter, y_base)
    return result.mean(), inject_mask.mean()


# ── Threshold sweep ───────────────────────────────────────────────────────────

def sweep_single(name, vals, y_base, y_adapter, direction, alwkg_acc):
    """
    Sweep all percentile thresholds for a single feature.
    direction: 'high_inject' = inject when feature is HIGH (entropy_risk)
               'low_inject'  = inject when feature is LOW  (sitelinks, refs, statements)
    """
    percentiles = list(range(5, 100, 5))
    thresholds  = [np.percentile(vals, p) for p in percentiles]

    print(f"\n  Feature: {name}  [{direction}]")
    print(f"  {'Threshold':>12} {'Pct':>5} {'Inject%':>9} "
          f"{'Accuracy':>10} {'vs Always-KG':>14} {'vs Base':>10}")
    print(f"  {'-'*65}")

    best_acc, best_row = 0, None
    for p, thresh in zip(percentiles, thresholds):
        if direction == "high_inject":
            mask = vals >= thresh   # inject when HIGH entropy_risk
        else:
            mask = vals <= thresh   # inject when LOW sitelinks/refs/statements

        acc, inj = simulate(y_base, y_adapter, mask)
        marker   = " ✓" if acc > alwkg_acc else ""
        print(f"  {thresh:>12.3f} {p:>5}% {100*inj:>8.1f}% "
              f"{100*acc:>9.2f}%{marker}")
        if acc > best_acc:
            best_acc = acc
            best_row = (thresh, p, acc, inj)

    if best_row:
        print(f"  → Best: thresh={best_row[0]:.3f} (p{best_row[1]}) "
              f"acc={100*best_row[2]:.2f}% inject={100*best_row[3]:.1f}%")
    return best_row


def sweep_combined_two(f1_name, f1_vals, f1_dir,
                        f2_name, f2_vals, f2_dir,
                        y_base, y_adapter, alwkg_acc):
    """
    For two features: try all combinations of their percentile thresholds.
    Inject when EITHER condition is met (OR gate) or BOTH (AND gate).
    """
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    t1_vals = [np.percentile(f1_vals, p) for p in percentiles]
    t2_vals = [np.percentile(f2_vals, p) for p in percentiles]

    best_or,  best_or_row  = 0, None
    best_and, best_and_row = 0, None

    for p1, t1 in zip(percentiles, t1_vals):
        m1 = (f1_vals >= t1) if f1_dir == "high_inject" else (f1_vals <= t1)
        for p2, t2 in zip(percentiles, t2_vals):
            m2 = (f2_vals >= t2) if f2_dir == "high_inject" else (f2_vals <= t2)

            # OR gate: inject if either condition met
            acc_or,  inj_or  = simulate(y_base, y_adapter, m1 | m2)
            # AND gate: inject only if both conditions met
            acc_and, inj_and = simulate(y_base, y_adapter, m1 & m2)

            if acc_or > best_or:
                best_or     = acc_or
                best_or_row = (p1, t1, p2, t2, acc_or, inj_or)
            if acc_and > best_and:
                best_and     = acc_and
                best_and_row = (p1, t1, p2, t2, acc_and, inj_and)

    print(f"\n  {f1_name} [{f1_dir}] × {f2_name} [{f2_dir}]")
    if best_or_row:
        r = best_or_row
        marker = " ✓" if r[4] > alwkg_acc else ""
        print(f"    OR  gate: {f1_name}>={'='+str(r[1]):<10} OR  "
              f"{f2_name}>={'='+str(r[3]):<10} "
              f"→ acc={100*r[4]:.2f}% inject={100*r[5]:.1f}%{marker}")
    if best_and_row:
        r = best_and_row
        marker = " ✓" if r[4] > alwkg_acc else ""
        print(f"    AND gate: {f1_name}>={'='+str(r[1]):<10} AND "
              f"{f2_name}>={'='+str(r[3]):<10} "
              f"→ acc={100*r[4]:.2f}% inject={100*r[5]:.1f}%{marker}")

    return best_or, best_and, best_or_row, best_and_row


def sweep_all_three(feats, y_base, y_adapter, alwkg_acc):
    """
    Three features combined with OR gate — sweep all threshold combinations.
    """
    names  = [f[0] for f in feats]
    vals   = [f[1] for f in feats]
    dirs   = [f[2] for f in feats]
    pcts   = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    best_acc, best_config = 0, None

    for p1 in pcts:
        t1 = np.percentile(vals[0], p1)
        m1 = (vals[0] >= t1) if dirs[0] == "high_inject" else (vals[0] <= t1)
        for p2 in pcts:
            t2 = np.percentile(vals[1], p2)
            m2 = (vals[1] >= t2) if dirs[1] == "high_inject" else (vals[1] <= t2)
            for p3 in pcts:
                t3 = np.percentile(vals[2], p3)
                m3 = (vals[2] >= t3) if dirs[2] == "high_inject" else (vals[2] <= t3)

                for gate in ["OR", "AND"]:
                    mask = (m1 | m2 | m3) if gate == "OR" else (m1 & m2 & m3)
                    acc, inj = simulate(y_base, y_adapter, mask)
                    if acc > best_acc:
                        best_acc    = acc
                        best_config = (gate, p1, t1, p2, t2, p3, t3, acc, inj)

    if best_config:
        g, p1, t1, p2, t2, p3, t3, acc, inj = best_config
        marker = " ✓" if acc > alwkg_acc else ""
        print(f"\n  Best 3-feature {g} gate:")
        print(f"    {names[0]} p{p1}={t1:.3f}  {g}  "
              f"{names[1]} p{p2}={t2:.3f}  {g}  "
              f"{names[2]} p{p3}={t3:.3f}")
        print(f"    → acc={100*acc:.2f}%  inject={100*inj:.1f}%{marker}")
    return best_acc, best_config


def analyze(entries, model):
    base_key    = f"{model}_base_score"
    adapter_key = f"{model}_adapter_score"

    feats    = [extract(e) for e in entries]
    y_base   = np.array([is_correct(e[base_key])    for e in entries])
    y_adapter= np.array([is_correct(e[adapter_key]) for e in entries])

    base_acc  = y_base.mean()
    alwkg_acc = y_adapter.mean()
    total     = len(entries)

    # Feature arrays
    entropy_risk   = np.array([f["entropy_risk"]   for f in feats])
    num_sitelinks  = np.array([f["num_sitelinks"]  for f in feats])
    num_references = np.array([f["num_references"] for f in feats])
    num_statements = np.array([f["num_statements"] for f in feats])
    entity_entropy = np.array([f["entity_entropy"] for f in feats])
    entropy        = np.array([f["entropy"]        for f in feats])

    print(f"\n{'='*70}")
    print(f"  THRESHOLD SWEEP — {model.upper()}  (n={total})")
    print(f"  Base={100*base_acc:.2f}%  Always-KG={100*alwkg_acc:.2f}%")
    print(f"{'='*70}")

    print(f"\n{'─'*70}")
    print(f"  PART 1: SINGLE FEATURE SWEEPS")
    print(f"  Hypothesis: inject when knowledge is WEAK or uncertainty is HIGH")
    print(f"{'─'*70}")

    # Negatively correlated: inject when HIGH
    r1 = sweep_single("entropy_risk",   entropy_risk,   y_base, y_adapter, "high_inject", alwkg_acc)
    r2 = sweep_single("entropy",        entropy,        y_base, y_adapter, "high_inject", alwkg_acc)
    r3 = sweep_single("entity_entropy", entity_entropy, y_base, y_adapter, "high_inject", alwkg_acc)

    # Positively correlated: inject when LOW
    r4 = sweep_single("num_sitelinks",  num_sitelinks,  y_base, y_adapter, "low_inject", alwkg_acc)
    r5 = sweep_single("num_references", num_references, y_base, y_adapter, "low_inject", alwkg_acc)
    r6 = sweep_single("num_statements", num_statements, y_base, y_adapter, "low_inject", alwkg_acc)

    print(f"\n{'─'*70}")
    print(f"  PART 2: TWO-FEATURE COMBINATIONS")
    print(f"  Testing OR and AND gates across all threshold pairs")
    print(f"{'─'*70}")

    # Most promising pairs based on correlation
    pairs = [
        ("entropy_risk",   entropy_risk,   "high_inject",
         "num_sitelinks",  num_sitelinks,  "low_inject"),
        ("entropy_risk",   entropy_risk,   "high_inject",
         "num_references", num_references, "low_inject"),
        ("entropy_risk",   entropy_risk,   "high_inject",
         "num_statements", num_statements, "low_inject"),
        ("num_sitelinks",  num_sitelinks,  "low_inject",
         "num_references", num_references, "low_inject"),
        ("num_references", num_references, "low_inject",
         "num_statements", num_statements, "low_inject"),
        ("entropy_risk",   entropy_risk,   "high_inject",
         "entity_entropy", entity_entropy, "high_inject"),
    ]

    best_two_acc = 0
    for f1n, f1v, f1d, f2n, f2v, f2d in pairs:
        or_acc, and_acc, or_row, and_row = sweep_combined_two(
            f1n, f1v, f1d, f2n, f2v, f2d, y_base, y_adapter, alwkg_acc)
        best_two_acc = max(best_two_acc, or_acc, and_acc)

    print(f"\n{'─'*70}")
    print(f"  PART 3: THREE-FEATURE OR/AND SWEEP")
    print(f"  entropy_risk (high) × num_sitelinks (low) × num_references (low)")
    print(f"{'─'*70}")

    three_feats = [
        ("entropy_risk",   entropy_risk,   "high_inject"),
        ("num_sitelinks",  num_sitelinks,  "low_inject"),
        ("num_references", num_references, "low_inject"),
    ]
    best_three, _ = sweep_all_three(three_feats, y_base, y_adapter, alwkg_acc)

    print(f"\n{'─'*70}")
    print(f"  PART 4: CONTRIBUTION ANALYSIS")
    print(f"  Does each feature add value over the best single feature?")
    print(f"{'─'*70}")

    # Find best single
    singles = [r for r in [r1,r2,r3,r4,r5,r6] if r is not None]
    best_single = max(singles, key=lambda x: x[2])
    best_single_acc = best_single[2]

    print(f"  Best single feature  : {100*best_single_acc:.2f}%")
    print(f"  Best two-feature     : {100*best_two_acc:.2f}%")
    print(f"  Best three-feature   : {100*best_three:.2f}%")
    print(f"  Always-KG            : {100*alwkg_acc:.2f}%")
    print(f"  Marginal gain 1→2    : {100*(best_two_acc-best_single_acc):+.3f}%")
    print(f"  Marginal gain 2→3    : {100*(best_three-best_two_acc):+.3f}%")

    print(f"\n  Conclusion:")
    if best_three > alwkg_acc:
        savings = 1.0 - best_single[3]
        print(f"  ✓ Features beat Always-KG")
        print(f"  ✓ Best single rule saves {100*savings:.1f}% KG computation")
    else:
        print(f"  ✗ Features do not reliably beat Always-KG")
        print(f"  → Structural features are correlated with errors but")
        print(f"    not sufficient alone to outperform always-inject strategy")

    return {
        "model":          model,
        "base_acc":       base_acc,
        "alwkg_acc":      alwkg_acc,
        "best_single":    best_single_acc,
        "best_two":       best_two_acc,
        "best_three":     best_three,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        entries = json.load(f)

    required = ["llama_base_score", "llama_adapter_score",
                "mistral_base_score", "mistral_adapter_score",
                "entropy", "entropy_risk", "wikidata_metadata"]
    entries = [e for e in entries if all(k in e for k in required)]
    print(f"[threshold_sweep] {len(entries)} complete entries")

    results = {}
    for model in ["llama", "mistral"]:
        results[model] = analyze(entries, model)

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<10} {'Base':>8} {'Always-KG':>11} "
          f"{'1-feat':>8} {'2-feat':>8} {'3-feat':>8}")
    print(f"  {'-'*58}")
    for model, r in results.items():
        print(f"  {model:<10} {100*r['base_acc']:>7.2f}% "
              f"{100*r['alwkg_acc']:>10.2f}% "
              f"{100*r['best_single']:>7.2f}% "
              f"{100*r['best_two']:>7.2f}% "
              f"{100*r['best_three']:>7.2f}%")


if __name__ == "__main__":
    main()