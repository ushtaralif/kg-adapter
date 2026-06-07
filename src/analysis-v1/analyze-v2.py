"""
Correct  = FACTUAL
Incorrect = HALLUCINATION or PARTIAL

Usage:
  python analyze-v2.py --input evaluated_final_cleaned_data.json
"""
import json
import argparse
import numpy as np
from itertools import combinations


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    required = [
        "llama_base_halluc_eval", "llama_adapter_halluc_eval",
        "mistral_base_halluc_eval", "mistral_adapter_halluc_eval",
        "entropy_risk", "entropy", "wikidata_metadata"
    ]
    return [e for e in data if all(k in e for k in required)]


def get_halluc_label(entry, model, version):
    key = f"{model}_{version}_halluc_eval"
    return entry[key]["hallucination_label"]


def is_factual(entry, model, version):
    return get_halluc_label(entry, model, version) == "FACTUAL"


def is_not_hallucination(entry, model, version):
    return get_halluc_label(entry, model, version) != "HALLUCINATION"


def get_features(entry):
    meta = entry.get("wikidata_metadata", {})
    num_statements, num_references = [], []
    num_properties, num_sitelinks  = [], []

    for qid, vals in meta.items():
        if isinstance(vals, dict):
            num_statements.append(vals.get("num_statements", 0))
            num_references.append(vals.get("num_references", 0))
            num_properties.append(vals.get("num_properties", 0))
            num_sitelinks.append(vals.get("num_sitelinks",  0))

    return {
        "entropy_risk":   entry.get("entropy_risk", 0),
        "entropy":        entry.get("entropy", 0),
        "num_statements": np.mean(num_statements) if num_statements else 0,
        "num_references": np.mean(num_references) if num_references else 0,
        "num_properties": np.mean(num_properties) if num_properties else 0,
        "num_sitelinks":  np.mean(num_sitelinks)  if num_sitelinks  else 0,
    }


def compute_gated_factual(entries, model, feat_name, threshold,
                           inject_when, metric="factual"):
    """
    metric: 'factual'       → count FACTUAL answers
            'no_halluc'     → count FACTUAL + PARTIAL (not HALLUCINATION)
    """
    correct = 0
    injected = 0
    for e in entries:
        feat_val = get_features(e)[feat_name]
        fire = (feat_val >= threshold) if inject_when == "high" \
               else (feat_val <= threshold)

        version = "adapter" if fire else "base"
        if fire:
            injected += 1

        if metric == "factual":
            if is_factual(e, model, version):
                correct += 1
        else:
            if is_not_hallucination(e, model, version):
                correct += 1

    return correct / len(entries), 100 * injected / len(entries)


def sweep_single_feature(entries, model, feat_name, inject_when,
                          percentiles, metric="factual"):
    vals     = [get_features(e)[feat_name] for e in entries]
    base_acc = np.mean([is_factual(e, model, "base")
                        if metric == "factual"
                        else is_not_hallucination(e, model, "base")
                        for e in entries])
    kg_acc   = np.mean([is_factual(e, model, "adapter")
                        if metric == "factual"
                        else is_not_hallucination(e, model, "adapter")
                        for e in entries])

    print(f"\n  Feature: {feat_name}  [{inject_when}_inject]")
    print(f"  {'Threshold':>12} {'Pct':>6} {'Inject%':>9} "
          f"{'FACTUAL%':>10} {'vs Always-KG':>14}")
    print(f"  {'-'*60}")

    best_acc, best_inj, best_p, best_tau = 0, 0, 0, 0
    for p in percentiles:
        tau     = np.percentile(vals, p)
        acc, inj_pct = compute_gated_factual(
            entries, model, feat_name, tau, inject_when, metric)
        marker  = "✓" if acc > kg_acc else " "
        print(f"  {tau:>12.3f} {p:>5}% {inj_pct:>8.1f}% "
              f"{100*acc:>9.2f}% {marker}")
        if acc > best_acc:
            best_acc, best_inj = acc, inj_pct
            best_p, best_tau = p, tau

    print(f"  → Best: thresh={best_tau:.3f} (p{best_p}) "
          f"FACTUAL={100*best_acc:.2f}% inject={best_inj:.1f}%")
    return best_acc, best_tau, best_p


def sweep_two_features(entries, model, feat_directions,
                        percentiles, metric="factual"):
    kg_acc = np.mean([is_factual(e, model, "adapter")
                      if metric == "factual"
                      else is_not_hallucination(e, model, "adapter")
                      for e in entries])

    print(f"\n{'─'*70}")
    print(f"  PART 2: TWO-FEATURE COMBINATIONS")
    print(f"{'─'*70}")

    feat_names = list(feat_directions.keys())
    best_or_acc, best_and_acc = 0, 0
    best_or_str, best_and_str = "", ""

    for f1, f2 in combinations(feat_names, 2):
        d1 = feat_directions[f1]
        d2 = feat_directions[f2]
        vals1 = [get_features(e)[f1] for e in entries]
        vals2 = [get_features(e)[f2] for e in entries]

        best_or_local, best_and_local = 0, 0
        best_or_t, best_and_t = None, None

        for p1 in percentiles:
            t1 = np.percentile(vals1, p1)
            for p2 in percentiles:
                t2 = np.percentile(vals2, p2)

                or_correct = and_correct = 0
                or_inj = and_inj = 0

                for e in entries:
                    v1 = get_features(e)[f1]
                    v2 = get_features(e)[f2]

                    fire1 = (v1 >= t1) if d1 == "high" else (v1 <= t1)
                    fire2 = (v2 >= t2) if d2 == "high" else (v2 <= t2)

                    # OR gate
                    or_fire = fire1 or fire2
                    or_ver = "adapter" if or_fire else "base"
                    if or_fire:
                        or_inj += 1
                    ok_or = (is_factual(e, model, or_ver)
                             if metric == "factual"
                             else is_not_hallucination(e, model, or_ver))
                    if ok_or:
                        or_correct += 1

                    # AND gate
                    and_fire = fire1 and fire2
                    and_ver = "adapter" if and_fire else "base"
                    if and_fire:
                        and_inj += 1
                    ok_and = (is_factual(e, model, and_ver)
                              if metric == "factual"
                              else is_not_hallucination(e, model, and_ver))
                    if ok_and:
                        and_correct += 1

                or_acc  = or_correct  / len(entries)
                and_acc = and_correct / len(entries)

                if or_acc > best_or_local:
                    best_or_local = or_acc
                    best_or_t = (t1, p1, t2, p2,
                                 100*or_inj/len(entries))
                if and_acc > best_and_local:
                    best_and_local = and_acc
                    best_and_t = (t1, p1, t2, p2,
                                  100*and_inj/len(entries))

        vs_or  = "✓" if best_or_local  > kg_acc else " "
        vs_and = "✓" if best_and_local > kg_acc else " "

        print(f"\n  {f1} [{d1}] × {f2} [{d2}]")
        if best_or_t:
            t1,p1,t2,p2,inj = best_or_t
            print(f"    OR  gate: {f1}>={t1:.3f}(p{p1}) OR  "
                  f"{f2}>={t2:.3f}(p{p2})"
                  f" → FACTUAL={100*best_or_local:.2f}% "
                  f"inject={inj:.1f}% {vs_or}")
        if best_and_t:
            t1,p1,t2,p2,inj = best_and_t
            print(f"    AND gate: {f1}>={t1:.3f}(p{p1}) AND "
                  f"{f2}>={t2:.3f}(p{p2})"
                  f" → FACTUAL={100*best_and_local:.2f}% "
                  f"inject={inj:.1f}% {vs_and}")

        if best_or_local > best_or_acc:
            best_or_acc = best_or_local
            best_or_str = f"{f1}(p{best_or_t[1]}) OR {f2}(p{best_or_t[3]})"
        if best_and_local > best_and_acc:
            best_and_acc = best_and_local
            best_and_str = f"{f1}(p{best_and_t[1]}) AND {f2}(p{best_and_t[3]})"

    return best_or_acc, best_and_acc, best_or_str, best_and_str


def print_summary(model, base_acc, kg_acc, best_single,
                  best_or, best_and, metric):
    label = "FACTUAL%" if metric == "factual" else "non-HALLUC%"
    print(f"\n{'─'*70}")
    print(f"  SUMMARY — {model.upper()} ({label})")
    print(f"{'─'*70}")
    print(f"  Base LLM    : {100*base_acc:.2f}%")
    print(f"  Always-KG   : {100*kg_acc:.2f}%")
    print(f"  Best single : {100*best_single:.2f}%  "
          f"({'✓ beats Always-KG' if best_single > kg_acc else '✗ does not beat'})")
    print(f"  Best OR     : {100*best_or:.2f}%  "
          f"({'✓ beats Always-KG' if best_or > kg_acc else '✗ does not beat'})")
    print(f"  Best AND    : {100*best_and:.2f}%  "
          f"({'✓ beats Always-KG' if best_and > kg_acc else '✗ does not beat'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--metric", default="factual",
                        choices=["factual", "no_halluc"],
                        help="factual: count FACTUAL only. "
                             "no_halluc: count FACTUAL+PARTIAL.")
    args = parser.parse_args()

    entries = load_data(args.input)
    print(f"[sweep] {len(entries)} complete entries")
    print(f"[sweep] Metric: {args.metric}")

    percentiles    = list(range(5, 100, 5))
    feat_directions = {
        "entropy_risk":   "high",
        "entropy":        "high",
        "num_statements": "low",
        "num_references": "low",
        "num_properties": "low",
        "num_sitelinks":  "low",
    }

    for model in ["llama", "mistral"]:
        base_acc = np.mean([
            is_factual(e, model, "base")
            if args.metric == "factual"
            else is_not_hallucination(e, model, "base")
            for e in entries])
        kg_acc = np.mean([
            is_factual(e, model, "adapter")
            if args.metric == "factual"
            else is_not_hallucination(e, model, "adapter")
            for e in entries])

        print(f"\n{'='*70}")
        print(f"  HALLUCINATION SWEEP — {model.upper()} (n={len(entries)})")
        print(f"  Base={100*base_acc:.2f}%  Always-KG={100*kg_acc:.2f}%"
              f"  Metric={args.metric}")
        print(f"{'='*70}")

        print(f"\n{'─'*70}")
        print(f"  PART 1: SINGLE FEATURE SWEEPS")
        print(f"{'─'*70}")

        best_single = 0
        for feat, direction in feat_directions.items():
            acc, _, _ = sweep_single_feature(
                entries, model, feat, direction,
                percentiles, args.metric)
            if acc > best_single:
                best_single = acc

        best_or, best_and, or_str, and_str = sweep_two_features(
            entries, model, feat_directions,
            percentiles, args.metric)

        print_summary(model, base_acc, kg_acc,
                      best_single, best_or, best_and, args.metric)

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()