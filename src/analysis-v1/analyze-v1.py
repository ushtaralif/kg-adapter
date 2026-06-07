"""
Usage:
  python analyze-v1.py --input evaluated_final_cleaned_data.json
"""
import json
import argparse
import numpy as np
from itertools import combinations


def load_data(path):
    with open(path) as f:
        data = json.load(f)

    required = [
        "llama_base_single_eval", "llama_adapter_single_eval",
        "mistral_base_single_eval", "mistral_adapter_single_eval",
        "entropy_risk", "entropy", "wikidata_metadata"
    ]
    return [e for e in data if all(k in e for k in required)]


def get_score(entry, model, version):
    key = f"{model}_{version}_single_eval"
    return entry[key]["label_id"]


def is_correct(score, threshold=3):
    return score <= threshold


def get_features(entry):
    """Extract all structural features from entry."""
    meta = entry.get("wikidata_metadata", {})

    # Average across entities if multiple
    num_statements, num_references = [], []
    num_properties, num_sitelinks = [], []
    last_modified = []

    for qid, vals in meta.items():
        if isinstance(vals, dict):
            num_statements.append(vals.get("num_statements", 0))
            num_references.append(vals.get("num_references", 0))
            num_properties.append(vals.get("num_properties", 0))
            num_sitelinks.append(vals.get("num_sitelinks", 0))

    return {
        "entropy_risk":    entry.get("entropy_risk", 0),
        "entropy":         entry.get("entropy", 0),
        "num_statements":  np.mean(num_statements) if num_statements else 0,
        "num_references":  np.mean(num_references) if num_references else 0,
        "num_properties":  np.mean(num_properties) if num_properties else 0,
        "num_sitelinks":   np.mean(num_sitelinks)  if num_sitelinks  else 0,
    }


def compute_accuracy_gated(entries, model, feat_name, threshold,
                             inject_when, correct_thresh=3):
    """
    Compute accuracy under a gated strategy.
    inject_when: 'high' = inject when feat >= threshold
                 'low'  = inject when feat <= threshold
    """
    correct = 0
    for e in entries:
        feat_val = get_features(e)[feat_name]
        base_score    = get_score(e, model, "base")
        adapter_score = get_score(e, model, "adapter")

        if inject_when == "high":
            inject = feat_val >= threshold
        else:
            inject = feat_val <= threshold

        score = adapter_score if inject else base_score
        if is_correct(score, correct_thresh):
            correct += 1
    return correct / len(entries)


def compute_inject_pct(entries, feat_name, threshold, inject_when):
    count = 0
    for e in entries:
        val = get_features(e)[feat_name]
        if inject_when == "high" and val >= threshold:
            count += 1
        elif inject_when == "low" and val <= threshold:
            count += 1
    return 100 * count / len(entries)


def sweep_single_feature(entries, model, feat_name, inject_when,
                          percentiles, correct_thresh=3):
    vals = [get_features(e)[feat_name] for e in entries]
    thresholds = [np.percentile(vals, p) for p in percentiles]

    base_acc  = np.mean([is_correct(get_score(e, model, "base"),
                                    correct_thresh) for e in entries])
    alwaysk_acc = np.mean([is_correct(get_score(e, model, "adapter"),
                                      correct_thresh) for e in entries])

    print(f"\n  Feature: {feat_name}  [{inject_when}_inject]")
    print(f"  {'Threshold':>12} {'Pct':>6} {'Inject%':>9} "
          f"{'Accuracy':>10} {'vs Always-KG':>14} {'vs Base':>10}")
    print(f"  {'-'*65}")

    best_acc, best_pct, best_thresh, best_p = 0, 0, 0, 0
    for p, tau in zip(percentiles, thresholds):
        acc     = compute_accuracy_gated(entries, model, feat_name,
                                         tau, inject_when, correct_thresh)
        inj_pct = compute_inject_pct(entries, feat_name, tau, inject_when)
        vs_kg   = f"{'✓' if acc > alwaysk_acc else ' '}"
        print(f"  {tau:>12.3f} {p:>5}% {inj_pct:>8.1f}% "
              f"{100*acc:>9.2f}% {vs_kg}")
        if acc > best_acc:
            best_acc, best_pct = acc, inj_pct
            best_thresh, best_p = tau, p

    print(f"  → Best: thresh={best_thresh:.3f} (p{best_p}) "
          f"acc={100*best_acc:.2f}% inject={best_pct:.1f}%")
    return best_acc, best_thresh, best_p, inject_when


def sweep_two_features(entries, model, feat_configs, correct_thresh=3):
    """feat_configs: list of (feat_name, inject_when, threshold) tuples"""
    base_acc    = np.mean([is_correct(get_score(e, model, "base"),
                                      correct_thresh) for e in entries])
    alwaysk_acc = np.mean([is_correct(get_score(e, model, "adapter"),
                                      correct_thresh) for e in entries])

    print(f"\n{'─'*70}")
    print(f"  PART 2: TWO-FEATURE COMBINATIONS")
    print(f"{'─'*70}")

    percentiles = list(range(5, 100, 5))
    all_feat_vals = {}
    for e in entries:
        feats = get_features(e)
        for k, v in feats.items():
            all_feat_vals.setdefault(k, []).append(v)

    # Define feature directions
    feat_directions = {
        "entropy_risk":   "high",
        "entropy":        "high",
        "num_statements": "low",
        "num_references": "low",
        "num_properties": "low",
        "num_sitelinks":  "low",
    }

    feat_names = list(feat_directions.keys())
    best_or_acc, best_and_acc = 0, 0
    best_or_cfg, best_and_cfg = None, None

    for f1, f2 in combinations(feat_names, 2):
        d1, d2 = feat_directions[f1], feat_directions[f2]
        vals1 = [get_features(e)[f1] for e in entries]
        vals2 = [get_features(e)[f2] for e in entries]

        best_or_local, best_and_local = 0, 0
        best_or_t, best_and_t = None, None

        for p1 in percentiles:
            t1 = np.percentile(vals1, p1)
            for p2 in percentiles:
                t2 = np.percentile(vals2, p2)

                # OR gate
                or_correct = 0
                and_correct = 0
                or_inj = 0
                and_inj = 0

                for e in entries:
                    v1 = get_features(e)[f1]
                    v2 = get_features(e)[f2]
                    base_s    = get_score(e, model, "base")
                    adapter_s = get_score(e, model, "adapter")

                    fire1 = (v1 >= t1) if d1 == "high" else (v1 <= t1)
                    fire2 = (v2 >= t2) if d2 == "high" else (v2 <= t2)

                    # OR
                    or_fire = fire1 or fire2
                    or_s = adapter_s if or_fire else base_s
                    if is_correct(or_s, correct_thresh): or_correct += 1
                    if or_fire: or_inj += 1

                    # AND
                    and_fire = fire1 and fire2
                    and_s = adapter_s if and_fire else base_s
                    if is_correct(and_s, correct_thresh): and_correct += 1
                    if and_fire: and_inj += 1

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

        vs_or  = "✓" if best_or_local  > alwaysk_acc else " "
        vs_and = "✓" if best_and_local > alwaysk_acc else " "

        print(f"\n  {f1} [{d1}] × {f2} [{d2}]")
        if best_or_t:
            t1,p1,t2,p2,inj = best_or_t
            print(f"    OR  gate: {f1}>={t1:.3f}(p{p1}) OR  "
                  f"{f2}>={t2:.3f}(p{p2})"
                  f" → acc={100*best_or_local:.2f}% "
                  f"inject={inj:.1f}% {vs_or}")
        if best_and_t:
            t1,p1,t2,p2,inj = best_and_t
            print(f"    AND gate: {f1}>={t1:.3f}(p{p1}) AND "
                  f"{f2}>={t2:.3f}(p{p2})"
                  f" → acc={100*best_and_local:.2f}% "
                  f"inject={inj:.1f}% {vs_and}")

        if best_or_local > best_or_acc:
            best_or_acc = best_or_local
            best_or_cfg = (f1, f2, best_or_t)
        if best_and_local > best_and_acc:
            best_and_acc = best_and_local
            best_and_cfg = (f1, f2, best_and_t)

    return best_or_acc, best_and_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=int, default=3,
                        help="Correct if label_id <= threshold (default 3)")
    args = parser.parse_args()

    entries = load_data(args.input)
    print(f"[sweep] {len(entries)} complete entries")
    print(f"[sweep] Correct = label_id <= {args.threshold}")

    percentiles = list(range(5, 100, 5))

    feat_directions = {
        "entropy_risk":   "high",   # high entropy → inject
        "entropy":        "high",
        "num_statements": "low",    # low statements → inject
        "num_references": "low",
        "num_properties": "low",
        "num_sitelinks":  "low",
    }

    for model in ["llama", "mistral"]:
        base_acc = np.mean([
            is_correct(get_score(e, model, "base"), args.threshold)
            for e in entries])
        alwaysk_acc = np.mean([
            is_correct(get_score(e, model, "adapter"), args.threshold)
            for e in entries])

        print(f"\n{'='*70}")
        print(f"  THRESHOLD SWEEP — {model.upper()}  (n={len(entries)})")
        print(f"  Base={100*base_acc:.2f}%  Always-KG={100*alwaysk_acc:.2f}%")
        print(f"{'='*70}")

        print(f"\n{'─'*70}")
        print(f"  PART 1: SINGLE FEATURE SWEEPS")
        print(f"{'─'*70}")

        best_single, best_single_acc = None, 0
        for feat, direction in feat_directions.items():
            acc, tau, p, d = sweep_single_feature(
                entries, model, feat, direction,
                percentiles, args.threshold)
            if acc > best_single_acc:
                best_single_acc = acc
                best_single = (feat, tau, p, d)

        best_or_acc, best_and_acc = sweep_two_features(
            entries, model, [], args.threshold)

        print(f"\n{'─'*70}")
        print(f"  PART 3: CONTRIBUTION ANALYSIS")
        print(f"{'─'*70}")
        print(f"  Best single feature  : {100*best_single_acc:.2f}%")
        print(f"  Best two-feature OR  : {100*best_or_acc:.2f}%")
        print(f"  Best two-feature AND : {100*best_and_acc:.2f}%")
        print(f"  Always-KG            : {100*alwaysk_acc:.2f}%")
        print(f"  Base                 : {100*base_acc:.2f}%")


if __name__ == "__main__":
    main()