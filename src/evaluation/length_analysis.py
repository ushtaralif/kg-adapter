
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


def word_count(text: str) -> int:
    if not text: return 0
    return len(str(text).split())


def char_count(text: str) -> int:
    if not text: return 0
    return len(str(text).strip())


def is_correct(score: int, threshold: int = 4) -> bool:
    return score >= threshold


def load_data(path: str):
    with open(path) as f:
        data = json.load(f)
    required = ["llama_base_score", "llama_adapter_score",
                "mistral_base_score", "mistral_adapter_score",
                "llama_base", "llama_adapter",
                "mistral_base", "mistral_adapter",
                "answers"]
    return [e for e in data if all(k in e for k in required)]


def analyze_model(entries, model):
    base_key    = f"{model}_base"
    adapter_key = f"{model}_adapter"
    base_score_key    = f"{model}_base_score"
    adapter_score_key = f"{model}_adapter_score"

    results = []
    for e in entries:
        base_ans    = str(e.get(base_key,    "") or "")
        adapter_ans = str(e.get(adapter_key, "") or "")
        gold_ans    = e.get("answers", [""])
        if isinstance(gold_ans, list):
            gold_ans = gold_ans[0] if gold_ans else ""
        gold_ans = str(gold_ans)

        base_wc    = word_count(base_ans)
        adapter_wc = word_count(adapter_ans)
        gold_wc    = word_count(gold_ans)

        base_score    = e.get(base_score_key,    0)
        adapter_score = e.get(adapter_score_key, 0)

        results.append({
            "base_wc":        base_wc,
            "adapter_wc":     adapter_wc,
            "gold_wc":        gold_wc,
            "base_score":     base_score,
            "adapter_score":  adapter_score,
            "base_correct":   is_correct(base_score),
            "adapter_correct":is_correct(adapter_score),
            "length_delta":   adapter_wc - base_wc,        # negative = adapter shorter
            "score_delta":    adapter_score - base_score,  # positive = adapter better
            "base_vs_gold":   base_wc - gold_wc,           # how verbose vs gold
            "adapter_vs_gold":adapter_wc - gold_wc,
        })

    return results


def print_analysis(results, model, entries):
    print(f"\n{'='*65}")
    print(f"  LENGTH ANALYSIS — {model.upper()}")
    print(f"{'='*65}")

    base_wc    = [r["base_wc"]    for r in results]
    adapter_wc = [r["adapter_wc"] for r in results]
    gold_wc    = [r["gold_wc"]    for r in results]

    # ── 1. Basic length statistics ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PART 1: Answer Length Statistics (words)")
    print(f"{'─'*65}")
    print(f"  {'Metric':<25} {'Gold':>10} {'Base':>10} {'Adapter':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Mean words':<25} {np.mean(gold_wc):>9.1f}  "
          f"{np.mean(base_wc):>9.1f}  {np.mean(adapter_wc):>9.1f}")
    print(f"  {'Median words':<25} {np.median(gold_wc):>9.1f}  "
          f"{np.median(base_wc):>9.1f}  {np.median(adapter_wc):>9.1f}")
    print(f"  {'Std words':<25} {np.std(gold_wc):>9.1f}  "
          f"{np.std(base_wc):>9.1f}  {np.std(adapter_wc):>9.1f}")
    print(f"  {'P95 words':<25} {np.percentile(gold_wc,95):>9.1f}  "
          f"{np.percentile(base_wc,95):>9.1f}  "
          f"{np.percentile(adapter_wc,95):>9.1f}")

    adapter_shorter = sum(1 for r in results if r["adapter_wc"] < r["base_wc"])
    adapter_same    = sum(1 for r in results if r["adapter_wc"] == r["base_wc"])
    adapter_longer  = sum(1 for r in results if r["adapter_wc"] > r["base_wc"])
    n = len(results)

    print(f"\n  Adapter vs Base length:")
    print(f"    Adapter shorter : {adapter_shorter} ({100*adapter_shorter/n:.1f}%)")
    print(f"    Same length     : {adapter_same}    ({100*adapter_same/n:.1f}%)")
    print(f"    Adapter longer  : {adapter_longer}  ({100*adapter_longer/n:.1f}%)")
    print(f"    Mean length delta: {np.mean([r['length_delta'] for r in results]):.1f} words")

    print(f"\n  Closeness to gold answer length:")
    base_dist    = np.mean([abs(r["base_vs_gold"])    for r in results])
    adapter_dist = np.mean([abs(r["adapter_vs_gold"]) for r in results])
    print(f"    Base mean |length - gold|   : {base_dist:.1f} words")
    print(f"    Adapter mean |length - gold|: {adapter_dist:.1f} words")
    closer = "Adapter" if adapter_dist < base_dist else "Base"
    print(f"    → {closer} answers are closer to gold answer length")

    # ── 2. Correlation: length delta vs score delta ────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PART 2: Correlation — Length Change vs Score Change")
    print(f"{'─'*65}")

    length_deltas = [r["length_delta"] for r in results]
    score_deltas  = [r["score_delta"]  for r in results]

    # Pearson correlation
    corr = np.corrcoef(length_deltas, score_deltas)[0, 1]
    print(f"  Pearson r(length_delta, score_delta) = {corr:.3f}")

    if abs(corr) < 0.1:
        interp = "negligible — length change does NOT predict score change"
    elif abs(corr) < 0.3:
        interp = "weak — length change has minor relationship to score change"
    elif abs(corr) < 0.5:
        interp = "moderate — length change partially explains score change"
    else:
        interp = "strong — length change is a major driver of score change"

    sign = "negative" if corr < 0 else "positive"
    print(f"  Correlation is {sign} and {interp}")

    if corr < -0.1:
        print(f"  → Shorter answers tend to score higher (verbosity penalty exists)")
    elif corr > 0.1:
        print(f"  → Longer answers tend to score higher (detail is rewarded)")
    else:
        print(f"  → Length change is NOT a significant driver of score improvement")

    # ── 3. Score by length bucket ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PART 3: Accuracy by Answer Length Bucket")
    print(f"{'─'*65}")
    print(f"  {'Length bucket':<20} {'Base acc':>10} {'Adapter acc':>12} {'N':>6}")
    print(f"  {'-'*52}")

    buckets = [(1,3,"1-3 words"), (4,10,"4-10 words"),
               (11,25,"11-25 words"), (26,50,"26-50 words"),
               (51,999,"51+ words")]

    for lo, hi, label in buckets:
        base_in_bucket    = [r for r in results if lo <= r["base_wc"]    <= hi]
        adapter_in_bucket = [r for r in results if lo <= r["adapter_wc"] <= hi]

        if base_in_bucket:
            base_acc = np.mean([r["base_correct"] for r in base_in_bucket])
        else:
            base_acc = float("nan")

        if adapter_in_bucket:
            adapter_acc = np.mean([r["adapter_correct"] for r in adapter_in_bucket])
        else:
            adapter_acc = float("nan")

        print(f"  {label:<20} {100*base_acc:>9.1f}% {100*adapter_acc:>11.1f}% "
              f"{len(base_in_bucket):>6}")

    # ── 4. Same-length subset analysis ────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PART 4: Same-Length Subset Analysis")
    print(f"  (Questions where adapter and base give SAME word count)")
    print(f"{'─'*65}")

    same_len = [r for r in results if r["adapter_wc"] == r["base_wc"]]
    if same_len:
        base_acc_same    = np.mean([r["base_correct"]    for r in same_len])
        adapter_acc_same = np.mean([r["adapter_correct"] for r in same_len])
        print(f"  N (same length) : {len(same_len)} / {n} ({100*len(same_len)/n:.1f}%)")
        print(f"  Base accuracy   : {100*base_acc_same:.2f}%")
        print(f"  Adapter accuracy: {100*adapter_acc_same:.2f}%")
        print(f"  Delta           : {100*(adapter_acc_same-base_acc_same):+.2f}%")
        if adapter_acc_same > base_acc_same:
            print(f"  → Adapter STILL outperforms base at same length")
            print(f"    This is evidence of better factual knowledge, not just brevity")
        else:
            print(f"  → Adapter does NOT outperform base at same length")
            print(f"    Score gain may be length-driven")

    # ── 5. Score gain stratified by length change ─────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PART 5: Score Gain Stratified by Length Change Direction")
    print(f"{'─'*65}")

    shorter = [r for r in results if r["adapter_wc"] < r["base_wc"]]
    same    = [r for r in results if r["adapter_wc"] == r["base_wc"]]
    longer  = [r for r in results if r["adapter_wc"] > r["base_wc"]]

    for group, label in [(shorter, "Adapter shorter"),
                          (same,    "Same length"),
                          (longer,  "Adapter longer")]:
        if group:
            avg_score_delta = np.mean([r["score_delta"] for r in group])
            acc_base  = np.mean([r["base_correct"]    for r in group])
            acc_adap  = np.mean([r["adapter_correct"] for r in group])
            print(f"  {label} (n={len(group)}):")
            print(f"    Avg score delta : {avg_score_delta:+.2f}")
            print(f"    Base acc        : {100*acc_base:.1f}%")
            print(f"    Adapter acc     : {100*acc_adap:.1f}%")
            print(f"    Accuracy delta  : {100*(acc_adap-acc_base):+.1f}%")
            print()

    # ── 6. Verbosity hypothesis verdict ───────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  VERDICT: Is improvement due to shorter answers?")
    print(f"{'─'*65}")

    # Key evidence
    corr_weak  = abs(corr) < 0.3
    same_positive = (len(same_len) > 0 and
                     np.mean([r["adapter_correct"] for r in same_len]) >
                     np.mean([r["base_correct"]    for r in same_len]))
    shorter_pct = 100 * adapter_shorter / n

    print(f"  Evidence 1 — Length-score correlation: r={corr:.3f} "
          f"({'weak' if corr_weak else 'strong'})")
    print(f"  Evidence 2 — Same-length accuracy gain: "
          f"{'positive' if same_positive else 'zero/negative'}")
    print(f"  Evidence 3 — Adapter shorter in {shorter_pct:.1f}% of cases")

    if corr_weak and same_positive:
        print(f"\n  CONCLUSION: Improvement is primarily due to BETTER FACTUAL KNOWLEDGE")
        print(f"  Length reduction is a side effect, not the cause.")
        print(f"  The adapter produces shorter answers AND knows more facts.")
    elif not corr_weak and not same_positive:
        print(f"\n  CONCLUSION: Improvement is likely LENGTH-DRIVEN")
        print(f"  The adapter gets higher scores mainly by producing shorter answers")
        print(f"  that match the benchmark format, not by knowing more facts.")
    else:
        print(f"\n  CONCLUSION: MIXED evidence — both length and knowledge contribute")
        print(f"  Recommend examining same-length subset more carefully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", default="logs/length_analysis.json")
    args = parser.parse_args()

    entries = load_data(args.input)
    print(f"[length_analysis] {len(entries)} complete entries")

    all_results = {}
    for model in ["llama", "mistral"]:
        results = analyze_model(entries, model)
        all_results[model] = results
        print_analysis(results, model, entries)

    # Save summary
    summary = {}
    for model, results in all_results.items():
        summary[model] = {
            "mean_base_words":    float(np.mean([r["base_wc"]    for r in results])),
            "mean_adapter_words": float(np.mean([r["adapter_wc"] for r in results])),
            "mean_gold_words":    float(np.mean([r["gold_wc"]    for r in results])),
            "length_score_corr":  float(np.corrcoef(
                [r["length_delta"] for r in results],
                [r["score_delta"]  for r in results])[0,1]),
            "adapter_shorter_pct": float(
                100 * sum(1 for r in results if r["adapter_wc"] < r["base_wc"])
                / len(results)),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[length_analysis] Summary saved to {args.output}")


if __name__ == "__main__":
    main()