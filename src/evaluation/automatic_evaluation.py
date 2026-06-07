import json
import re
from collections import Counter

# =========================
# CONFIG
# =========================
INPUT_FILE  = "your_input_file.json"
OUTPUT_FILE = "your_output_file.json"


# =========================
# SCORING FUNCTIONS
# =========================

def normalize(text):
    """Lowercase, remove punctuation, strip whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def substring_match(prediction, answers):
    pred_norm = normalize(prediction)
    return int(any(normalize(ans) in pred_norm for ans in answers))


def token_f1(prediction, answers):

    pred_tokens = normalize(prediction).split()
    best_f1 = 0.0

    for ans in answers:
        ans_tokens   = normalize(ans).split()
        pred_counter = Counter(pred_tokens)
        ans_counter  = Counter(ans_tokens)
        common       = sum((pred_counter & ans_counter).values())

        if common == 0:
            continue

        precision = common / len(pred_tokens)
        recall    = common / len(ans_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)

    return round(best_f1, 4)


def lcs_length(a, b):
    """Length of Longest Common Subsequence of two token lists."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]

def answer_in_prediction(prediction, answers):

    pred_tokens = normalize(prediction).split()
    for ans in answers:
        ans_tokens = normalize(ans).split()
        # Check if ans_tokens appear as a contiguous subsequence in pred_tokens
        ans_len = len(ans_tokens)
        for j in range(len(pred_tokens) - ans_len + 1):
            if pred_tokens[j:j + ans_len] == ans_tokens:
                return 1
    return 0


def score_all(prediction, answers):
    """Return all metrics for a prediction against reference answers."""
    if not prediction or not answers:
        return {
            "substring":  0,
            "token_match": 0
        }
    return {
        "substring":   substring_match(prediction, answers),   # TriviaQA-style
        "token_match": answer_in_prediction(prediction, answers),  # KGQA-style
    }


# =========================
# MAIN
# =========================

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    results = []

    METRICS = ["substring", "token_match"]
    MODELS  = ["llama_base", "llama_adapter", "mistral_base", "mistral_adapter"]

    totals = {model: Counter() for model in MODELS}
    n = 0

    for i, item in enumerate(data):
        answers = item.get("answers", [])
        scores  = {}

        for model in MODELS:
            prediction   = item.get(model, "")
            model_scores = score_all(prediction, answers)

            for metric in METRICS:
                scores[f"{model}_{metric}"] = model_scores[metric]
                totals[model][metric]       += model_scores[metric]

        results.append({**item, **scores})
        n += 1

    # =========================
    # SUMMARY
    # =========================

    print(f"\n===== SUMMARY (n={n}) =====")
    print(f"{'Model':<20}  {'Substr':>8} {'TokMatch':>10}")
    print("-" * 68)

    for model in MODELS:
        avgs = {m: totals[model][m] / n for m in METRICS}
        print(
            f"{model:<20} "
            f"{avgs['substring']:>8.4f} "
            f"{avgs['token_match']:>10.4f} "
        )

    print(f"\nNotes:")
    print(f"  Substr     : gold answer is substring of prediction (TriviaQA-style)")
    print(f"  TokMatch   : gold answer tokens appear contiguously in prediction (KGQA-style)")


    # =========================
    # SAVE
    # =========================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()