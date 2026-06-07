"""
GPT-based Evaluation using LLM-as-Judge
1--10 categories
"""
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from openai import OpenAI

def load_api_key() -> str:
    key_file = Path(__file__).parent.parent.parent.parent / "gpt5-openaikey.txt"
    if not key_file.exists():
        raise FileNotFoundError(f"API key file not found: {key_file}")
    return key_file.read_text(encoding="utf-8").strip()

SYSTEM_PROMPT = """You are an expert factual accuracy evaluator with broad world knowledge. \
Evaluate answers for factual correctness using BOTH the provided reference answers AND your \
own knowledge. If a candidate answer is factually correct even if not listed in the references, \
score it accordingly. Focus ONLY on factual correctness — ignore style, length, and formatting. \
Do not favor longer responses. A concise correct answer outscores a verbose partially-incorrect one."""


def build_eval_prompt(question: str, gold_answers: list,
                      answers: dict) -> str:

    gold_str     = "\n".join(f"- {a}" for a in gold_answers)
    answers_str  = "\n".join(
        f"### {label}\n{text if text else '[empty]'}"
        for label, text in answers.items()
    )

    return f"""## Question
{question}

## Acceptable Reference Answers (any one is sufficient)
{gold_str}

## Semantic Equivalence and Leniency Rules
- Abbreviations and full forms are equivalent ("US" = "United States")
- Alternate valid spellings are equivalent
- A more specific location WITHIN a reference location is correct
  (e.g., "Giza Governorate" is part of Egypt → acceptable)
- A more specific entity that is a TYPE OF or INSTANCE OF a reference is correct
- A more specific correct answer is as good as or better than a general one
- Titles and honorifics are optional ("Dr. Smith" = "Smith")
- Reasonable paraphrases conveying the same fact are equivalent
- The answer is correct if it matches ANY ONE reference answer
- Use your world knowledge to judge if the answer is actually correct
  even if not explicitly listed in the references

## Candidate Answers
{answers_str}

## Evaluation Instructions
For EACH candidate answer independently:
1. Identify all key factual claims.
2. Check against reference answers AND your world knowledge.
3. Note fabrications and critical omissions.
4. Assign a score from 1-10 using the rubric.

## Scoring Rubric
10: Factually correct and semantically equivalent to a gold answer
 9: Correct with only trivial stylistic differences
 8: Core answer correct with additional accurate context
 7: Correct at different granularity or with minor imprecision
 6: Main facts right but one noticeable error or meaningful omission
 5: Roughly equal correct and incorrect content
 4: Some correct elements but errors outweigh accuracy
 3: Primary claims wrong with minor accurate details
 2: Core claim wrong; fundamentally misleading
 1: No correct information; irrelevant, empty, or a refusal

IMPORTANT: Use the FULL range from 1-10. A score of 5 means borderline.
Do not default to 7-8 for everything.

Respond in this exact JSON format:
{{
  "LLaMA Base":      {{"reasoning": "...", "errors": [], "score": <1-10>}},
  "LLaMA Adapter":   {{"reasoning": "...", "errors": [], "score": <1-10>}},
  "Mistral Base":    {{"reasoning": "...", "errors": [], "score": <1-10>}},
  "Mistral Adapter": {{"reasoning": "...", "errors": [], "score": <1-10>}}
}}"""

def evaluate_all(client, question: str, gold_answers: list,
                 answers: dict, retries: int = 3) -> dict:
    prompt = build_eval_prompt(question, gold_answers, answers)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model           = "gpt-5.4",
                temperature     = 0,
                response_format = {"type": "json_object"},
                messages        = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
            )
            result = json.loads(response.choices[0].message.content.strip())
            # Validate scores
            for key in result:
                if isinstance(result[key], dict) and "score" in result[key]:
                    result[key]["score"] = max(1, min(10, int(result[key]["score"])))
            return result
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ERROR: {e}")
                return {k: {"reasoning": str(e), "errors": [], "score": 0}
                        for k in answers}
            time.sleep(2 ** attempt)

def print_summary(examples: list, score_key: str, label: str):
    scores = [
        entry[score_key]
        for entry in examples
        if score_key in entry and isinstance(entry.get(score_key), (int, float))
        and entry[score_key] > 0
    ]
    if not scores:
        print(f"\n{label}: No scores found.")
        return

    total = len(scores)
    avg   = sum(scores) / total
    high  = sum(1 for s in scores if s >= 8)
    mid   = sum(1 for s in scores if 4 <= s < 8)
    low   = sum(1 for s in scores if s < 4)
    dist  = Counter(scores)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Average score     : {avg:.2f} / 10")
    print(f"  Score distribution:")
    for s in range(10, 0, -1):
        bar = "█" * dist.get(s, 0)
        print(f"    {s:2d}: {dist.get(s,0):>4}  {bar}")
    print(f"  {'-'*50}")
    print(f"  Correct    (8-10) : {high:>4} / {total} ({100*high/total:.1f}%)")
    print(f"  Partial    (4-7)  : {mid:>4}  / {total} ({100*mid/total:.1f}%)")
    print(f"  Wrong      (1-3)  : {low:>4}  / {total} ({100*low/total:.1f}%)")
    print(f"  Total             : {total}")


def print_comparison_table(examples: list):
    keys = {
        "LLaMA Base":      "llama_base_score",
        "LLaMA Adapter":   "llama_adapter_score",
        "Mistral Base":    "mistral_base_score",
        "Mistral Adapter": "mistral_adapter_score",
    }
    print(f"\n{'='*65}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Avg':>6} {'Correct%':>10} {'Partial%':>10} {'Wrong%':>10}")
    print(f"  {'-'*60}")
    for label, key in keys.items():
        scores = [e[key] for e in examples
                  if key in e and isinstance(e.get(key), (int, float)) and e[key] > 0]
        if not scores:
            continue
        total   = len(scores)
        avg     = sum(scores) / total
        correct = sum(1 for s in scores if s >= 8)
        partial = sum(1 for s in scores if 4 <= s < 8)
        wrong   = sum(1 for s in scores if s < 4)
        print(f"  {label:<20} {avg:>6.2f} {100*correct/total:>9.1f}% "
              f"{100*partial/total:>9.1f}% {100*wrong/total:>9.1f}%")
    print(f"{'='*65}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Path to combined input JSON")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--delay",  type=float, default=0.5)
    args = parser.parse_args()

    client   = OpenAI(api_key=load_api_key())
    out_path = Path(args.input).parent / "gpt_evaluated_combined-v2.json"

    with open(args.input, encoding="utf-8") as f:
        combined = json.load(f)

    for entry in combined:
        entry.setdefault("llama_base",       entry.get("llama_base",       ""))
        entry.setdefault("llama_adapter",    entry.get("llama_adapter",    ""))
        entry.setdefault("mistral_base",     entry.get("mistral_base",     ""))
        entry.setdefault("mistral_adapter",  entry.get("mistral_adapter",  ""))

    if args.limit:
        combined = combined[:args.limit]
        print(f"[gpt_eval] Running on {args.limit} samples only.")

    print(f"[gpt_eval] Evaluating {len(combined)} samples (4 answers per sample)...")

    for i, entry in enumerate(combined):

        if "llama_base_score" in entry and entry["llama_base_score"]:
            print(f"[{i:>4}] Already evaluated, skipping.")
            continue

        answers = {
            "LLaMA Base":      entry["llama_base"],
            "LLaMA Adapter":   entry["llama_adapter"],
            "Mistral Base":    entry["mistral_base"],
            "Mistral Adapter": entry["mistral_adapter"],
        }

        result = evaluate_all(
            client,
            question    = entry["question"],
            gold_answers= entry["answers"],
            answers     = answers,
        )

        for label, key_prefix in [
            ("LLaMA Base",      "llama_base"),
            ("LLaMA Adapter",   "llama_adapter"),
            ("Mistral Base",    "mistral_base"),
            ("Mistral Adapter", "mistral_adapter"),
        ]:
            r = result.get(label, {})
            entry[f"{key_prefix}_score"]  = r.get("score",     0)
            entry[f"{key_prefix}_reason"] = r.get("reasoning", "")
            entry[f"{key_prefix}_errors"] = r.get("errors",    [])

        print(f"[{i:>4}] "
              f"LB={entry['llama_base_score']:>2} "
              f"LA={entry['llama_adapter_score']:>2} "
              f"MB={entry['mistral_base_score']:>2} "
              f"MA={entry['mistral_adapter_score']:>2} "
              f"| {entry['question'][:45]}")

        time.sleep(args.delay)

        if (i + 1) % 10 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            print(f"[gpt_eval] Progress saved at sample {i+1}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\n[gpt_eval] Done. Saved to {out_path}")

    print_summary(combined, "llama_base_score",      "LLaMA Base")
    print_summary(combined, "llama_adapter_score",   "LLaMA Adapter")
    print_summary(combined, "mistral_base_score",    "Mistral Base")
    print_summary(combined, "mistral_adapter_score", "Mistral Adapter")
    print_comparison_table(combined)


if __name__ == "__main__":
    main()