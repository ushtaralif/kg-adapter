"""
GPT-Based Evaluation Methods
1--7 Categories Evaluation
Hallucination Detection
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

MODEL_NAME = "gpt-5.4"
SLEEP_BETWEEN_CALLS = 0.0
MAX_RETRIES = 3

MODELS = {
    "llama_base":      "llama_base",
    "llama_adapter":   "llama_adapter",
    "mistral_base":    "mistral_base",
    "mistral_adapter": "mistral_adapter",
}



try:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(parent_dir, "../../../", "gpt5-openaikey.txt")
    with open(key_file, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'gpt5-openaikey.txt' was not found in the parent directory."
    )

client = OpenAI(api_key=api_key)


if len(sys.argv) < 2:
    raise ValueError("Usage: python script.py <input_json_file>")

input_json_file = sys.argv[1]

with open(input_json_file, "r", encoding="utf-8") as f:
    loaded = json.load(f)

if isinstance(loaded, dict) and "examples" in loaded:
    examples = loaded["examples"]
else:
    examples = loaded

if not isinstance(examples, list):
    raise ValueError("Input JSON must be a list or a dict containing an 'examples' list.")


def normalize_gold_answer(gold: Any) -> str:

    if gold is None:
        return ""

    if isinstance(gold, str):
        return gold.strip()

    if isinstance(gold, list):
        vals = [str(x).strip() for x in gold if str(x).strip()]
        return "; ".join(vals)

    if isinstance(gold, dict):
        if "gold_answers" in gold and isinstance(gold["gold_answers"], list):
            vals = [str(x).strip() for x in gold["gold_answers"] if str(x).strip()]
            return "; ".join(vals)

        parts = []
        for k, v in gold.items():
            if isinstance(v, list):
                vv = "; ".join(str(x).strip() for x in v if str(x).strip())
                parts.append(f"{k}: {vv}")
            else:
                parts.append(f"{k}: {str(v).strip()}")
        return " | ".join(parts)

    return str(gold).strip()


def get_question(entry: Dict[str, Any]) -> str:
    for k in ("question", "Question", "corrected_question"):
        if k in entry and entry[k]:
            return str(entry[k]).strip()
    return ""


def get_gold(entry: Dict[str, Any]) -> str:
    for k in ("answers", "gold_answers", "Answer", "answer"):
        if k in entry:
            return normalize_gold_answer(entry[k])
    return ""


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def response_text(resp) -> str:

    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    # Fallbacks
    try:
        output = getattr(resp, "output", None) or []
        chunks = []
        for item in output:
            content = getattr(item, "content", None) or []
            for c in content:
                if getattr(c, "type", "") == "output_text":
                    chunks.append(getattr(c, "text", ""))
        return "".join(chunks).strip()
    except Exception:
        return ""


def call_json_response(
    prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
    system_prompt: str
) -> Dict[str, Any]:

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )

            txt = response_text(resp)
            if not txt:
                raise ValueError("Empty response text from model.")
            return json.loads(txt)

        except Exception as e:
            last_err = e
            print(f"[WARN] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(1.0 * attempt)

    raise RuntimeError(f"All retries failed. Last error: {last_err}")


SINGLE_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "label_id": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5, 6, 7]
        },
        "label": {
            "type": "string",
            "enum": [
                "Perfectly Accurate",
                "Very accurate",
                "Accurate but with some redundant or irrelevant information",
                "Inaccurate (probable intrinsic error e.g from inaccurate/out-of-date training data)",
                "Inaccurate (probable fabrication/hallucination)",
                "Inaccurate (irrelevant)",
                "Completely inaccurate"
            ]
        }
    },
    "required": ["label_id", "label"],
    "additionalProperties": False
}

HALLUC_SCHEMA = {
    "type": "object",
    "properties": {
        "hallucination_label": {
            "type": "string",
            "enum": ["FACTUAL", "PARTIAL", "HALLUCINATION"]
        }
    },
    "required": ["hallucination_label"],
    "additionalProperties": False
}

def build_single_eval_prompt(question: str, gold: str, answer: str) -> str:
    return f"""Question: {question}
Correct Answer: {gold}
Model Answer: {answer}

Classify the model answer into exactly one of the following categories:

1. Perfectly Accurate
2. Very accurate
3. Accurate but with some redundant or irrelevant information
4. Inaccurate (probable intrinsic error e.g from inaccurate/out-of-date training data)
5. Inaccurate (probable fabrication/hallucination)
6. Inaccurate (irrelevant)
7. Completely inaccurate

Respond with only the classification number and its label.
"""

def build_halluc_prompt(question: str, gold: str, answer: str) -> str:
    return f"""Question: {question}
Gold Answer: {gold}
Model Answer: {answer}

Determine whether the model answer contains hallucinated information.

Definitions:
- FACTUAL: answer matches the gold fact
- PARTIAL: contains a correct fact but also redundant, vague, or partially mismatched content
- HALLUCINATION: contains fabricated or incorrect information

Return one label:
- FACTUAL
- PARTIAL
- HALLUCINATION
"""

SYSTEM_SINGLE = "You are an expert evaluator of factual correctness."
SYSTEM_HALLUC = "You are an expert evaluator of hallucination in model answers."


def evaluate_single(question: str, gold: str, answer: str) -> Dict[str, Any]:
    prompt = build_single_eval_prompt(question, gold, answer)
    return call_json_response(
        prompt=prompt,
        schema_name="single_eval",
        schema=SINGLE_EVAL_SCHEMA,
        system_prompt=SYSTEM_SINGLE
    )

def evaluate_halluc(question: str, gold: str, answer: str) -> Dict[str, Any]:
    prompt = build_halluc_prompt(question, gold, answer)
    return call_json_response(
        prompt=prompt,
        schema_name="halluc_eval",
        schema=HALLUC_SCHEMA,
        system_prompt=SYSTEM_HALLUC
    )


for idx, entry in enumerate(examples):
    question = get_question(entry)
    gold = get_gold(entry)

    if not question or not gold:
        print(f"[WARN] skipping idx={idx} due to missing question/gold")
        continue

    for model_name, answer_key in MODELS.items():
        model_answer = safe_text(entry.get(answer_key, ""))

        if not model_answer:
            print(f"[WARN] idx={idx} missing answer for {model_name}")
            continue

        single_key = f"{model_name}_single_eval"
        halluc_key = f"{model_name}_halluc_eval"

        if single_key not in entry or not entry[single_key]:
            entry[single_key] = evaluate_single(question, gold, model_answer)
            print(f"[OK] idx={idx} {single_key}: {entry[single_key]}")
            if SLEEP_BETWEEN_CALLS:
                time.sleep(SLEEP_BETWEEN_CALLS)

        if halluc_key not in entry or not entry[halluc_key]:
            entry[halluc_key] = evaluate_halluc(question, gold, model_answer)
            print(f"[OK] idx={idx} {halluc_key}: {entry[halluc_key]}")
            if SLEEP_BETWEEN_CALLS:
                time.sleep(SLEEP_BETWEEN_CALLS)

output_file = "evaluated_" + os.path.basename(input_json_file)

if isinstance(loaded, dict) and "examples" in loaded:
    loaded["examples"] = examples
    save_obj = loaded
else:
    save_obj = examples

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(save_obj, f, indent=2, ensure_ascii=False)

print(f"Evaluated answers saved to {output_file}")