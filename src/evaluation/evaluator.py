
import json
import re
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config, get_active_model_cfg
from src.training.trainer import build_model


# Text normalization

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_answer(text: str) -> str:
    for marker in ["Answer:", "Context:", "Question:", "Explanation:", "Note:"]:
        if text.startswith(marker):
            text = text[len(marker):].strip()
    text = text.split("\n")[0].strip()

    for sep in [". ", "! ", "? "]:
        if sep in text:
            text = text[:text.index(sep)].strip()

    text = text.strip('"\'')
    text = text.strip(" .,;:")
    return text


def exact_match(pred: str, golds: list) -> float:
    pred_n = normalize(pred)
    return float(any(normalize(g) == pred_n for g in golds))


def token_f1(pred: str, golds: list) -> float:
    pred_tokens = normalize(pred).split()
    best_f1 = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        prec = len(common) / len(pred_tokens) if pred_tokens else 0.0
        rec  = len(common) / len(gold_tokens) if gold_tokens else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    return best_f1


def rouge_l(pred: str, golds: list) -> float:
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] \
                            else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    pred_tokens = normalize(pred).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        l    = lcs(pred_tokens, gold_tokens)
        prec = l / len(pred_tokens) if pred_tokens else 0.0
        rec  = l / len(gold_tokens) if gold_tokens else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best = max(best, f1)
    return best


# Context builder

def build_context(triples: list, labels: dict) -> str:
    if not triples:
        return ""
    seen, parts = set(), []
    for s, p, o in triples:
        stmt = f"{labels.get(s, s)} {labels.get(p, p)} {labels.get(o, o)}"
        if stmt not in seen:
            seen.add(stmt)
            parts.append(stmt)
    return ". ".join(parts) + "."


def build_prompt(question: str, triples: list, labels: dict) -> str:
    context = build_context(triples, labels)
    if context:
        return f"Context: {context}\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


# Generation

def generate_answer(model, tokenizer, prompt: str,
                    device: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=480).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_answer_with_adapter(model, tokenizer, prompt: str,
                                  subj_ids, pred_ids, obj_ids,
                                  triple_mask, has_kg,
                                  device: str, max_new_tokens: int = 32) -> str:
    inputs    = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=480).to(device)
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    B         = input_ids.size(0)

    with torch.no_grad():
        # Compute KG bias and set on model
        kg_emb    = model.kg_encoder(subj_ids, pred_ids, obj_ids, triple_mask)
        kg_tokens = model.kg_projector(kg_emb) * model.kg_scale
        kg_tokens = kg_tokens * has_kg.float().view(B, 1, 1)
        kg_flat   = kg_tokens.reshape(B, -1)
        kg_bias   = model.kg_to_hidden(kg_flat).unsqueeze(1)
        model._kg_bias = kg_bias

        output = model.llm.generate(
            input_ids      = input_ids,
            attention_mask = attn_mask,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )
        model._kg_bias = None

    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Evaluation loop

def evaluate_split(cfg: dict, adapter_path: str = None, num_samples: int = None):
    device    = cfg["training"]["device"]
    model_cfg = get_active_model_cfg(cfg)
    eval_cfg  = cfg["evaluation"]
    proc      = cfg["paths"]["processed_dir"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["local_path"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    with open(str(Path(proc) / "test_with_triples.json")) as f:
        samples = json.load(f)
    with open(str(Path(proc) / "vocab.json")) as f:
        vocab = json.load(f)
    with open(str(Path(proc) / "labels.json")) as f:
        labels = json.load(f)

    if num_samples is not None:
        samples = samples[:num_samples]
        print(f"[eval] Running on {num_samples} samples only.")

    entity2id   = vocab["entity2id"]
    relation2id = vocab["relation2id"]
    max_new_tokens = eval_cfg["max_new_tokens"]

    # Load base LLM
    print("[eval] Loading base LLM...")
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_cfg["local_path"],
        local_files_only = True,
        dtype            = torch.bfloat16,
        device_map       = device,
    )
    base_llm.eval()

    # Load adapter model
    print("[eval] Loading adapter model...")
    adapter_model = build_model(cfg, device)
    adapter_model.load_adapter(
        adapter_path or str(Path(cfg["paths"]["checkpoint_dir"]) / "best"))
    adapter_model.eval()

    base_results    = []
    adapter_results = []

    for sample in tqdm(samples, desc="Evaluating"):
        answers = sample["answers"]
        triples = sample["triples"]

        base_prompt    = f"Question: {sample['question']}\nAnswer:"

        adapter_prompt = build_prompt(sample["question"], triples, labels)

        base_pred    = clean_answer(generate_answer(
            base_llm, tokenizer, base_prompt, device, max_new_tokens))

        if triples:
            subj_ids = torch.tensor(
                [[entity2id.get(t[0], 0) for t in triples]],
                dtype=torch.long).to(device)
            pred_ids = torch.tensor(
                [[relation2id.get(t[1], 0) for t in triples]],
                dtype=torch.long).to(device)
            obj_ids  = torch.tensor(
                [[entity2id.get(t[2], 0) for t in triples]],
                dtype=torch.long).to(device)
            t_mask   = torch.ones(1, len(triples), dtype=torch.long).to(device)
            has_kg   = torch.tensor([True],  dtype=torch.bool).to(device)
        else:
            subj_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
            pred_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
            obj_ids  = torch.zeros(1, 1, dtype=torch.long).to(device)
            t_mask   = torch.zeros(1, 1, dtype=torch.long).to(device)
            has_kg   = torch.tensor([False], dtype=torch.bool).to(device)

        adapter_pred = clean_answer(generate_answer_with_adapter(
            adapter_model, tokenizer, adapter_prompt,
            subj_ids, pred_ids, obj_ids, t_mask, has_kg,
            device, max_new_tokens))

        base_results.append({
            "id":       sample["id"],
            "question": sample["question"],
            "answers":  answers,
            "pred":     base_pred,
            "em":       exact_match(base_pred, answers),
            "f1":       token_f1(base_pred, answers),
            "rouge_l":  rouge_l(base_pred, answers),
        })
        adapter_results.append({
            "id":       sample["id"],
            "question": sample["question"],
            "answers":  answers,
            "pred":     adapter_pred,
            "em":       exact_match(adapter_pred, answers),
            "f1":       token_f1(adapter_pred, answers),
            "rouge_l":  rouge_l(adapter_pred, answers),
        })

    def avg(results, key):
        return sum(r[key] for r in results) / max(len(results), 1)

    print("\n" + "="*60)
    print(f"{'Metric':<15} {'Base LLM':>12} {'KG Adapter':>12} {'Delta':>10}")
    print("="*60)
    for metric in ["em", "f1", "rouge_l"]:
        base_score    = avg(base_results,    metric)
        adapter_score = avg(adapter_results, metric)
        delta         = adapter_score - base_score
        label         = {"em":"Exact Match","f1":"F1","rouge_l":"ROUGE-L"}[metric]
        print(f"{label:<15} {base_score:>12.4f} {adapter_score:>12.4f} {delta:>+10.4f}")
    print("="*60)

    # Save
    # out_dir = Path(cfg["paths"]["log_dir"])
    # out_dir.mkdir(parents=True, exist_ok=True)
    # suffix = f"_{num_samples}" if num_samples else ""
    # with open(out_dir / f"base_results{suffix}.json", "w") as f:
    #     json.dump(base_results, f, indent=2)
    # with open(out_dir / f"adapter_results{suffix}.json", "w") as f:
    #     json.dump(adapter_results, f, indent=2)
    # print(f"\n[eval] Results saved to {out_dir}")

    # ── Save answers only ────────────────────────────────────────────────
    combined_results = []

    for b, a in zip(base_results, adapter_results):
        combined_results.append({
            "id": b["id"],
            "question": b["question"],
            "answers": b["answers"],
            "base": b["pred"],
            "with_kg": a["pred"]
        })

    out_dir = Path(cfg["paths"]["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{num_samples}" if num_samples else ""
    out_file = out_dir / f"answers{suffix}.json"

    with open(out_file, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n[eval] Answers saved to {out_file}")

    n_preview = len(samples) if num_samples and num_samples <= 20 else 5
    print("\n[eval] Sample predictions:")
    for i in range(min(n_preview, len(samples))):
        print(f"\n  Q       : {base_results[i]['question']}")
        print(f"  Answers : {base_results[i]['answers']}")
        print(f"  Base    : {base_results[i]['pred']}")
        print(f"  Adapter : {adapter_results[i]['pred']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--adapter",     default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    cfg  = load_config(args.config)
    evaluate_split(cfg, args.adapter, args.num_samples)