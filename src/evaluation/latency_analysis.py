
import argparse
import copy
import json
import time
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config
from src.training.trainer import build_model


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def print_results(results):
    base  = results["base"]
    adap  = results["adapter"]
    model = results["model"].upper()

    print(f"\n{'='*65}")
    print(f"  {model} — LATENCY RESULTS (n={results['n_samples']})")
    print(f"{'='*65}")

    overhead_ms  = adap["total_mean"] - base["mean"]
    overhead_pct = 100 * overhead_ms / max(base["mean"], 1)
    kg_enc_pct   = 100 * adap["kg_enc_mean"] / max(adap["total_mean"], 1)
    p50_overhead = adap["total_p50"] - base["p50"]

    print(f"\n  {'Metric':<30} {'Base LLM':>12} {'KG Adapter':>12} {'Overhead':>12}")
    print(f"  {'-'*68}")
    print(f"  {'Mean (ms)':<30} {base['mean']:>11.1f}  {adap['total_mean']:>11.1f}  {overhead_ms:>+10.1f}")
    print(f"  {'Std  (ms)':<30} {base['std']:>11.1f}  {adap['total_std']:>11.1f}")
    print(f"  {'P50  (ms)':<30} {base['p50']:>11.1f}  {adap['total_p50']:>11.1f}  {p50_overhead:>+10.1f}")
    print(f"  {'P95  (ms)':<30} {base['p95']:>11.1f}  {adap['total_p95']:>11.1f}  {adap['total_p95']-base['p95']:>+10.1f}")
    print(f"  {'P99  (ms)':<30} {base['p99']:>11.1f}  {adap['total_p99']:>11.1f}  {adap['total_p99']-base['p99']:>+10.1f}")

    print(f"\n  Mean overhead : {overhead_ms:.1f} ms/query ({overhead_pct:.2f}%)")
    print(f"  P50  overhead : {p50_overhead:.1f} ms  ← more reliable than mean")
    print(f"  KG encoding   : {adap['kg_enc_mean']:.2f} ms ({kg_enc_pct:.3f}% of total adapter time)")
    print(f"  LLM generation: {adap['llm_mean']:.1f} ms ({100-kg_enc_pct:.3f}% of total adapter time)")

    print(f"\n  GPU memory — Base: {results['mem_base_mb']:.0f} MB  "
          f"Adapter: {results['mem_adapter_mb']:.0f} MB  "
          f"Delta: {results['mem_adapter_mb']-results['mem_base_mb']:+.0f} MB")
    print(f"  Trainable params: {results['adapter_params']:,} "
          f"({100*results['adapter_params']/max(results['base_params'],1):.3f}% of LLM)")

    # GPU-hours projection using P50 overhead (cleaner than mean)
    kg_only_ms   = adap["kg_enc_mean"]
    hr_kg_only   = (kg_only_ms   * 1_000_000 / 1000) / 3600
    hr_p50_oh    = (p50_overhead * 1_000_000 / 1000) / 3600
    hr_base      = (base["mean"] * 1_000_000 / 1000) / 3600
    hr_adap      = (adap["total_mean"] * 1_000_000 / 1000) / 3600

    print(f"\n  GPU-hours / 1M queries:")
    print(f"    Base LLM          : {hr_base:.2f}h  (${hr_base*3.5:.2f} @ A100 $3.50/hr)")
    print(f"    KG Adapter (mean) : {hr_adap:.2f}h  (${hr_adap*3.5:.2f} @ A100)")
    print(f"    P50 overhead      : {hr_p50_oh:.4f}h (${hr_p50_oh*3.5:.4f} @ A100)")
    print(f"    KG encoding only  : {hr_kg_only:.4f}h (${hr_kg_only*3.5:.4f} @ A100)")
    print(f"    → True KG cost: {kg_only_ms:.2f} ms/query = ${hr_kg_only*3.5:.4f} per 1M queries")


def run_for_model(cfg, active, args, device):
    proc      = cfg["paths"]["processed_dir"]
    model_cfg = cfg["models"][active]

    print(f"\n{'#'*65}")
    print(f"  RUNNING: {active.upper()}")
    print(f"{'#'*65}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["local_path"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(str(Path(proc) / "test_with_triples.json")) as f:
        samples = json.load(f)
    with open(str(Path(proc) / "vocab.json")) as f:
        vocab = json.load(f)
    with open(str(Path(proc) / "labels.json")) as f:
        labels = json.load(f)

    samples     = samples[:args.n_samples + args.n_warmup]
    entity2id   = vocab["entity2id"]
    relation2id = vocab["relation2id"]

    def make_tensors(triples):
        if triples:
            s = torch.tensor([[entity2id.get(t[0], 0)   for t in triples]], dtype=torch.long).to(device)
            p = torch.tensor([[relation2id.get(t[1], 0) for t in triples]], dtype=torch.long).to(device)
            o = torch.tensor([[entity2id.get(t[2], 0)   for t in triples]], dtype=torch.long).to(device)
            m  = torch.ones(1, len(triples), dtype=torch.long).to(device)
            hk = torch.tensor([True], dtype=torch.bool).to(device)
        else:
            s = p = o = torch.zeros(1, 1, dtype=torch.long).to(device)
            m  = torch.zeros(1, 1, dtype=torch.long).to(device)
            hk = torch.tensor([False], dtype=torch.bool).to(device)
        return s, p, o, m, hk

    def make_inputs(question, triples):
        context = ""
        if triples:
            parts = [f"{labels.get(s,s)} {labels.get(p,p)} {labels.get(o,o)}"
                     for s, p, o in triples]
            context = ". ".join(parts) + "."
        prompt = (f"Context: {context}\nQuestion: {question}\nAnswer:"
                  if context else f"Question: {question}\nAnswer:")
        return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)

    # ── Load base LLM ─────────────────────────────────────────────────────────
    print(f"[{active}] Loading base LLM from {model_cfg['local_path']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["local_path"], local_files_only=True,
        dtype=torch.bfloat16, device_map=device)
    base_model.eval()
    base_params = sum(p.numel() for p in base_model.parameters())
    if torch.cuda.is_available(): torch.cuda.synchronize()
    mem_base_mb = get_gpu_memory_mb()
    print(f"[{active}] Base loaded. Memory: {mem_base_mb:.0f} MB, Params: {base_params:,}")

    # ── Load KG Adapter ───────────────────────────────────────────────────────
    print(f"[{active}] Loading KG Adapter...")
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["models"]["active"] = active
    adapter_model = build_model(cfg_copy, device)
    ckpt_base = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_path = ckpt_base / "mistral" / "best" if active == "mistral" else ckpt_base / "best"
    print(f"[{active}] Checkpoint: {ckpt_path}")
    adapter_model.load_adapter(str(ckpt_path), name=f"{active}_adapter")
    adapter_model.eval()
    adapter_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    mem_adapter_mb = get_gpu_memory_mb()
    print(f"[{active}] Adapter loaded. Memory: {mem_adapter_mb:.0f} MB, "
          f"Trainable: {adapter_params:,}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"[{active}] Warming up ({args.n_warmup} queries)...")
    for s in samples[:args.n_warmup]:
        inp = make_inputs(s["question"], s["triples"])
        with torch.no_grad():
            base_model.generate(
                input_ids=inp["input_ids"].to(device),
                attention_mask=inp["attention_mask"].to(device),
                max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)

    # ── Measure ───────────────────────────────────────────────────────────────
    measure_samples = samples[args.n_warmup: args.n_warmup + args.n_samples]
    print(f"[{active}] Measuring {len(measure_samples)} queries "
          f"({args.n_runs} runs each)...")

    base_times, total_times, kg_enc_times, llm_times = [], [], [], []

    for s in tqdm(measure_samples, desc=f"  {active}"):
        inp       = make_inputs(s["question"], s["triples"])
        input_ids = inp["input_ids"].to(device)
        attn_mask = inp["attention_mask"].to(device)
        subj, pred, obj, mask, has_kg = make_tensors(s["triples"])

        # Base
        bt = []
        for _ in range(args.n_runs):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                base_model.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            bt.append((time.perf_counter() - t0) * 1000)
        base_times.append(float(np.median(bt)))

        # Adapter
        at_list, kt_list, lt_list = [], [], []
        for _ in range(args.n_runs):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_kg0 = time.perf_counter()
            with torch.no_grad():
                B = input_ids.size(0)
                kg_emb    = adapter_model.kg_encoder(subj, pred, obj, mask)
                kg_tokens = adapter_model.kg_projector(kg_emb) * adapter_model.kg_scale
                kg_tokens = kg_tokens * has_kg.float().view(B, 1, 1)
                kg_flat   = kg_tokens.reshape(B, -1)
                kg_bias   = adapter_model.kg_to_hidden(kg_flat).unsqueeze(1)
                adapter_model._kg_bias = kg_bias
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_kg1 = time.perf_counter()
            with torch.no_grad():
                adapter_model.llm.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
                adapter_model._kg_bias = None
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_llm1 = time.perf_counter()
            kt_list.append((t_kg1  - t_kg0)  * 1000)
            lt_list.append((t_llm1 - t_kg1)  * 1000)
            at_list.append((t_llm1 - t_kg0)  * 1000)

        total_times.append(float(np.median(at_list)))
        kg_enc_times.append(float(np.median(kt_list)))
        llm_times.append(float(np.median(lt_list)))

    results = {
        "model":          active,
        "n_samples":      len(measure_samples),
        "base_params":    base_params,
        "adapter_params": adapter_params,
        "mem_base_mb":    mem_base_mb,
        "mem_adapter_mb": mem_adapter_mb,
        "base": {
            "mean": float(np.mean(base_times)),
            "std":  float(np.std(base_times)),
            "p50":  float(np.percentile(base_times, 50)),
            "p95":  float(np.percentile(base_times, 95)),
            "p99":  float(np.percentile(base_times, 99)),
        },
        "adapter": {
            "total_mean":  float(np.mean(total_times)),
            "total_std":   float(np.std(total_times)),
            "total_p50":   float(np.percentile(total_times, 50)),
            "total_p95":   float(np.percentile(total_times, 95)),
            "total_p99":   float(np.percentile(total_times, 99)),
            "kg_enc_mean": float(np.mean(kg_enc_times)),
            "kg_enc_std":  float(np.std(kg_enc_times)),
            "llm_mean":    float(np.mean(llm_times)),
            "llm_std":     float(np.std(llm_times)),
        },
    }

    print_results(results)

    # Free GPU before next model
    del base_model, adapter_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--n_warmup",   type=int, default=10)
    parser.add_argument("--n_runs",     type=int, default=3)
    parser.add_argument("--output_dir", default="src/evaluation")
    parser.add_argument("--models",     nargs="+", default=["mistral", "llama"],
                        help="Models to run. Default: both")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = cfg["training"]["device"]
    print(f"[latency] Device: {device}")
    print(f"[latency] Models: {args.models}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for active in args.models:
        if active not in cfg["models"]:
            print(f"[latency] Skipping '{active}' — not found in config models block")
            continue
        results = run_for_model(cfg, active, args, device)
        all_results[active] = results
        out_path = out_dir / f"latency_{active}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[latency] Saved: {out_path}")

    # Combined summary
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f"  COMBINED SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Metric':<32} {'LLaMA':>14} {'Mistral':>14}")
        print(f"  {'-'*62}")

        def safe(d, key, default=float("nan")):
            return d.get(key, default) if d else default

        rows = [
            ("Base LLM mean (ms)",         lambda r: r["base"]["mean"]),
            ("Adapter mean (ms)",           lambda r: r["adapter"]["total_mean"]),
            ("Mean overhead (ms)",          lambda r: r["adapter"]["total_mean"] - r["base"]["mean"]),
            ("P50 overhead (ms)",           lambda r: r["adapter"]["total_p50"]  - r["base"]["p50"]),
            ("Mean overhead (%)",           lambda r: 100*(r["adapter"]["total_mean"]-r["base"]["mean"])/max(r["base"]["mean"],1)),
            ("KG encoding mean (ms)",       lambda r: r["adapter"]["kg_enc_mean"]),
            ("Adapter trainable params",    lambda r: r["adapter_params"]),
        ]
        for label, fn in rows:
            ll = fn(all_results["llama"])   if "llama"   in all_results else float("nan")
            mi = fn(all_results["mistral"]) if "mistral" in all_results else float("nan")
            if "params" in label:
                print(f"  {label:<32} {ll:>13,.0f}  {mi:>13,.0f}")
            elif "%" in label:
                print(f"  {label:<32} {ll:>13.2f}%  {mi:>13.2f}%")
            else:
                print(f"  {label:<32} {ll:>13.2f}   {mi:>13.2f}")

    combined_path = out_dir / "latency_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[latency] Combined saved: {combined_path}")


if __name__ == "__main__":
    main()











# """
# Latency and Computational Overhead Analysis
# --------------------------------------------
# Measures per-query latency for Base LLM vs KG Adapter.
# Reports:
#   - Avg latency per query (ms)
#   - Latency overhead from KG injection
#   - GPU memory usage
#   - Estimated GPU-hours per 1M queries
#   - Breakdown: KG encoding time vs LLM generation time
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 python -m src.evaluation.latency_analysis \
#     --config configs/config.yaml \
#     --n_samples 100 \
#     --n_warmup 10 \
#     --output src/evaluation/latency_analysis.json
# """
# import argparse
# import json
# import time
# import sys
# import torch
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from src.config import load_config, get_active_model_cfg
# from src.training.trainer import build_model
#
#
# def get_gpu_memory_mb():
#     if torch.cuda.is_available():
#         return torch.cuda.memory_allocated() / 1024**2
#     return 0.0
#
#
# def measure_base_latency(model, tokenizer, batch, device, n_runs=3):
#     """Measure base LLM generation latency (no KG)."""
#     times = []
#     with torch.no_grad():
#         for _ in range(n_runs):
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t0 = time.perf_counter()
#             _ = model.generate(
#                 input_ids      = batch["input_ids"].to(device),
#                 attention_mask = batch["attention_mask"].to(device),
#                 max_new_tokens = 64,
#                 do_sample      = False,
#             )
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             times.append((time.perf_counter() - t0) * 1000)  # ms
#     return times
#
#
# def measure_adapter_latency(adapter, batch, device, n_runs=3):
#     """Measure full KG adapter latency — includes KG encoding + LLM generation."""
#     times_total  = []
#     times_kg_enc = []
#     times_llm    = []
#
#     with torch.no_grad():
#         for _ in range(n_runs):
#             b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
#                  for k, v in batch.items()}
#
#             # ── KG encoding time ──────────────────────────────────────────────
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_kg_start = time.perf_counter()
#
#             subj = b["subj_ids"]
#             pred = b["pred_ids"]
#             obj  = b["obj_ids"]
#             mask = b["triple_mask"]
#
#             kg_emb    = adapter.kg_encoder(subj, pred, obj, mask)
#             kg_tokens = adapter.kg_projector(kg_emb) * adapter.kg_scale
#             kg_flat   = kg_tokens.reshape(kg_tokens.shape[0], -1)
#             kg_bias   = adapter.kg_to_hidden(kg_flat)
#
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_kg_end = time.perf_counter()
#
#             # ── LLM generation time ───────────────────────────────────────────
#             adapter._kg_bias = kg_bias
#             adapter._has_kg  = b["has_kg"]
#
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_llm_start = time.perf_counter()
#
#             _ = adapter.llm.generate(
#                 input_ids      = b["input_ids"],
#                 attention_mask = b["attention_mask"],
#                 max_new_tokens = 64,
#                 do_sample      = False,
#             )
#
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             t_llm_end = time.perf_counter()
#
#             adapter._kg_bias = None
#             adapter._has_kg  = None
#
#             times_kg_enc.append((t_kg_end   - t_kg_start)  * 1000)
#             times_llm.append(   (t_llm_end  - t_llm_start) * 1000)
#             times_total.append( (t_llm_end  - t_kg_start)  * 1000)
#
#     return times_total, times_kg_enc, times_llm
#
#
# def print_results(results):
#     print("\n" + "="*65)
#     print("  LATENCY AND COMPUTATIONAL OVERHEAD ANALYSIS")
#     print("="*65)
#
#     base = results["base"]
#     adap = results["adapter"]
#
#     print(f"\n{'─'*65}")
#     print(f"  PER-QUERY LATENCY (ms)  [n={results['n_samples']} queries, "
#           f"batch_size=1]")
#     print(f"{'─'*65}")
#     print(f"  {'Metric':<35} {'Base LLM':>12} {'KG Adapter':>12} {'Overhead':>12}")
#     print(f"  {'-'*72}")
#
#     overhead_ms  = adap["total_mean"] - base["mean"]
#     overhead_pct = 100 * overhead_ms / base["mean"]
#     kg_enc_pct   = 100 * adap["kg_enc_mean"] / adap["total_mean"]
#
#     print(f"  {'Mean latency (ms)':<35} {base['mean']:>11.1f}  "
#           f"{adap['total_mean']:>11.1f}  {overhead_ms:>+10.1f}")
#     print(f"  {'Std  latency (ms)':<35} {base['std']:>11.1f}  "
#           f"{adap['total_std']:>11.1f}")
#     print(f"  {'P50  latency (ms)':<35} {base['p50']:>11.1f}  "
#           f"{adap['total_p50']:>11.1f}")
#     print(f"  {'P95  latency (ms)':<35} {base['p95']:>11.1f}  "
#           f"{adap['total_p95']:>11.1f}")
#     print(f"  {'P99  latency (ms)':<35} {base['p99']:>11.1f}  "
#           f"{adap['total_p99']:>11.1f}")
#
#     print(f"\n  Overhead: {overhead_ms:.1f} ms/query ({overhead_pct:.2f}%)")
#     print(f"  KG encoding: {adap['kg_enc_mean']:.1f} ms "
#           f"({kg_enc_pct:.1f}% of total adapter time)")
#     print(f"  LLM generation: {adap['llm_mean']:.1f} ms "
#           f"({100-kg_enc_pct:.1f}% of total adapter time)")
#
#     print(f"\n{'─'*65}")
#     print(f"  GPU MEMORY USAGE")
#     print(f"{'─'*65}")
#     print(f"  Base LLM   : {results['mem_base_mb']:.1f} MB")
#     print(f"  KG Adapter : {results['mem_adapter_mb']:.1f} MB")
#     print(f"  Overhead   : {results['mem_adapter_mb']-results['mem_base_mb']:.1f} MB "
#           f"(+{100*(results['mem_adapter_mb']-results['mem_base_mb'])/max(results['mem_base_mb'],1):.1f}%)")
#
#     print(f"\n{'─'*65}")
#     print(f"  GPU-HOURS PROJECTION (per 1M queries)")
#     print(f"{'─'*65}")
#
#     ms_per_query_base  = base["mean"]
#     ms_per_query_adap  = adap["total_mean"]
#     ms_per_query_overhead = overhead_ms
#
#     sec_1M_base     = ms_per_query_base     * 1_000_000 / 1000
#     sec_1M_adap     = ms_per_query_adap     * 1_000_000 / 1000
#     sec_1M_overhead = ms_per_query_overhead * 1_000_000 / 1000
#
#     hr_1M_base     = sec_1M_base     / 3600
#     hr_1M_adap     = sec_1M_adap     / 3600
#     hr_1M_overhead = sec_1M_overhead / 3600
#
#     # Cost estimates (AWS A100 ~$3.50/hr, A6000 ~$2.00/hr)
#     cost_a100_base    = hr_1M_base     * 3.50
#     cost_a100_adap    = hr_1M_adap     * 3.50
#     cost_a100_overhead= hr_1M_overhead * 3.50
#
#     print(f"  {'Scenario':<30} {'Base LLM':>12} {'KG Adapter':>12} {'Overhead':>12}")
#     print(f"  {'-'*68}")
#     print(f"  {'Total GPU-hours':<30} {hr_1M_base:>11.2f}h {hr_1M_adap:>11.2f}h "
#           f"{hr_1M_overhead:>+10.2f}h")
#     print(f"  {'Cost @ $3.50/hr (A100)':<30} ${cost_a100_base:>10.2f} "
#           f"${cost_a100_adap:>10.2f} ${cost_a100_overhead:>+9.2f}")
#     print(f"\n  KG encoding overhead alone: {hr_1M_overhead:.2f} GPU-hours "
#           f"per 1M queries")
#     print(f"  = {sec_1M_overhead/60:.1f} GPU-minutes per 1M queries")
#     print(f"  = {overhead_ms:.1f} ms per query ({overhead_pct:.2f}% overhead)")
#
#     print(f"\n{'─'*65}")
#     print(f"  TRAINABLE PARAMETER OVERHEAD")
#     print(f"{'─'*65}")
#     print(f"  Base LLM params    : {results['base_params']:,}")
#     print(f"  Adapter params     : {results['adapter_params']:,}")
#     print(f"  Overhead           : {results['adapter_params']:,} "
#           f"({100*results['adapter_params']/max(results['base_params'],1):.2f}% of LLM)")
#
#     print(f"\n{'─'*65}")
#     print(f"  INTERPRETATION")
#     print(f"{'─'*65}")
#     print(f"  KG encoding adds {overhead_ms:.1f} ms ({overhead_pct:.1f}%) per query.")
#     if overhead_pct < 5:
#         print(f"  This is negligible — the adapter overhead is dominated by LLM generation.")
#     elif overhead_pct < 15:
#         print(f"  This is modest — most time is still spent in LLM generation.")
#     else:
#         print(f"  This is significant — consider caching KG embeddings for repeated entities.")
#     print(f"  At 1M queries, overhead = {hr_1M_overhead:.2f} GPU-hours "
#           f"(${cost_a100_overhead:.2f} on A100).")
#     print(f"  The {results['adapter_params']:,} adapter parameters add only "
#           f"{(results['mem_adapter_mb']-results['mem_base_mb']):.0f} MB GPU memory.")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config",    required=True)
#     parser.add_argument("--n_samples", type=int, default=100)
#     parser.add_argument("--n_warmup",  type=int, default=10)
#     parser.add_argument("--n_runs",    type=int, default=3,
#                         help="Repeated runs per sample for stable timing")
#     parser.add_argument("--output",    default="logs/latency_analysis.json")
#     args = parser.parse_args()
#
#     cfg       = load_config(args.config)
#     device    = cfg["training"]["device"]
#     model_cfg = get_active_model_cfg(cfg)
#     active    = cfg["models"]["active"]
#     proc      = cfg["paths"]["processed_dir"]
#
#     print(f"[latency] Device: {device}, Model: {active}")
#
#     # ── Tokenizer ─────────────────────────────────────────────────────────────
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_cfg["local_path"], local_files_only=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     # ── Load test samples directly (same as evaluator) ────────────────────────
#     with open(str(Path(proc) / "test_with_triples.json")) as f:
#         samples = json.load(f)
#     with open(str(Path(proc) / "vocab.json")) as f:
#         vocab = json.load(f)
#     with open(str(Path(proc) / "labels.json")) as f:
#         labels = json.load(f)
#
#     samples = samples[:args.n_samples + args.n_warmup]
#     entity2id   = vocab["entity2id"]
#     relation2id = vocab["relation2id"]
#
#     def make_tensors(triples, dev):
#         if triples:
#             subj = torch.tensor([[entity2id.get(t[0], 0)   for t in triples]],
#                                  dtype=torch.long).to(dev)
#             pred = torch.tensor([[relation2id.get(t[1], 0) for t in triples]],
#                                  dtype=torch.long).to(dev)
#             obj  = torch.tensor([[entity2id.get(t[2], 0)   for t in triples]],
#                                  dtype=torch.long).to(dev)
#             mask = torch.ones(1, len(triples), dtype=torch.long).to(dev)
#             has_kg = torch.tensor([True], dtype=torch.bool).to(dev)
#         else:
#             subj = pred = obj = torch.zeros(1, 1, dtype=torch.long).to(dev)
#             mask   = torch.zeros(1, 1, dtype=torch.long).to(dev)
#             has_kg = torch.tensor([False], dtype=torch.bool).to(dev)
#         return subj, pred, obj, mask, has_kg
#
#     def make_inputs(question, triples):
#         context = ""
#         if triples:
#             parts = []
#             for s, p, o in triples:
#                 parts.append(f"{labels.get(s,s)} {labels.get(p,p)} {labels.get(o,o)}")
#             context = ". ".join(parts) + "."
#         if context:
#             prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
#         else:
#             prompt = f"Question: {question}\nAnswer:"
#         return tokenizer(prompt, return_tensors="pt",
#                          truncation=True, max_length=480)
#
#     # ── Load base LLM ─────────────────────────────────────────────────────────
#     print("[latency] Loading base LLM...")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_cfg["local_path"],
#         local_files_only=True,
#         dtype=torch.bfloat16,
#         device_map=device,
#     )
#     base_model.eval()
#     base_params = sum(p.numel() for p in base_model.parameters())
#
#     if torch.cuda.is_available(): torch.cuda.synchronize()
#     mem_base_mb = get_gpu_memory_mb()
#     print(f"[latency] Base LLM loaded. Memory: {mem_base_mb:.0f} MB")
#
#     # ── Load KG Adapter (exact same as evaluator) ─────────────────────────────
#     print("[latency] Loading KG Adapter...")
#     adapter_model = build_model(cfg, device)
#     ckpt_base = Path(cfg["paths"]["checkpoint_dir"])
#     if active == "mistral":
#         ckpt_path = ckpt_base / "mistral" / "best"
#     else:
#         ckpt_path = ckpt_base / "best"
#     print(f"[latency] Loading checkpoint from: {ckpt_path}")
#     adapter_model.load_adapter(str(ckpt_path), name=f"{active}_adapter")
#     adapter_model.eval()
#
#     adapter_params = sum(p.numel() for p in adapter_model.parameters()
#                          if p.requires_grad)
#
#     if torch.cuda.is_available(): torch.cuda.synchronize()
#     mem_adapter_mb = get_gpu_memory_mb()
#     print(f"[latency] Adapter loaded. Memory: {mem_adapter_mb:.0f} MB")
#
#     # ── Warmup ────────────────────────────────────────────────────────────────
#     print(f"[latency] Warming up ({args.n_warmup} queries)...")
#     for s in samples[:args.n_warmup]:
#         inp = make_inputs(s["question"], s["triples"])
#         with torch.no_grad():
#             base_model.generate(
#                 input_ids=inp["input_ids"].to(device),
#                 attention_mask=inp["attention_mask"].to(device),
#                 max_new_tokens=64, do_sample=False,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id)
#
#     # ── Measure ───────────────────────────────────────────────────────────────
#     measure_samples = samples[args.n_warmup: args.n_warmup + args.n_samples]
#     print(f"[latency] Measuring {len(measure_samples)} queries...")
#
#     base_times   = []
#     total_times  = []
#     kg_enc_times = []
#     llm_times    = []
#
#     for s in tqdm(measure_samples):
#         inp    = make_inputs(s["question"], s["triples"])
#         input_ids   = inp["input_ids"].to(device)
#         attn_mask   = inp["attention_mask"].to(device)
#         subj, pred, obj, mask, has_kg = make_tensors(s["triples"], device)
#
#         # Base latency
#         bt = measure_base_latency(
#             base_model, tokenizer,
#             {"input_ids": input_ids, "attention_mask": attn_mask},
#             device, args.n_runs)
#         base_times.append(float(np.median(bt)))
#
#         # Adapter latency — replicate generate_answer_with_adapter logic
#         at_list, kt_list, lt_list = [], [], []
#         for _ in range(args.n_runs):
#             if torch.cuda.is_available(): torch.cuda.synchronize()
#             t_kg0 = time.perf_counter()
#
#             with torch.no_grad():
#                 B = input_ids.size(0)
#                 kg_emb    = adapter_model.kg_encoder(subj, pred, obj, mask)
#                 kg_tokens = adapter_model.kg_projector(kg_emb) * adapter_model.kg_scale
#                 kg_tokens = kg_tokens * has_kg.float().view(B, 1, 1)
#                 kg_flat   = kg_tokens.reshape(B, -1)
#                 kg_bias   = adapter_model.kg_to_hidden(kg_flat).unsqueeze(1)
#                 adapter_model._kg_bias = kg_bias
#
#             if torch.cuda.is_available(): torch.cuda.synchronize()
#             t_kg1 = time.perf_counter()
#
#             with torch.no_grad():
#                 adapter_model.llm.generate(
#                     input_ids=input_ids,
#                     attention_mask=attn_mask,
#                     max_new_tokens=64,
#                     do_sample=False,
#                     pad_token_id=tokenizer.pad_token_id,
#                     eos_token_id=tokenizer.eos_token_id)
#                 adapter_model._kg_bias = None
#
#             if torch.cuda.is_available(): torch.cuda.synchronize()
#             t_llm1 = time.perf_counter()
#
#             kt_list.append((t_kg1  - t_kg0)  * 1000)
#             lt_list.append((t_llm1 - t_kg1)  * 1000)
#             at_list.append((t_llm1 - t_kg0)  * 1000)
#
#         total_times.append(float(np.median(at_list)))
#         kg_enc_times.append(float(np.median(kt_list)))
#         llm_times.append(float(np.median(lt_list)))
#
#     # ── Aggregate and report ──────────────────────────────────────────────────
#     results = {
#         "n_samples":      len(measure_samples),
#         "model":          active,
#         "base_params":    base_params,
#         "adapter_params": adapter_params,
#         "mem_base_mb":    mem_base_mb,
#         "mem_adapter_mb": mem_adapter_mb,
#         "base": {
#             "mean": float(np.mean(base_times)),
#             "std":  float(np.std(base_times)),
#             "p50":  float(np.percentile(base_times, 50)),
#             "p95":  float(np.percentile(base_times, 95)),
#             "p99":  float(np.percentile(base_times, 99)),
#         },
#         "adapter": {
#             "total_mean":  float(np.mean(total_times)),
#             "total_std":   float(np.std(total_times)),
#             "total_p50":   float(np.percentile(total_times, 50)),
#             "total_p95":   float(np.percentile(total_times, 95)),
#             "total_p99":   float(np.percentile(total_times, 99)),
#             "kg_enc_mean": float(np.mean(kg_enc_times)),
#             "kg_enc_std":  float(np.std(kg_enc_times)),
#             "llm_mean":    float(np.mean(llm_times)),
#             "llm_std":     float(np.std(llm_times)),
#         },
#     }
#
#     print_results(results)
#
#     Path(args.output).parent.mkdir(parents=True, exist_ok=True)
#     with open(args.output, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n[latency] Results saved to {args.output}")
#
#
# if __name__ == "__main__":
#     main()