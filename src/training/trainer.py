
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config, get_active_model_cfg
from src.adapter.kg_encoder import KGEncoder
from src.adapter.kg_projector import KGProjector
from src.adapter.kg_adapter import KGAdapter
from src.data.collator import WWQTripleDataset, collate_fn


def build_model(cfg: dict, device: str) -> KGAdapter:
    model_cfg = get_active_model_cfg(cfg)
    enc_cfg   = cfg["kg_encoder"]
    proj_cfg  = cfg["kg_projector"]
    ada_cfg   = cfg["adapter"]

    kg_encoder = KGEncoder(
        num_entities  = 3408,
        num_relations = 114,
        emb_dim       = enc_cfg["entity_emb_dim"],
        hidden_dim    = enc_cfg["hidden_dim"],
        output_dim    = enc_cfg["output_dim"],
        dropout       = enc_cfg["dropout"],
        aggregation   = enc_cfg["aggregation"],
    ).to(device)

    kg_projector = KGProjector(
        input_dim     = enc_cfg["output_dim"],
        d_model       = model_cfg["hidden_size"],
        num_kg_tokens = ada_cfg["num_kg_tokens"],
        hidden_dim    = proj_cfg["hidden_dim"],
        dropout       = proj_cfg["dropout"],
    ).to(device)

    model = KGAdapter(
        local_model_path=model_cfg["local_path"],
        kg_encoder=kg_encoder,
        kg_projector=kg_projector,
        kg_scale_factor=ada_cfg["kg_scale_factor"],
        max_length=cfg["training"]["max_length"],
        device=device,
    )
    model.kg_to_hidden.to(device)
    return model


def build_loaders(cfg: dict, tokenizer, train_cfg: dict):
    proc  = cfg["paths"]["processed_dir"]
    vocab = str(Path(proc) / "vocab.json")

    def make_loader(split, shuffle):
        ds = WWQTripleDataset(
            path=str(Path(proc) / f"{split}_with_triples.json"),
            vocab_path=vocab,
            tokenizer=tokenizer,
            max_length=train_cfg["max_length"],
            labels_path=str(Path(proc) / "labels.json"),
        )
        return DataLoader(
            ds,
            batch_size  = train_cfg["batch_size"],
            shuffle     = shuffle,
            collate_fn  = partial(collate_fn,
                                  pad_token_id=tokenizer.pad_token_id or 0),
            num_workers = 4,
            pin_memory  = True,
        )

    return make_loader("train", shuffle=True), \
           make_loader("valid", shuffle=False)


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                subj_ids       = batch["subj_ids"],
                pred_ids       = batch["pred_ids"],
                obj_ids        = batch["obj_ids"],
                triple_mask    = batch["triple_mask"],
                has_kg         = batch["has_kg"],
                labels         = batch["labels"],
            )
            if torch.isfinite(outputs.loss):
                total_loss += outputs.loss.item()
                steps += 1
    return total_loss / max(steps, 1)


def train(cfg: dict):
    train_cfg = cfg["training"]
    device    = train_cfg["device"]
    ckpt_dir  = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, device)

    model_cfg = get_active_model_cfg(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["local_path"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, valid_loader = build_loaders(cfg, tokenizer, train_cfg)

    # Debug first batch
    first_batch = next(iter(train_loader))
    labels      = first_batch["labels"]
    valid_count = (labels != -100).sum().item()
    print(f"[debug] labels shape={labels.shape}  valid_tokens={valid_count}/{labels.numel()}")
    del first_batch

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr           = train_cfg["learning_rate"],
        weight_decay = train_cfg["weight_decay"],
        eps          = 1e-8,
    )
    total_steps = (len(train_loader) * train_cfg["num_epochs"]
                   // train_cfg["gradient_accumulation_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1))

    best_val_loss = float("inf")
    global_step   = 0
    accum_steps   = train_cfg["gradient_accumulation_steps"]

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        model.kg_encoder.train()
        model.kg_projector.train()

        epoch_loss  = 0.0
        valid_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                subj_ids       = batch["subj_ids"],
                pred_ids       = batch["pred_ids"],
                obj_ids        = batch["obj_ids"],
                triple_mask    = batch["triple_mask"],
                has_kg         = batch["has_kg"],
                labels         = batch["labels"],
            )
            loss = outputs.loss / accum_steps

            if not torch.isfinite(loss):
                print(f"\n[trainer] WARNING: non-finite loss at step {step}, skipping")
                optimizer.zero_grad()
                continue

            loss.backward()

            # Clamp raw gradients before norm-based clipping
            for p in model.trainable_parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-10.0, 10.0)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), train_cfg["max_grad_norm"])

            if not torch.isfinite(grad_norm):
                print(f"\n[trainer] WARNING: nan grad_norm at step {step}, skipping")
                optimizer.zero_grad()
                continue

            epoch_loss  += loss.item() * accum_steps
            valid_steps += 1

            # Log every step for epoch 1, else every 50
            if epoch == 0 or global_step % 50 == 0:
                print(f"\n[trainer] step={global_step}  "
                      f"loss={loss.item()*accum_steps:.4f}  "
                      f"grad_norm={grad_norm:.4f}")

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % train_cfg["save_every_n_steps"] == 0:
                    model.save_adapter(str(ckpt_dir / f"step_{global_step}"))

                if global_step % train_cfg["eval_every_n_steps"] == 0:
                    val_loss = evaluate(model, valid_loader, device)
                    print(f"\n[trainer] step={global_step}  val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_adapter(str(ckpt_dir / "best"))
                    model.train()

        avg_loss = epoch_loss / max(valid_steps, 1)
        print(f"[trainer] Epoch {epoch+1}  train_loss={avg_loss:.4f}")

    model.save_adapter(cfg["paths"]["adapter_dir"])
    print(f"[trainer] Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))