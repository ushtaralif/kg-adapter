import torch
import json
from pathlib import Path
from torch.utils.data import Dataset


class WWQTripleDataset(Dataset):

    def __init__(self, path: str, vocab_path: str, tokenizer,
                 max_length: int = 512, labels_path: str = None):
        with open(path) as f:
            self.samples = json.load(f)
        with open(vocab_path) as f:
            vocab = json.load(f)

        self.entity2id   = vocab["entity2id"]
        self.relation2id = vocab["relation2id"]
        self.tokenizer   = tokenizer
        self.max_length  = max_length

        # Load labels
        if labels_path and Path(labels_path).exists():
            with open(labels_path) as f:
                self.labels = json.load(f)
        else:
            self.labels = {}

        self.prompt_template = "Context: {context}\nQuestion: {question}\nAnswer:"
        self.prompt_no_ctx   = "Question: {question}\nAnswer:"

    def __len__(self):
        return len(self.samples)

    def _build_context(self, triples: list) -> str:
        """Convert triples to natural language context using labels."""
        if not triples:
            return ""
        seen  = set()
        parts = []
        for s, p, o in triples:
            s_label = self.labels.get(s, s)
            p_label = self.labels.get(p, p)
            o_label = self.labels.get(o, o)
            stmt    = f"{s_label} {p_label} {o_label}"
            if stmt not in seen:
                seen.add(stmt)
                parts.append(stmt)
        return ". ".join(parts) + "."

    def __getitem__(self, idx):
        s = self.samples[idx]

        answer  = s["answers"][0] if s["answers"] else ""
        triples = s["triples"]
        context = self._build_context(triples)

        if context:
            prompt = self.prompt_template.format(
                context=context, question=s["question"])
        else:
            prompt = self.prompt_no_ctx.format(question=s["question"])

        full = prompt + " " + answer

        prompt_enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length,
            return_tensors="pt", add_special_tokens=True)

        full_enc = self.tokenizer(
            full, truncation=True, max_length=self.max_length,
            return_tensors="pt", add_special_tokens=True)

        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        prompt_len = prompt_enc["input_ids"].shape[1]
        labels     = input_ids.clone()
        labels[:prompt_len] = -100

        subj_ids = [self.entity2id.get(t[0], 0) for t in triples]
        pred_ids = [self.relation2id.get(t[1], 0) for t in triples]
        obj_ids  = [self.entity2id.get(t[2], 0)  for t in triples]

        if not subj_ids:
            subj_ids = [0]; pred_ids = [0]; obj_ids = [0]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "subj_ids":       torch.tensor(subj_ids, dtype=torch.long),
            "pred_ids":       torch.tensor(pred_ids, dtype=torch.long),
            "obj_ids":        torch.tensor(obj_ids,  dtype=torch.long),
            "has_kg":         torch.tensor([len(triples) > 0], dtype=torch.bool),
        }


def collate_fn(batch: list, pad_token_id: int) -> dict:

    def pad_seq(seqs, pad_val=0):
        max_len = max(s.size(0) for s in seqs)
        return torch.stack([
            torch.cat([s, torch.full((max_len - s.size(0),), pad_val, dtype=s.dtype)])
            for s in seqs
        ])

    input_ids      = pad_seq([b["input_ids"]      for b in batch], pad_token_id)
    attention_mask = pad_seq([b["attention_mask"]  for b in batch], 0)
    labels         = pad_seq([b["labels"]          for b in batch], -100)
    subj_ids       = pad_seq([b["subj_ids"]        for b in batch], 0)
    pred_ids       = pad_seq([b["pred_ids"]        for b in batch], 0)
    obj_ids        = pad_seq([b["obj_ids"]         for b in batch], 0)

    triple_lens = [b["subj_ids"].size(0) for b in batch]
    max_triples = max(triple_lens)
    triple_mask = torch.zeros(len(batch), max_triples, dtype=torch.long)
    for i, l in enumerate(triple_lens):
        triple_mask[i, :l] = 1

    has_kg = torch.cat([b["has_kg"] for b in batch])

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "subj_ids":       subj_ids,
        "pred_ids":       pred_ids,
        "obj_ids":        obj_ids,
        "triple_mask":    triple_mask,
        "has_kg":         has_kg,
    }