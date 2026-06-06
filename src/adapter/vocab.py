
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config


class Vocabulary:
    PAD = 0  # padding index

    def __init__(self):
        self.entity2id:   dict = {"<PAD>": 0}
        self.relation2id: dict = {"<PAD>": 0}

    def build(self, processed_dir: str) -> None:
        processed_dir = Path(processed_dir)
        for split in ["train_with_triples", "valid_with_triples", "test_with_triples"]:
            path = processed_dir / f"{split}.json"
            if not path.exists():
                continue
            with open(path) as f:
                samples = json.load(f)
            for s in samples:
                for triple in s["triples"]:
                    subj, pred, obj = triple
                    if subj not in self.entity2id:
                        self.entity2id[subj]   = len(self.entity2id)
                    if obj not in self.entity2id:
                        self.entity2id[obj]    = len(self.entity2id)
                    if pred not in self.relation2id:
                        self.relation2id[pred] = len(self.relation2id)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"entity2id": self.entity2id,
                       "relation2id": self.relation2id}, f)
        print(f"[vocab] Saved: {len(self.entity2id)} entities, "
              f"{len(self.relation2id)} relations → {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        v = cls()
        v.entity2id   = data["entity2id"]
        v.relation2id = data["relation2id"]
        return v

    @property
    def num_entities(self):  return len(self.entity2id)

    @property
    def num_relations(self): return len(self.relation2id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    vocab = Vocabulary()
    vocab.build(cfg["paths"]["processed_dir"])
    vocab.save(Path(cfg["paths"]["processed_dir"]) / "vocab.json")
    print(f"[vocab] num_entities  : {vocab.num_entities}")
    print(f"[vocab] num_relations : {vocab.num_relations}")