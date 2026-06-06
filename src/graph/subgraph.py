
import json
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config
from src.data.dataset import WWQDataset, WWQSample


class TripleIndex:

    def __init__(self, triples_path: str):
        with open(triples_path) as f:
            raw = json.load(f)
        self.by_subject  = {k: [tuple(t) for t in v] for k, v in raw.items()}
        self.by_property = defaultdict(list)
        for triples in self.by_subject.values():
            for t in triples:
                self.by_property[t[1]].append(t)
        print(f"[TripleIndex] Loaded {sum(len(v) for v in self.by_subject.values()):,} triples")

    def get_subgraph(self, entities: list, properties: list,
                     max_per_entity: int = 50, max_total: int = 200) -> list:
        prop_set = set(properties)
        seen, triples = set(), []

        # Pass 1: entity-centric
        for ent in entities:
            count = 0
            for t in self.by_subject.get(ent, []):
                if count >= max_per_entity: break
                if prop_set and t[1] not in prop_set: continue
                if t not in seen:
                    seen.add(t); triples.append(t); count += 1
                if len(triples) >= max_total: return triples

        # Pass 2: property-centric
        ent_set = set(entities)
        for prop in properties:
            for t in self.by_property.get(prop, []):
                if t[0] in ent_set or t[2] in ent_set:
                    if t not in seen:
                        seen.add(t); triples.append(t)
                if len(triples) >= max_total: return triples

        return triples


def process_split(split: str, cfg: dict) -> list:
    path         = cfg["paths"][f"wwq_{split}"]
    triples_path = Path(cfg["paths"]["processed_dir"]) / "wwq_triples.json"
    sg_cfg       = cfg["subgraph"]

    print(f"[subgraph] Loading {split} from {path}")
    ds    = WWQDataset(path)
    index = TripleIndex(str(triples_path))

    results, no_triples = [], 0

    for sample in tqdm(ds, desc=f"Building subgraphs [{split}]"):
        sample.triples = index.get_subgraph(
            sample.entities, sample.properties,
            max_per_entity=sg_cfg["max_triples_per_entity"],
            max_total=sg_cfg["max_total_triples"],
        )
        if len(sample.triples) == 0:
            no_triples += 1

        results.append({
            "id":         sample.id,
            "question":   sample.question,
            "sparql":     sample.sparql,
            "answers":    sample.answers,
            "entities":   sample.entities,
            "properties": sample.properties,
            "triples":    [list(t) for t in sample.triples],
        })

    out_dir  = Path(cfg["paths"]["processed_dir"])
    out_path = out_dir / f"{split}_with_triples.json"
    with open(out_path, "w") as f:
        json.dump(results, f)

    total = len(results)
    avg_t = sum(len(r["triples"]) for r in results) / max(total, 1)
    print(f"\n[subgraph] ✓ {split} done")
    print(f"  Samples              : {total}")
    print(f"  Avg triples/sample   : {avg_t:.1f}")
    print(f"  Samples w/ 0 triples : {no_triples} ({100*no_triples/max(total,1):.1f}%)")
    print(f"  Saved to             : {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split",  default="train", choices=["train","valid","test"])
    parser.add_argument("--show",   type=int, default=2)
    args = parser.parse_args()

    cfg     = load_config(args.config)
    results = process_split(args.split, cfg)

    for r in results[:args.show]:
        print(f"\n  ID       : {r['id']}")
        print(f"  Question : {r['question']}")
        print(f"  Triples  : {len(r['triples'])}  sample={r['triples'][:3]}")