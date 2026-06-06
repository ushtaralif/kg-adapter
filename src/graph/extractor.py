
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config
from src.data.dataset import WWQDataset


def collect_wwq_entities_and_properties(cfg: dict):
    """Collect all unique entities and properties across all splits."""
    entities, properties = set(), set()
    for split in ["train", "valid", "test"]:
        path = cfg["paths"][f"wwq_{split}"]
        if not Path(path).exists():
            continue
        ds = WWQDataset(path)
        for s in ds:
            entities.update(s.entities)
            properties.update(s.properties)
    print(f"[extractor] Unique entities   : {len(entities)}")
    print(f"[extractor] Unique properties : {len(properties)}")
    return entities, properties


def extract_triples(cfg: dict) -> dict:
    """
    Single pass through dump.
    Returns dict: entity_id -> list of [S, P, O] triples where entity appears.
    """
    dump_path  = cfg["paths"]["wikidata_dump"]
    out_path   = Path(cfg["paths"]["processed_dir"]) / "wwq_triples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sg_cfg     = cfg["subgraph"]

    entities, properties = collect_wwq_entities_and_properties(cfg)

    print(f"[extractor] Scanning dump: {dump_path}")

    # entity_id → list of (S, P, O)
    index = defaultdict(list)
    total, kept = 0, 0

    open_fn = gzip.open if str(dump_path).endswith(".gz") else open

    with open_fn(dump_path, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Scanning dump", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            s, p, o = parts
            if not (s.startswith("Q") and p.startswith("P") and o.startswith("Q")):
                continue

            total += 1
            if s in entities:
                index[s].append([s, p, o])
                kept += 1

    print(f"[extractor] Scanned  : {total:,} triples")
    print(f"[extractor] Kept     : {kept:,} triples")
    print(f"[extractor] Entities with triples: {len(index)}/{len(entities)}")

    # Save
    serializable = {k: v for k, v in index.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f)
    print(f"[extractor] Saved to : {out_path}")
    return index


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    index = extract_triples(cfg)

    for eid, triples in list(index.items())[:3]:
        print(f"\n  Entity {eid}: {len(triples)} triples")
        for t in triples[:3]:
            print(f"    {t}")