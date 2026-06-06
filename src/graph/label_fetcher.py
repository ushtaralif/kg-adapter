
import json
import time
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def fetch_labels_batch(ids: list, retries: int = 3) -> dict:

    ids_str = "|".join(ids)
    url     = (f"{WIKIDATA_API}?action=wbgetentities&ids={ids_str}"
               f"&props=labels&languages=en&format=json")
    req = Request(url, headers={"User-Agent": "kg-adapter-research/1.0"})

    for attempt in range(retries):
        try:
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            break
        except (URLError, Exception) as e:
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)

    labels = {}
    for eid, edata in data.get("entities", {}).items():
        label = edata.get("labels", {}).get("en", {}).get("value", None)
        if label:
            labels[eid] = label
    return labels


def fetch_all_labels(cfg: dict) -> dict:
    proc       = Path(cfg["paths"]["processed_dir"])
    vocab_path = proc / "vocab.json"
    out_path   = proc / "labels.json"

    if out_path.exists():
        print(f"[labels] Already exists at {out_path}. Delete to re-fetch.")
        with open(out_path) as f:
            return json.load(f)

    with open(vocab_path) as f:
        vocab = json.load(f)

    all_ids = list(vocab["entity2id"].keys()) + list(vocab["relation2id"].keys())
    all_ids = [i for i in all_ids if i != "<PAD>"]
    print(f"[labels] Fetching labels for {len(all_ids)} entities/properties...")

    labels     = {}
    batch_size = 50

    for i in tqdm(range(0, len(all_ids), batch_size), desc="Fetching labels"):
        batch   = all_ids[i:i + batch_size]
        result  = fetch_labels_batch(batch)
        labels.update(result)
        time.sleep(0.1)

    for eid in all_ids:
        if eid not in labels:
            labels[eid] = eid

    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2)

    found = sum(1 for k, v in labels.items() if v != k)
    print(f"[labels] Done. Found {found}/{len(all_ids)} labels. Saved to {out_path}")
    return labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg    = load_config(args.config)
    labels = fetch_all_labels(cfg)

    # Preview
    items = list(labels.items())
    print("\nSample labels:")
    for k, v in items[:10]:
        print(f"  {k}: {v}")