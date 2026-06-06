"""
Fetch missing triples from Wikidata API for zero-triple samples.
Targets only samples where triples == [].
"""
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


def fetch_entity_claims(entity_id: str, properties: list, retries: int = 3) -> list:

    url = (f"{WIKIDATA_API}?action=wbgetentities&ids={entity_id}"
           f"&props=claims&format=json")
    req = Request(url, headers={"User-Agent": "kg-adapter-research/1.0"})

    for attempt in range(retries):
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            break
        except (URLError, Exception) as e:
            if attempt == retries - 1:
                return []
            time.sleep(2 ** attempt)

    entity_data = data.get("entities", {}).get(entity_id, {})
    claims      = entity_data.get("claims", {})
    triples     = []

    for prop, claim_list in claims.items():
        # Filter to only WWQ properties if provided
        if properties and prop not in properties:
            continue
        for claim in claim_list:
            try:
                snak  = claim["mainsnak"]
                if snak["snaktype"] != "value":
                    continue
                value = snak["datavalue"]["value"]

                if isinstance(value, dict) and "id" in value:
                    obj = value["id"]
                    if obj.startswith("Q"):
                        triples.append([entity_id, prop, obj])
            except (KeyError, TypeError):
                continue

    return triples


def fill_missing(split: str, cfg: dict) -> None:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    path          = processed_dir / f"{split}_with_triples.json"

    with open(path) as f:
        samples = json.load(f)

    missing = [s for s in samples if len(s["triples"]) == 0]
    print(f"[fetch_missing] {split}: {len(missing)} zero-triple samples")

    if not missing:
        print(f"[fetch_missing] Nothing to fetch for {split}.")
        return

    filled = 0
    for sample in tqdm(missing, desc=f"Fetching [{split}]"):
        fetched = []
        for eid in sample["entities"]:
            triples = fetch_entity_claims(eid, sample["properties"])
            fetched.extend(triples)
            time.sleep(0.1)

        if fetched:
            sample["triples"] = fetched
            filled += 1

        time.sleep(0.05)

    # Save back
    with open(path, "w") as f:
        json.dump(samples, f)

    still_missing = sum(1 for s in samples if len(s["triples"]) == 0)
    print(f"[fetch_missing] Filled   : {filled}/{len(missing)}")
    print(f"[fetch_missing] Still 0  : {still_missing}")
    print(f"[fetch_missing] Saved to : {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split",  default="all",
                        choices=["train", "valid", "test", "all"])
    args = parser.parse_args()

    cfg    = load_config(args.config)
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    for split in splits:
        fill_missing(split, cfg)