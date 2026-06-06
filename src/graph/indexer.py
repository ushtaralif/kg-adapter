

import argparse
import gzip
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import lmdb
import msgpack

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config

LMDB_MAP_SIZE = 200 * 1024 ** 3


# ── Serialization ─────────────────────────────────────────────────────────────

def _pack(data: list) -> bytes:
    return msgpack.packb(data, use_bin_type=True)

def _unpack(raw: bytes) -> list:
    return msgpack.unpackb(raw, raw=False)


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(dump_path: str, index_path: str,
                flush_every: int = 500_000,
                log_every: int = 1_000_000) -> None:
    dump_path  = Path(dump_path)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        print(f"[indexer] Index already exists at {index_path}. "
              "Delete it manually to rebuild.")
        return

    print(f"[indexer] Source : {dump_path}")
    print(f"[indexer] Index  : {index_path}")
    print(f"[indexer] Flush every {flush_every:,} triples")

    env = lmdb.open(str(index_path), map_size=LMDB_MAP_SIZE,
                    max_dbs=0, writemap=True, map_async=True)

    subj_buf: dict = defaultdict(list)
    prop_buf: dict = defaultdict(list)
    total     = 0
    malformed = 0
    t0        = time.time()

    def flush(sbuf, pbuf):
        with env.begin(write=True) as txn:
            for k, new_pairs in sbuf.items():
                existing = txn.get(k)
                pairs = _unpack(existing) if existing else []
                pairs.extend(new_pairs)
                txn.put(k, _pack(pairs))
            for k, new_pairs in pbuf.items():
                existing = txn.get(k)
                pairs = _unpack(existing) if existing else []
                pairs.extend(new_pairs)
                txn.put(k, _pack(pairs))
        sbuf.clear()
        pbuf.clear()

    open_fn = gzip.open if str(dump_path).endswith(".gz") else open

    with open_fn(dump_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                malformed += 1
                continue
            s, p, o = parts
            if not (s.startswith("Q") and p.startswith("P") and o.startswith("Q")):
                malformed += 1
                continue

            subj_buf[f"s:{s}".encode()].append([p, o])
            prop_buf[f"p:{p}".encode()].append([s, o])
            total += 1

            if total % flush_every == 0:
                flush(subj_buf, prop_buf)
                elapsed = time.time() - t0
                print(f"[indexer] {total:>12,} triples | "
                      f"{elapsed:6.0f}s | {total/elapsed:,.0f} t/s | "
                      f"malformed={malformed:,}")

            if total % log_every == 0:
                stat = env.stat()
                db_kb = stat["psize"] * (stat["leaf_pages"] + stat["branch_pages"]) // 1024
                print(f"[indexer] LMDB ~{db_kb/1024:.1f} MB")

    if subj_buf or prop_buf:
        flush(subj_buf, prop_buf)

    with env.begin(write=True) as txn:
        meta = {
            "total_triples":   total,
            "malformed_lines": malformed,
            "dump_path":       str(dump_path),
            "built_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        txn.put(b"__meta__", _pack(meta))

    env.close()
    elapsed = time.time() - t0
    print(f"\n[indexer] ✓ Done in {elapsed:.1f}s")
    print(f"[indexer]   Total triples   : {total:,}")
    print(f"[indexer]   Malformed lines : {malformed:,}")
    print(f"[indexer]   Index path      : {index_path}")


# ── Read-only query wrapper ───────────────────────────────────────────────────

class WikidataIndex:
    """
    Read-only wrapper for the LMDB index.

    idx = WikidataIndex("data/wikidata/index.lmdb")
    idx.get_triples_for_entity("Q31")   → [("P38", "Q4916"), ...]
    idx.get_triples_for_property("P38") → [("Q31", "Q4916"), ...]
    idx.get_subgraph(["Q31"], ["P38"])   → [("Q31","P38","Q4916"), ...]
    """

    def __init__(self, index_path: str):
        self.env = lmdb.open(str(index_path), readonly=True,
                             lock=False, max_readers=128)

    def get_triples_for_entity(self, entity_id: str) -> list:
        """All (predicate, object) where entity_id is subject."""
        return self._get(f"s:{entity_id}".encode())

    def get_triples_for_property(self, prop_id: str) -> list:
        """All (subject, object) pairs that use this property."""
        return self._get(f"p:{prop_id}".encode())

    def get_subgraph(self,
                     entities: list,
                     properties: list,
                     max_per_entity: int = 50,
                     max_total: int = 200) -> list:

        prop_set = set(properties)
        ent_set  = set(entities)
        seen     = set()
        triples  = []

        # Pass 1: entity-centric
        for ent in entities:
            count = 0
            for p, o in self.get_triples_for_entity(ent):
                if count >= max_per_entity:
                    break
                if prop_set and p not in prop_set:
                    continue
                key = (ent, p, o)
                if key not in seen:
                    seen.add(key)
                    triples.append(key)
                    count += 1
                if len(triples) >= max_total:
                    return triples

        # Pass 2: property-centric (fills gaps)
        for prop in properties:
            for s, o in self.get_triples_for_property(prop):
                if s in ent_set or o in ent_set:
                    key = (s, prop, o)
                    if key not in seen:
                        seen.add(key)
                        triples.append(key)
                if len(triples) >= max_total:
                    return triples

        return triples

    def meta(self) -> dict:
        with self.env.begin() as txn:
            raw = txn.get(b"__meta__")
        return _unpack(raw) if raw else {}

    def close(self):
        self.env.close()

    def _get(self, key: bytes) -> list:
        with self.env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            return []
        return [tuple(p) for p in _unpack(raw)]

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


# ── Verification ──────────────────────────────────────────────────────────────

def verify_index(index_path: str):
    print("\n[indexer] Running verification …")
    with WikidataIndex(index_path) as idx:
        print(f"  Metadata         : {idx.meta()}")
        t = idx.get_triples_for_entity("Q31")
        print(f"  Q31 triples      : {len(t)}  sample={t[:3]}")
        p = idx.get_triples_for_property("P31")
        print(f"  P31 triples      : {len(p)}  sample={p[:3]}")
        sg = idx.get_subgraph(["Q31"], ["P38", "P1151"])
        print(f"  Subgraph(Q31,[P38,P1151]) : {len(sg)} triples")
        for t_ in sg[:5]:
            print(f"    {t_}")
    print("[indexer] Verification OK ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Wikidata LMDB index")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--dump",        default=None, help="Override dump path")
    parser.add_argument("--index",       default=None, help="Override index path")
    parser.add_argument("--flush-every", type=int, default=500_000)
    parser.add_argument("--verify",      action="store_true")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    dump_path  = args.dump  or cfg["paths"]["wikidata_dump"]
    index_path = args.index or cfg["paths"]["wikidata_index"]

    build_index(dump_path, index_path, flush_every=args.flush_every)

    if args.verify:
        verify_index(index_path)