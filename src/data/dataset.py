
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config

ENTITY_RE   = re.compile(r'(?:wd:|wdt:|p:|ps:|pq:|psv:|pqv:)?(\bQ\d+\b)')
PROPERTY_RE = re.compile(r'(?:wdt:|p:|ps:|pq:|psv:|pqv:|wdp:)?(\bP\d+\b)')


@dataclass
class WWQSample:
    id:         str
    question:   str
    sparql:     str
    answers:    list
    entities:   list = field(default_factory=list)
    properties: list = field(default_factory=list)
    triples:    list = field(default_factory=list)  # filled by subgraph builder


def extract_entities_and_properties(sparql: str):
    """Extract deduplicated Q-IDs and P-IDs from a SPARQL string."""
    entities   = list(dict.fromkeys(ENTITY_RE.findall(sparql)))
    properties = list(dict.fromkeys(PROPERTY_RE.findall(sparql)))
    return entities, properties


class WWQDataset:

    def __init__(self, path: str):
        self.path    = Path(path)
        self.samples = self._load()

    def _load(self) -> list:
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]

        samples = []
        for item in raw:
            sparql = item.get("output_sparql", item.get("query", ""))
            ents, props = extract_entities_and_properties(sparql)

            answers = item.get("Answers", item.get("Answers", []))
            if isinstance(answers, str):
                answers = [answers]

            samples.append(WWQSample(
                id         = str(item.get("id", len(samples))),
                question   = item.get("Question", "").strip(),
                sparql     = sparql,
                answers    = answers,
                entities   = ents,
                properties = props,
            ))
        return samples

    def __len__(self)                  -> int:            return len(self.samples)
    def __getitem__(self, idx)         -> WWQSample:      return self.samples[idx]
    def __iter__(self) -> Iterator[WWQSample]:            return iter(self.samples)

    def stats(self) -> dict:
        n_ents  = [len(s.entities)   for s in self.samples]
        n_props = [len(s.properties) for s in self.samples]
        n_ans   = [len(s.answers)    for s in self.samples]
        return {
            "total_samples":          len(self.samples),
            "avg_entities_per_query": round(sum(n_ents)  / max(len(n_ents),  1), 2),
            "avg_props_per_query":    round(sum(n_props) / max(len(n_props), 1), 2),
            "avg_answers":            round(sum(n_ans)   / max(len(n_ans),   1), 2),
            "samples_with_no_entity": sum(1 for x in n_ents  if x == 0),
            "samples_with_no_prop":   sum(1 for x in n_props if x == 0),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split",  default="train", choices=["train","dev","test"])
    parser.add_argument("--show",   type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds  = WWQDataset(cfg["paths"][f"wwq_{args.split}"])
    print(f"\nStats: {ds.stats()}")
    for s in ds.samples[:args.show]:
        print(f"\n  ID       : {s.id}")
        print(f"  Question : {s.question}")
        print(f"  Entities : {s.entities}")
        print(f"  Props    : {s.properties}")
        print(f"  Answers  : {s.answers}")
