# Gated KG Adapter for Frozen LLMs


> **When to Inject: Selective Knowledge Graph Augmentation for Frozen Large Language Models**

---

## Overview

We propose a lightweight **KG Adapter** that injects Wikidata knowledge into frozen LLMs at inference time. 

We further introduce a **structural gating mechanism** that uses Wikidata entity features to decide *when* to inject, reducing KG computation by up to **94.9%** while improving accuracy.

### Run Locally

```bash
# Clone repository
git clone https://github.com/ushtaralif/kg-adapter.git
cd kg-adapter

# Install dependencies
pip install -r requirements.txt

# Update paths in config
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your local paths

# Build Wikidata index 
python -m src.graph.indexer --config configs/config.yaml

# Extract triples and build vocab
python -m src.graph.extractor --config configs/config.yaml
python -m src.adapter.vocab   --config configs/config.yaml

# Build subgraphs
python -m src.graph.subgraph --split train --config configs/config.yaml
python -m src.graph.subgraph --split valid --config configs/config.yaml
python -m src.graph.subgraph --split test  --config configs/config.yaml

# Train adapter
CUDA_VISIBLE_DEVICES=0 python -m src.training.trainer --config configs/config.yaml

# Evaluate
python -m src.evaluation.evaluator --config configs/config.yaml
```

---

## Model Weights

Pre-trained adapter weights are available on HuggingFace Hub:

| Model | HuggingFace Repo | Size |
|-------|-----------------|------|
| LLaMA-3 8B Adapter | [anonymous9800694/kg-adapter-llama3-8b](https://huggingface.co/anonymous9800694/kg-adapter-llama3-8b) | 279MB |
| Mistral-7B Adapter | [anonymous9800694/kg-adapter-mistral-7b](https://huggingface.co/anonymous9800694/kg-adapter-mistral-7b) | 279MB |

Each repo contains:
- `adapter_weights.pt` — KGEncoder + KGProjector + kg_to_hidden weights
- `vocab.json` — Entity and relation vocabulary (3,408 entities, 114 relations)

---

## Repository Structure

```
kg-adapter/
├── src/
│   ├── adapter/
│   │   ├── kg_adapter.py       # KGAdapter: frozen LLM + injection
│   │   ├── kg_encoder.py       # KGEncoder: 2-layer MLP over (S,P,O) triples
│   │   ├── kg_projector.py     # KGProjector: maps KG embedding to LLM hidden space
│   │   └── vocab.py            # Vocabulary builder
│   ├── graph/
│   │   ├── indexer.py          # Wikidata dump LMDB indexer
│   │   ├── extractor.py        # Triple extractor
│   │   ├── subgraph.py         # Subgraph builder per sample
│   │   └── label_fetcher.py    # Wikidata entity label fetcher
│   ├── data/
│   │   ├── dataset.py          # WikiWebQuestions dataset loader
│   │   └── collator.py         # PyTorch collator
│   ├── training/
│   │   └── trainer.py          # Training loop
│   ├── evaluation/
│   │   ├── evaluator.py        # Automatic metrics (TokenMatch, Substr)
│   │   ├── classify_using_gpt-v2.py  # GPT-based evaluation
│   │   └── latency_analysis.py

├── configs/
│   └── config.example.yaml     # Configuration template
├── data/
│   └── raw/
│       └── sample.json         # Sample test
├── kg-adapter.ipynb             # Reproducibility notebook (run on Colab)
├── requirements.txt
└── README.md
```

---

## Requirements

See `requirements.txt` for the full list.

---

## Citation

```bibtex
@article{anonymous2026wheninject,
  title  = {When to Inject: Selective Knowledge Graph Augmentation for Frozen Large Language Models},
  author = {Anonymous},
  year   = {2026},
}
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YR9vbWBV99ZQz6Q2jOcnsQQyZqzwGbra?usp=sharing)