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

### Run Analysis Scripts
 
```bash
cd src/analysis-v1
 
# GPT 7-point scoring injection analysis
python analyze-v1.py --input evaluated_final_cleaned_data.json
 
# Hallucination detection injection analysis
python analyze-v2.py --input evaluated_final_cleaned_data.json --metric no_halluc
 
# GPT 10-point injection analysis
python analyze-v3.py --input gpt_evaluated_combined-v2.json
```


## Dataset
 
We use **WikiWebQuestions (WWQ)** — a KGQA benchmark with Wikidata SPARQL queries.
 
- Train: 1,225 questions
- Validation: 176 questions
- Test: 352 questions
---
## Data and Evaluation Results

Pre-computed model answers, GPT evaluation scores, and Wikidata structural features for all 352 test samples are available for download:

📁 **[Download from Google Drive](https://drive.google.com/drive/folders/1RZrAL1rgEq-nybMfPFFtpYXC87aZ3PeZ?usp=share_link)**

The folder contains:
- `llama_answers.json` — LLaMA-3 8B base and adapter answers
- `mistral_answers.json` — Mistral-7B base and adapter answers
- `evaluated_final_cleaned_data.json`_ GPT-5.4 evaluation scores (7-point) and Categorical Evaluation along with Wikidata structural features (`entropy_risk`, `num_sitelinks`, `num_statements`, `num_references`, `wikidata_metadata`) for all 352 test entries.
- `gpt_evaluated_combined-v2.json` — GPT-5.4 evaluation scores (7-point and 10-point) with Wikidata structural features.

These files allow full reproduction of all paper results without running inference.

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
│   │   ├── kg_adapter.py            # KGAdapter: frozen LLM + hook-based injection
│   │   ├── kg_encoder.py            # KGEncoder: 2-layer MLP over (S,P,O) triples
│   │   ├── kg_projector.py          # KGProjector: maps KG embedding to LLM hidden space
│   │   └── vocab.py                 # Vocabulary builder
│   ├── graph/
│   │   ├── indexer.py               # Wikidata dump LMDB indexer
│   │   ├── extractor.py             # Triple extractor
│   │   ├── subgraph.py              # Subgraph builder per sample
│   │   └── label_fetcher.py         # Wikidata entity label fetcher
│   ├── data/
│   │   ├── dataset.py               # WikiWebQuestions dataset loader
│   │   └── collator.py              # PyTorch collator
│   ├── training/
│   │   └── trainer.py               # Training loop
│   ├── evaluation/
│   │   ├── evaluator.py             # Automatic metrics (Substr and TokMatch)
│   │   └── classify_using_gpt-v2.py # GPT-based evaluation (7-point + hallucination)
│   └── analysis-v1/
│       ├── analyze-v1.py            # GPT 7-point scoring analysis
│       ├── analyze-v2.py            # Hallucination detection analysis
│       └── analyze-v3.py            # GPT 10-point combined score analysis
├── configs/
│   └── config.example.yaml          # Configuration template
├── data/
│   └── raw/
│       └── sample.json              # Sample test entries with Wikidata triples
├── kg-adapter.ipynb                  # Reproducibility notebook (run on Colab)
├── Requirements.txt
└── README.md
```

---

## Requirements

See `requirements.txt` for the full list.

---
## License

This project is licensed under the Apache License 2.0.
© 2026 University of Tsukuba, National Institute of Advanced Industrial Science and Technology (AIST)
See the [LICENSE](LICENSE) file for full details.

## Citation

```bibtex
@article{ali2026wheninject,
  title  = {When to Inject: Gated Adapter-Based Selective Knowledge Graph Augmentation for Large Language Models},
  author = {Ali, Ushtar},
  year   = {2026},
}
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YR9vbWBV99ZQz6Q2jOcnsQQyZqzwGbra?usp=sharing)