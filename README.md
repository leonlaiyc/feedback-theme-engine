# Feedback Theme Engine

A reusable NLP pipeline that converts unstructured text (reviews, complaints, transaction notes)
into structured business insight via embeddings → clustering → statistical validation → LLM labeling.

## Design Philosophy

**General engine + one demo scenario.** The core pipeline is domain-agnostic. The demo uses Amazon
product reviews (Home & Kitchen category) as a concrete, public dataset with diverse complaint types.

---

## Honest Scope

This section marks what is AI-assisted versus a deliberate design decision.
It will be kept up-to-date as each phase is completed.

| Decision | Origin | Notes |
|---|---|---|
| Category: Home_and_Kitchen | Human (deliberate) | Mid-size, diverse complaint themes (appliances, cookware, storage) |
| Text: title + body concatenation | Human (deliberate) | Title captures the core complaint; body has context |
| Min-word threshold: 5 | Human (deliberate) | Trades recall for signal quality; affects later prevalence counts |
| Random seed: 42 | Conventional default | Reproducibility only |
| Downsample target: 8 000 | Human (deliberate) | Large enough for meaningful clusters; small enough for free-tier LLM labeling |

---

## Project Structure

```
feedback-theme-engine/
├── src/
│   └── data_prep.py      # Phase 0: data acquisition, cleaning, sanity checks
├── notebooks/            # Optional exploratory notebooks (never a source of truth)
├── data/
│   ├── raw/              # gitignored — never committed
│   └── processed/        # gitignored — never committed
├── requirements.txt
└── README.md
```

---

## Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Data acquisition, cleaning, sanity checks | ✅ done |
| 1 | Embedding generation | planned |
| 2 | Clustering | planned |
| 3 | Statistical validation | planned |
| 4 | LLM cluster labeling | planned |
| 5 | Business insight report | planned |

---

## Quickstart

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
python src/data_prep.py
```

Outputs land in `data/processed/`. All paths are relative to the project root.

---

## Reproducibility

All scripts use a fixed random seed (`RANDOM_SEED = 42`). Dependencies are pinned in
`requirements.txt`. The script runs end-to-end from a clean checkout with no manual steps.
