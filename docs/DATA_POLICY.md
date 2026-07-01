# Data Policy

This repository must keep data handling conservative and explicit.

## What Must Not Be Committed

- Raw external datasets.
- Real processed datasets generated from external data.
- API keys, tokens, credentials, `.env` files, or other secrets.
- Large model artifacts or embedding outputs.
- Generated outputs from real data unless explicitly approved.

## What May Be Committed

- Toy samples.
- Schema examples.
- Small synthetic fixtures for tests.
- Derived summary outputs only when license-safe, small, and explicitly approved.

## Processed Data

Real processed datasets should not be committed unless they are small,
license-safe, and explicitly approved. The default is to regenerate processed
data locally from user-provided raw data.

## Environment Files

Do not add `.env` in Phase 0. A `.env.example` may be added later only if it is
useful and contains no real secrets.

## Dataset Acquisition

Phase 1 uses `McAuley-Lab/Amazon-Reviews-2023` as a public research dataset for
local experimentation. The initial subset is `raw_review_All_Beauty`, which
includes review text and rating outcomes useful for later exploratory
theme-to-rating analysis.

The repository must not redistribute raw Amazon review data or processed real
review samples. Users may prepare a local sample with
`scripts/prepare_reviews.py`, but generated files must remain under ignored
folders such as `data/raw/` or `data/processed/`.

Some Hugging Face datasets may require remote dataset loading code. The
preparation script keeps this disabled by default and exposes
`--trust-remote-code` as an explicit user opt-in. Enable it only after reviewing
and trusting the dataset source.

Phase 1 is limited to ingestion, schema normalization, validation, sampling, and
documentation. It does not produce embeddings, clusters, statistical tests,
LLM-generated labels, or business conclusions.

## Embedding Artifacts

Phase 2 generates local sentence embedding files for prepared review samples.
These files are derived from review text and must remain ignored local artifacts.
Do not commit `.npy` files, model caches, real processed review data, or ID files
generated from real review samples.

The default embedding output directory is `data/processed/embeddings/`, which is
covered by the repository data ignore rules. Phase 2 does not require paid APIs,
external API keys, clustering, statistical tests, or LLM labeling.

## Theme Discovery Artifacts

Phase 3 generates local UMAP, HDBSCAN, cluster assignment, representative review,
diagnostic, and keyword outputs from local embeddings and review text. These
outputs are derived from review data and must not be committed.

The default theme output directory is `data/processed/themes/`, which is covered
by the repository data ignore rules. Phase 3 does not perform statistical
significance testing, rating association analysis, Wilson confidence intervals,
LLM labeling, or LLM summarization.

## Statistical Signal Artifacts

Phase 4 generates local statistical signal outputs such as `theme_signals.csv`
and `theme_signals.json`. These files are derived from review ratings and theme
assignments and must not be committed.

The default signal output directory is `data/processed/themes/`, which is
covered by the repository data ignore rules. Phase 4 does not use paid APIs,
external API keys, LLM labeling, or LLM summarization.

Statistical outputs are exploratory. They should not be described as causal
impact, confirmatory proof, or real company conclusions.

## Insight Report Artifacts

Phase 5 generates local theme insight files such as `theme_insights.json` and
`theme_insights.md`. These files may contain review-derived snippets, labels,
and draft recommendations, so they must not be committed unless explicitly
approved and safe.

The default provider is `mock` and does not call any external API. Future live
API integrations must be optional, disabled by default, configured only through
environment variables, and must never hard-code or commit API keys.

LLM-assisted outputs are draft labels and summaries grounded in supplied
evidence. They should not be described as facts without evidence, causal impact,
or company-specific conclusions.

## Review Before Commit

Before committing, check git status and staged files for data or secrets. If
there is any doubt, do not commit the file.
