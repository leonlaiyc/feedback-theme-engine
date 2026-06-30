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

## Review Before Commit

Before committing, check git status and staged files for data or secrets. If
there is any doubt, do not commit the file.
