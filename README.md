# Customer Feedback Theme Engine

An embedding-first analytics framework for discovering customer feedback themes and turning them into evidence-backed product insights.

## Why This Project Exists

Product reviews, support tickets, and survey comments often contain the clearest
signals about customer needs, frustrations, and purchase drivers.

The hard part is not summarizing comments one by one. The hard part is building
a defensible workflow that groups semantically similar feedback, quantifies
which themes matter, and keeps every conclusion tied to source evidence.

This project demonstrates a practical approach to customer feedback analytics for portfolio and interview settings.

## What This Project Demonstrates

- Review data modeling and validation.
- Conservative text cleaning for customer feedback.
- Safe local ingestion for a public review dataset subset.
- Sentence embeddings for semantic representation.
- Dimensionality reduction and density-based clustering for theme discovery.
- TF-IDF or c-TF-IDF as explainability and keyword representation, not the main semantic method.
- Statistical signal analysis for prevalence, rating association, effect size, and uncertainty.
- LLM-assisted labeling and business-readable summaries, with source-review traceability.

## Methodology Overview

The planned pipeline starts with review text and rating outcomes. Reviews are
validated, cleaned conservatively, embedded into semantic vectors, projected into
a lower-dimensional space, and clustered with density-based methods.

Each cluster is represented with keywords and examples, evaluated through
statistical signals, and later labeled with LLM assistance. Final insights should
remain traceable to review evidence.

## Current Status

Phase 1 is complete. The repository now includes local-only ingestion, schema
normalization, validation summaries, deterministic sampling, and a lightweight
preparation script for a small Amazon Reviews 2023 category sample.

Phase 2 is next: sentence embeddings and semantic representation.

## Phase 1: Data Ingestion

Phase 1 uses the public research dataset
`McAuley-Lab/Amazon-Reviews-2023` for local experimentation. The starting subset
is `raw_review_All_Beauty` because it is relatively manageable and includes
review text plus a rating outcome.

That pairing enables later theme-to-rating signal analysis, but Phase 1 does not
create embeddings, clusters, statistical tests, LLM labels, or business
conclusions.

Raw review files and processed real review samples are not redistributed in this
repository. The local preparation script writes processed samples only under
ignored data paths such as `data/processed/`.

To prepare a small local sample after installing dependencies:

```powershell
py -3 scripts/prepare_reviews.py --category raw_review_All_Beauty --sample-size 5000 --seed 42 --output data/processed/reviews_sample.parquet
```

Some Hugging Face datasets may require remote dataset loading code. This project
keeps that disabled by default. Use `--trust-remote-code` only after reviewing
and trusting the dataset source.

The normalized schema is:

```text
review_id, product_id, parent_product_id, user_id, review_text, review_title,
rating, helpful_vote, verified_purchase, timestamp
```

If the raw dataset does not provide `review_id`, the ingestion code creates a
deterministic generated ID from stable row fields. Missing optional fields are
retained as nullable values so downstream validation can report data quality
honestly.

## Planned Roadmap

- Phase 1: Data ingestion and review schema.
- Phase 2: Sentence embeddings and semantic representation.
- Phase 3: UMAP and HDBSCAN theme discovery.
- Phase 4: Statistical signal layer.
- Phase 5: LLM-assisted labeling and insight generation.
- Phase 6: Portfolio packaging.

## Data Policy Summary

Raw external datasets, real processed datasets, API keys, tokens, credentials,
and `.env` files must not be committed. Only toy samples, schema examples, and
explicitly approved derived summaries should be tracked.

This repository does not include raw Amazon review files, processed real review
samples, company-specific proprietary data, or confidential data.

## Scope and Limitations

This is an early-stage portfolio project and demonstration framework. It is not
a production SaaS product, not a company-specific analysis, and not evidence of
real-world business conclusions for any named company. Results produced in later
phases should be treated as exploratory unless the methodology and data support
stronger claims.
