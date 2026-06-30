# Customer Feedback Theme Engine

An embedding-first analytics framework for discovering customer feedback themes
and turning them into evidence-backed product insights.

## Why This Project Exists

Product reviews, support tickets, and survey comments often contain the clearest
signals about customer needs, frustrations, and purchase drivers.

This project turns unstructured feedback into structured, evidence-backed themes
that can be reviewed, measured, and explained. It is built as a portfolio-ready
analytics engineering project for customer feedback analysis.

## What This Project Demonstrates

- Review data modeling and validation.
- Conservative text cleaning for customer feedback.
- Safe local ingestion for a public review dataset subset.
- Sentence embeddings for semantic representation.
- Dimensionality reduction and density-based clustering for theme discovery.
- TF-IDF or c-TF-IDF as explainability and keyword representation, not the main
  semantic method.
- Statistical signal analysis for prevalence, rating association, effect size,
  and uncertainty.
- LLM-assisted labeling and business-readable summaries, with source-review
  traceability.

## Methodology Overview

The planned pipeline starts with review text and rating outcomes. Reviews are
validated, cleaned conservatively, embedded into semantic vectors, projected into
a lower-dimensional space, and clustered with density-based methods.

Each cluster is represented with keywords and examples, evaluated through
statistical signals, and later labeled with LLM assistance. Final insights should
remain traceable to review evidence.

## Planned Roadmap

- Phase 1: Data ingestion and review schema.
- Phase 2: Sentence embeddings and semantic representation.
- Phase 3: UMAP and HDBSCAN theme discovery.
- Phase 4: Statistical signal layer.
- Phase 5: LLM-assisted labeling and insight generation.
- Phase 6: Portfolio packaging.

## Current Status

Phase 3 is complete. The repository includes local-only ingestion, schema
normalization, deterministic sampling, local sentence embeddings, and UMAP +
HDBSCAN candidate theme discovery.

Phase 4 is next: statistical signal analysis for exploratory prioritization.

## Phase 1 Data Ingestion

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
py -3 scripts/prepare_reviews.py `
  --category raw_review_All_Beauty `
  --sample-size 5000 `
  --seed 42 `
  --output data/processed/reviews_sample.parquet
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

## Phase 2 Sentence Embeddings

Phase 2 converts cleaned review text into dense semantic vectors with the local
open-source model `sentence-transformers/all-MiniLM-L6-v2` by default.
Embeddings are a representation layer for later clustering and diagnostics; they
are not business conclusions by themselves.

To embed a prepared local Phase 1 sample:

```powershell
py -3 scripts/embed_reviews.py `
  --input data/processed/reviews_sample.parquet `
  --output-dir data/processed/embeddings `
  --prefix reviews_sample `
  --limit 500
```

Generated embedding artifacts are written under ignored local paths such as
`data/processed/embeddings/` and must not be committed. Phase 2 does not perform
UMAP, HDBSCAN, clustering, statistical tests, or LLM labeling.

TF-IDF or c-TF-IDF remains useful in later phases for explainability and cluster
keyword representation, not as the main semantic clustering method.

## Phase 3 Theme Discovery

Phase 3 applies UMAP to reduce high-dimensional sentence embeddings before
clustering. HDBSCAN then discovers dense groups of semantically similar reviews
and marks low-density points with label `-1` as noise.

Cluster labels are initial candidate themes. They are not final business
conclusions, and Phase 3 does not perform statistical significance testing,
rating association analysis, or LLM labeling.

The workflow keeps each cluster traceable to review IDs and source review text.
It also produces deterministic representative examples, cluster diagnostics, and
TF-IDF keyword representations for explainability.

To run local candidate theme discovery after Phase 2 embedding generation:

```powershell
py -3 scripts/discover_themes.py `
  --reviews-input data/processed/reviews_sample.parquet `
  --ids-input data/processed/embeddings/reviews_sample_ids.json `
  --embeddings-input data/processed/embeddings/reviews_sample_embeddings.npy `
  --output-dir data/processed/themes
```

Generated theme outputs are written under ignored local paths such as
`data/processed/themes/` and must not be committed.

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
