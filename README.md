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

Phase 5 is complete. The repository includes local-only ingestion, schema
normalization, deterministic sampling, local sentence embeddings, and UMAP +
HDBSCAN candidate theme discovery, plus exploratory statistical signals for
theme prioritization and mock-default insight drafting.

Phase 6 is next: portfolio packaging.

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

## Phase 4 Statistical Signals

Phase 4 quantifies candidate themes with prevalence, uncertainty, rating
association, effect size, and multiple-testing correction. The output supports
exploratory prioritization, not causal proof or confirmatory inference.

For each non-noise theme, the signal layer calculates:

- Review count and prevalence share.
- Wilson confidence interval for prevalence.
- Mean rating inside the theme and outside the theme.
- Rating gap between theme and non-theme reviews.
- Mann-Whitney U p-value as a default non-parametric association test.
- Rank-biserial effect size.
- Benjamini-Hochberg FDR-adjusted p-value.
- Cautious interpretation label.

To analyze a local Phase 3 cluster assignment file that includes ratings:

```powershell
py -3 scripts/analyze_theme_signals.py `
  --cluster-input data/processed/themes/cluster_assignments.parquet `
  --output-dir data/processed/themes `
  --min-theme-size 10
```

Themes are discovered from the same data being analyzed, so these tests are
post-discovery signals. A stronger future design can discover themes on one
split and validate associations on a holdout split. Generated signal outputs are
ignored local artifacts and must not be committed.

## Phase 5 Insight Drafting

Phase 5 uses LLM-style prompts for theme labeling and business-readable insight
drafts. The default provider is a deterministic local mock, so tests and local
development require no paid API, external key, or network call.

The LLM layer receives only structured evidence:

- Cluster ID.
- TF-IDF keywords.
- Representative review snippets.
- Prevalence and Wilson confidence interval.
- Rating gap, adjusted p-value, interpretation label, and caution flags.

LLM outputs are labels and draft explanations only. They do not create clusters,
override statistical signals, invent evidence, or make causal claims.

To draft local insights from Phase 3 and Phase 4 artifacts:

```powershell
py -3 scripts/label_theme_insights.py `
  --signals-input data/processed/themes/theme_signals.csv `
  --representatives-input data/processed/themes/representative_reviews.csv `
  --keywords-input data/processed/themes/cluster_keywords.csv `
  --output-dir data/processed/themes `
  --provider mock
```

Generated insight reports are ignored local artifacts and must not be committed
unless explicitly approved and safe.

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
