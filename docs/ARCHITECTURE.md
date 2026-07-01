# Architecture

## Pipeline

```text
Raw reviews
  -> Local ingestion and schema normalization
  -> Data validation
  -> Text cleaning
  -> Sentence embeddings
  -> Dimensionality reduction
  -> HDBSCAN clustering
  -> Theme representation
  -> Statistical signal layer
  -> LLM labeling
  -> Evidence-backed insight report
```

## Components

### Raw Reviews

Reviews are expected to contain text plus an outcome such as rating. Phase 1
starts with the `raw_review_All_Beauty` category from
`McAuley-Lab/Amazon-Reviews-2023` for local experimentation. Raw external
datasets and processed real review samples are kept outside git.

### Local Ingestion and Schema Normalization

The ingestion layer streams or loads a selected Amazon Reviews 2023 category,
takes a deterministic local sample, and writes processed output only under
`data/processed/`.

Hugging Face remote dataset code execution is disabled by default. Users can opt
in with `--trust-remote-code` only after reviewing and trusting the dataset
source.

The normalized Phase 1 schema is:

```text
review_id, product_id, parent_product_id, user_id, review_text, review_title,
rating, helpful_vote, verified_purchase, timestamp
```

If a raw row does not include `review_id`, the project generates a deterministic
ID from stable row fields. Missing optional fields remain nullable and are
surfaced through validation metrics.

### Data Validation

Validation checks required normalized columns, empty cleaned review text, numeric
ratings between 1 and 5 when present, missing rating counts, rating
distribution, duplicate review IDs, and verified-purchase distribution before
downstream processing.

### Text Cleaning

Cleaning is conservative. The goal is to normalize obvious formatting issues
without stripping meaning-heavy punctuation or customer language.

### Sentence Embeddings

The embedding model converts review text into dense semantic vectors. These
vectors are the primary representation for discovering themes because they
capture similarity beyond exact keyword overlap.

Phase 2 uses `sentence-transformers/all-MiniLM-L6-v2` as the default local
open-source model. Embeddings are generated from prepared review text and saved
as ignored local artifacts under `data/processed/embeddings/`.

Embeddings support later semantic clustering and diagnostics. They do not
produce business conclusions by themselves.

### UMAP Dimensionality Reduction

UMAP projects high-dimensional embeddings into a lower-dimensional space that is
more suitable for density-based clustering and diagnostics.

Phase 3 applies UMAP to local sentence embeddings before clustering. The reduced
coordinates are intermediate local artifacts and should not be treated as
business findings by themselves.

### HDBSCAN Clustering

HDBSCAN discovers dense semantic groups without requiring a fixed number of
clusters. It can also identify noise points that do not fit a strong theme.

Phase 3 uses HDBSCAN labels as initial candidate theme IDs. Label `-1` is kept
as noise or outlier reviews and excluded from representative-review and keyword
summaries.

### Theme Representation

Theme representation combines cluster examples, keywords, and summary
statistics. TF-IDF or c-TF-IDF may be used here to explain clusters and surface
distinctive terms.

Phase 3 provides a first representation layer with cluster assignments,
diagnostics, deterministic representative reviews, and TF-IDF keywords. These
outputs remain traceable to review IDs and source review text.

### Statistical Signal Layer

The statistical layer quantifies theme prevalence, rating association, effect
size, confidence intervals, and exploratory uncertainty. This layer helps
separate frequent themes from themes that are meaningfully associated with
customer outcomes.

Phase 4 computes non-noise theme prevalence with Wilson confidence intervals,
rating association with Mann-Whitney U tests, rank-biserial effect sizes, and
Benjamini-Hochberg FDR-adjusted p-values.

These outputs are exploratory prioritization signals. Themes are discovered from
the data before testing, so the rating associations are not confirmatory proof
and should not be interpreted causally. A stronger future design can discover
themes on one split and validate associations on a holdout split.

### LLM Labeling

LLMs are used for cluster labels, concise summaries, and business-readable
insight generation. They operate on curated cluster evidence and do not replace
embedding-based discovery or statistical analysis.

Phase 5 implements an evidence-bounded labeling and insight drafting layer. The
default provider is a deterministic mock, so the workflow is reproducible and
does not require external API keys.

The labeling prompt includes only cluster IDs, TF-IDF keywords, representative
review snippets, prevalence, uncertainty, rating gap, adjusted p-value,
interpretation labels, and caution flags. The prompt explicitly instructs the
model not to invent evidence, claim causality, or override statistical results.

### Evidence-Backed Insight Report

Final outputs should connect each insight to source review examples, theme
statistics, and documented limitations.

Phase 5 renders draft JSON and Markdown insight reports from structured evidence
and validated label outputs. These reports remain local generated artifacts
unless explicitly approved for safe publication.

## Method Roles

- Embedding model role: provide semantic representations of review text.
- Clustering algorithm role: discover recurring customer themes from embedding
  geometry.
- TF-IDF / c-TF-IDF role: support keyword representation and explainability.
- Statistical layer role: quantify prevalence, association, effect size, and
  uncertainty for exploratory prioritization.
- LLM role: improve readability through labeling and summarization from
  supplied evidence only.

## Portfolio Packaging

Phase 6 adds public project documentation around the existing pipeline. The
tracked examples under `examples/` are synthetic only and demonstrate report
shape, not real review findings.
