# Architecture

## Planned Pipeline

```text
Raw reviews
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

Reviews are expected to contain text plus an outcome such as rating. Raw external datasets are kept outside git.

### Data Validation

Validation checks required columns, basic data types, missing values, and expected rating behavior before downstream processing.

### Text Cleaning

Cleaning is conservative. The goal is to normalize obvious formatting issues without stripping meaning-heavy punctuation or customer language.

### Sentence Embeddings

The embedding model converts review text into dense semantic vectors. These vectors are the primary representation for discovering themes because they capture similarity beyond exact keyword overlap.

### Dimensionality Reduction

UMAP is planned to project high-dimensional embeddings into a lower-dimensional space that is more suitable for density-based clustering and diagnostics.

### HDBSCAN Clustering

HDBSCAN is planned to discover dense semantic groups without requiring a fixed number of clusters. It can also identify noise points that do not fit a strong theme.

### Theme Representation

Theme representation combines cluster examples, keywords, and summary statistics. TF-IDF or c-TF-IDF may be used here to explain clusters and surface distinctive terms.

### Statistical Signal Layer

The statistical layer quantifies theme prevalence, rating association, effect size, confidence intervals, and exploratory uncertainty. This layer helps separate frequent themes from themes that are meaningfully associated with customer outcomes.

### LLM Labeling

LLMs are planned for cluster labels, concise summaries, and business-readable insight generation. They should operate on curated cluster evidence and should not replace embedding-based discovery or statistical analysis.

### Evidence-Backed Insight Report

Final outputs should connect each insight to source review examples, theme statistics, and documented limitations.

## Method Roles

- Embedding model role: provide semantic representations of review text.
- Clustering algorithm role: discover recurring customer themes from embedding geometry.
- TF-IDF / c-TF-IDF role: support keyword representation and explainability.
- Statistical layer role: quantify prevalence, association, effect size, and uncertainty.
- LLM role: improve readability through labeling and summarization from evidence.

