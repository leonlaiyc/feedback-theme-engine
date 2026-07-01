# Customer Feedback Theme Engine

An embedding-first Python workflow for turning unstructured customer feedback into traceable, evidence-backed theme insights.

## Project Summary

Customer Feedback Theme Engine validates local review data, embeds review text, discovers candidate feedback themes, measures exploratory rating signals, and renders evidence-backed insight drafts.

The repository is designed as a public portfolio project that demonstrates a defensible analytics workflow. It does not redistribute real review data, does not require paid APIs by default, and does not claim production readiness or real company conclusions.

## Why This Project Exists

Customer reviews, support tickets, and survey comments often contain repeated product issues and purchase drivers. Raw text alone is difficult to group, measure, prioritize, or explain.

This project shows one practical way to move from unstructured feedback to structured theme insights while preserving data boundaries, source evidence, and statistical caution.

## What This Project Demonstrates

- Customer feedback analytics from unstructured review text.
- Data validation and local-only data handling.
- Conservative text cleaning that preserves customer meaning.
- Sentence embeddings for semantic representation.
- UMAP + HDBSCAN for candidate theme discovery.
- TF-IDF / c-TF-IDF for cluster explainability.
- Exploratory statistical signals for prevalence, rating association, effect size, uncertainty, and FDR correction.
- Mock-default LLM-assisted labeling and summarization from structured evidence.
- Evidence-backed reporting with representative reviews, keywords, and theme statistics.
- Reproducible CLI workflow and tested Python modules.

## Methodology Overview

The pipeline follows this path:

```text
Raw reviews
  -> schema validation
  -> conservative text cleaning
  -> sentence embeddings
  -> UMAP
  -> HDBSCAN
  -> representative reviews and TF-IDF keywords
  -> prevalence and rating-association signals
  -> mock-default LLM-assisted labels
  -> evidence-backed insight report
```

Sentence embeddings provide the semantic representation used for similarity and clustering. HDBSCAN creates candidate themes from the embedding geometry after UMAP projection. TF-IDF and c-TF-IDF support keyword explanation for clusters, but they are not the main clustering method.

The statistical layer is an exploratory prioritization layer. It measures prevalence, rating association, effect size, uncertainty, and multiple-testing-adjusted p-values, but it does not prove causality. The LLM layer labels and summarizes evidence-backed clusters; it does not discover clusters, override statistics, or invent supporting evidence.

## End-to-End Workflow

Install dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

Prepare a local review sample:

```powershell
py -3 scripts/prepare_reviews.py `
  --category raw_review_All_Beauty `
  --sample-size 5000 `
  --seed 42 `
  --output data/processed/reviews_sample.parquet
```

Generate sentence embeddings:

```powershell
py -3 scripts/embed_reviews.py `
  --input data/processed/reviews_sample.parquet `
  --output-dir data/processed/embeddings `
  --prefix reviews_sample `
  --limit 500
```

Discover candidate themes:

```powershell
py -3 scripts/discover_themes.py `
  --reviews-input data/processed/reviews_sample.parquet `
  --ids-input data/processed/embeddings/reviews_sample_ids.json `
  --embeddings-input data/processed/embeddings/reviews_sample_embeddings.npy `
  --output-dir data/processed/themes
```

Analyze exploratory statistical signals:

```powershell
py -3 scripts/analyze_theme_signals.py `
  --cluster-input data/processed/themes/cluster_assignments.parquet `
  --output-dir data/processed/themes `
  --min-theme-size 10
```

Label theme insights with the deterministic mock provider:

```powershell
py -3 scripts/label_theme_insights.py `
  --signals-input data/processed/themes/theme_signals.csv `
  --representatives-input data/processed/themes/representative_reviews.csv `
  --keywords-input data/processed/themes/cluster_keywords.csv `
  --output-dir data/processed/themes `
  --provider mock
```

Generated artifacts are written under ignored local paths such as `data/processed/`, `data/processed/embeddings/`, and `data/processed/themes/`. Real review samples, embeddings, clusters, statistical outputs, and generated insight reports should not be committed.

## Synthetic Example Output

Synthetic example only, not real Amazon review results.

| theme_label | prevalence | rating_gap | adjusted_p_value | interpretation | suggested_action |
| --- | ---: | ---: | ---: | --- | --- |
| Damaged packaging | 0.18 | -0.72 | 0.012 | Frequent theme with lower associated ratings in the toy sample. | Review fulfillment handoff points and packaging QA notes. |
| Fragrance preference | 0.14 | 0.31 | 0.084 | Moderate prevalence with weak positive association in the toy sample. | Keep fragrance language clear in product copy and examples. |
| Product size mismatch | 0.09 | -0.58 | 0.041 | Smaller theme with lower associated ratings in the toy sample. | Improve size guidance and expectation-setting content. |
| Repeat purchase driver | 0.22 | 0.64 | 0.009 | Frequent theme with higher associated ratings in the toy sample. | Preserve the attributes customers cite as repeat-purchase reasons. |

Additional synthetic-only example files are available in `examples/`.

## Data Policy

Raw external datasets and processed real datasets are not redistributed in this repository. Generated artifacts from real data are ignored by default, including local review samples, embeddings, theme outputs, statistical signal outputs, and generated insight reports.

Secrets and API keys must never be committed. The default LLM workflow uses a deterministic local mock provider and does not require an external API call. Tracked examples are synthetic only and are clearly labeled as synthetic.

## Scope and Limitations

This is a portfolio project, not production SaaS. The statistical layer is exploratory, not causal. Theme discovery and rating association are useful for prioritization, but they should not be treated as confirmatory proof.

The LLM provider is mock by default. A live API integration would require environment variables, secret handling, provider-specific review, and additional validation. A real deployment would also require data governance, monitoring, evaluation, quality review, privacy review, and stakeholder sign-off.

This repository does not claim that findings represent Amazon, any company, or any real business conclusion.
