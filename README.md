# Customer Feedback Theme Engine

An embedding-first analytics framework for discovering customer feedback themes and turning them into evidence-backed product insights.

## Why This Project Exists

Product reviews, support tickets, and survey comments often contain the clearest signals about customer needs, frustrations, and purchase drivers. The hard part is not summarizing comments one by one. The hard part is building a defensible workflow that groups semantically similar feedback, quantifies which themes matter, and keeps every conclusion tied to source evidence.

This project demonstrates a practical approach to customer feedback analytics for portfolio and interview settings.

## What This Project Demonstrates

- Review data modeling and validation.
- Conservative text cleaning for customer feedback.
- Sentence embeddings for semantic representation.
- Dimensionality reduction and density-based clustering for theme discovery.
- TF-IDF or c-TF-IDF as explainability and keyword representation, not the main semantic method.
- Statistical signal analysis for prevalence, rating association, effect size, and uncertainty.
- LLM-assisted labeling and business-readable summaries, with source-review traceability.

## Methodology Overview

The planned pipeline starts with review text and rating outcomes. Reviews are validated, cleaned conservatively, embedded into semantic vectors, projected into a lower-dimensional space, and clustered with density-based methods. Each cluster is represented with keywords and examples, evaluated through statistical signals, and later labeled with LLM assistance. Final insights should remain traceable to review evidence.

## Current Status

Phase 0 initialized: repository foundation, documentation, data policy, repo-scoped Codex skills, minimal Python skeleton, and basic tests.

## Planned Roadmap

- Phase 1: Data ingestion and review schema.
- Phase 2: Sentence embeddings and semantic representation.
- Phase 3: UMAP and HDBSCAN theme discovery.
- Phase 4: Statistical signal layer.
- Phase 5: LLM-assisted labeling and insight generation.
- Phase 6: Portfolio packaging.

## Data Policy Summary

Raw external datasets, real processed datasets, API keys, tokens, credentials, and `.env` files must not be committed. Only toy samples, schema examples, and explicitly approved derived summaries should be tracked.

Dataset download steps will be documented in a later phase. This repository does not include company-specific, proprietary, or confidential data.

## Scope and Limitations

This is an early-stage portfolio project and demonstration framework. It is not a production SaaS product, not a company-specific analysis, and not evidence of real-world business conclusions for any named company. Results produced in later phases should be treated as exploratory unless the methodology and data support stronger claims.
