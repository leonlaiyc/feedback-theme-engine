# Portfolio Summary

## Target Roles

- Data Analyst with strong Python and product analytics skills.
- Product Analyst focused on customer feedback, reviews, and decision support.
- Analytics Engineer who can build reproducible analysis workflows.
- Applied Data Scientist working with NLP, clustering, and statistical interpretation.
- AI Product Analyst who can use LLMs responsibly without replacing measurement.

## Capability by Phase

| Phase | Capability demonstrated |
| --- | --- |
| Phase 1 | Safe local ingestion, schema normalization, validation, deterministic sampling, and data policy discipline. |
| Phase 2 | Sentence embedding workflow, local model usage, artifact handling, and semantic representation. |
| Phase 3 | UMAP + HDBSCAN theme discovery, noise handling, cluster diagnostics, representative reviews, and TF-IDF explainability. |
| Phase 4 | Exploratory statistical signal layer with prevalence, confidence intervals, rating gaps, effect sizes, and FDR correction. |
| Phase 5 | Evidence-bounded mock LLM labeling, safe prompt design, structured output parsing, and insight report rendering. |
| Phase 6 | Portfolio packaging, synthetic examples, interview defense, public profile materials, and data-safety audit. |

## How This Differs From a Simple LLM Wrapper

This project does not ask an LLM to read raw reviews and produce unsupported conclusions. The main discovery path uses embeddings, dimensionality reduction, and clustering. The measurement layer calculates explicit statistical signals. The LLM layer receives structured evidence and is limited to labeling and summarization.

That design makes the work easier to audit in an interview. It is possible to explain which parts are data validation, which parts are semantic modeling, which parts are statistics, and which parts are language generation.

## Honest Limitations

- This is a portfolio project, not production SaaS.
- The example outputs tracked in the repo are synthetic only.
- Real generated artifacts are intentionally ignored and should not be committed.
- Theme-rating signals are exploratory because discovery and testing happen on the same data.
- The default LLM provider is a deterministic mock, not a live external model.
- A production deployment would need data governance, privacy review, monitoring, evaluation, stakeholder review, and secret-safe provider integration.

## Suggested GitHub Repository Description

Embedding-first customer feedback analytics pipeline for theme discovery, exploratory rating signals, and evidence-backed mock LLM insight drafts.

## Suggested GitHub Topics

- customer-feedback
- product-analytics
- analytics-engineering
- natural-language-processing
- sentence-embeddings
- clustering
- umap
- hdbscan
- statistical-analysis
- responsible-ai
- portfolio-project
