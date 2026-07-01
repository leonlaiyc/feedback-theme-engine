# Project Spec

## Problem Statement

Customer feedback is often stored as unstructured text, making it difficult to
identify recurring product themes, quantify their importance, and connect
qualitative evidence to measurable customer outcomes such as ratings.

This project defines a defensible workflow for transforming product reviews into
structured, evidence-backed customer themes and decision-ready insights.

## Target User

The target user is an analytics, product, or data science stakeholder who needs
to understand what customers are repeatedly saying, how those themes relate to
rating outcomes, and which themes deserve attention.

## Portfolio Goal

The portfolio goal is to demonstrate applied analytics engineering, customer
analytics thinking, semantic NLP methods, statistical signal interpretation, and
responsible LLM usage in one coherent project.

## Inputs and Outputs

Planned inputs:

- Review identifier.
- Product identifier or grouping field.
- Review text.
- Rating or outcome measure.
- Optional metadata in later phases.

Planned outputs:

- Validated review dataset.
- Cleaned review text.
- Embedding representations.
- Discovered theme clusters.
- Cluster diagnostics and representative review examples.
- Theme-level keyword and example evidence.
- Theme prevalence, uncertainty, rating association, effect size, and
  multiple-testing-adjusted signal statistics.
- LLM-assisted theme labels and summaries.
- Evidence-backed insight report.

## Core Methodology

1. Validate review data and required schema.
2. Clean text conservatively while preserving meaning.
3. Encode reviews with sentence embeddings.
4. Reduce embedding dimensionality for clustering.
5. Discover dense semantic groups with HDBSCAN.
6. Represent themes with keywords, examples, and source evidence.
7. Quantify theme prevalence, rating association, effect size, uncertainty, and
   multiple-testing-adjusted exploratory signals.
8. Use LLMs only for cluster labeling, theme summaries, and readable insight
   generation.
9. Keep final insights traceable to source reviews.

## Non-Goals

- Building a production SaaS application.
- Claiming conclusions about any named company.
- Treating LLM summaries as the primary discovery method.
- Replacing semantic clustering with simple keyword matching.
- Committing raw datasets or real processed data.
- Running paid APIs during Phase 0.

## Success Criteria

- The project has a clean, testable Python structure.
- Methodology is documented clearly enough to defend in interviews.
- Data and secrets policies are explicit and enforced through `.gitignore`.
- Every planned insight type has a path back to source review evidence.
- Statistical claims are framed with appropriate uncertainty and limitations.
- Theme-rating signals are treated as exploratory prioritization, not causal or
  confirmatory proof.

## Interview Defend Points

- Why embeddings are used for semantic representation.
- Why TF-IDF or c-TF-IDF is used for representation and explainability rather
  than primary discovery.
- Why density-based clustering is appropriate for uneven theme sizes and noise.
- How theme prevalence and rating association are quantified.
- How uncertainty and exploratory limitations are communicated.
- Why LLMs are restricted to labeling and summarization rather than replacing
  the pipeline.
