# Roadmap

## Phase 0: Project Foundation

Goal: Establish repository structure, documentation, safety policies, minimal Python skeleton, and tests.

Deliverables:

- README, project spec, roadmap, architecture, data policy, and portfolio positioning docs.
- Repo-scoped Codex skills.
- Minimal package skeleton.
- Basic schema and text-cleaning tests.

Interview defendable:

- Project scope, methodology, and data boundaries.
- Why this is an analytics engineering project rather than a simple LLM demo.

Exit criteria:

- Foundation files committed.
- Compile and tests pass or blockers are documented.
- Data and secrets guardrails are in place.

## Phase 1: Data Ingestion and Review Schema

Status: complete.

Goal: Define safe ingestion patterns and validate review data without committing raw datasets.

Deliverables:

- Local ingestion script for `McAuley-Lab/Amazon-Reviews-2023`.
- Starting category: `raw_review_All_Beauty`.
- Explicit `trust_remote_code` opt-in for Hugging Face dataset loading.
- Normalized review schema with review text and rating outcome.
- Required-column, rating-range, empty-text, duplicate-ID, and distribution validation.
- Deterministic sampling with optional minimum text length.
- Toy tests only; no real dataset fixtures in git.
- Clear separation of raw and processed ignored paths.

Interview defendable:

- How data quality issues are detected.
- Why raw and processed real data are excluded from git.
- Why ingestion starts with All_Beauty and stops before modeling.

Exit criteria:

- Ingestion can prepare a local sample from a user-accessible dataset source.
- Validation errors are clear and tested with synthetic data.
- No raw or processed real review files are tracked.
- Phase 1 does not implement embeddings, clustering, statistical tests, or LLM labeling.

## Phase 2: Sentence Embeddings and Semantic Representation

Status: next.

Goal: Represent review text with sentence embeddings.

Deliverables:

- Embedding interface.
- Local model configuration.
- Embedding artifact policy.
- Small smoke tests or toy examples.

Interview defendable:

- Why sentence embeddings are better suited than bag-of-words for semantic grouping.
- How embedding artifacts are handled safely.

Exit criteria:

- Toy reviews can be embedded locally.
- No large embedding artifacts are committed.

## Phase 3: UMAP + HDBSCAN Theme Discovery

Goal: Discover semantic feedback themes using dimensionality reduction and density-based clustering.

Deliverables:

- UMAP projection module.
- HDBSCAN clustering module.
- Cluster diagnostics.
- Noise handling strategy.

Interview defendable:

- Why UMAP is used before clustering.
- Why HDBSCAN is appropriate for unknown, uneven theme shapes.
- How outliers and noise are interpreted.

Exit criteria:

- Toy or local data can produce clusters.
- Cluster outputs include evidence examples.

## Phase 4: Statistical Signal Layer

Goal: Quantify which themes matter and how they relate to rating outcomes.

Deliverables:

- Theme prevalence metrics.
- Rating association measures.
- Effect sizes and confidence intervals.
- Multiple testing correction guidance.
- Exploratory versus confirmatory interpretation notes.

Interview defendable:

- How prevalence differs from importance.
- Why uncertainty matters in theme prioritization.
- How to avoid overstating exploratory results.

Exit criteria:

- Theme metrics are computed and tested on toy data.
- Outputs include appropriate caveats.

## Phase 5: LLM-Assisted Labeling and Insight Generation

Goal: Use LLMs to label clusters and generate readable insights while preserving evidence traceability.

Deliverables:

- Prompt templates.
- Cluster labeling workflow.
- Source-evidence citation format.
- Business-readable insight report generator.

Interview defendable:

- Why LLMs are not used as the primary clustering mechanism.
- How hallucination risk is reduced.
- How every insight links back to examples.

Exit criteria:

- Labels and summaries are generated from cluster evidence.
- No paid API call is required by default.

## Phase 6: Portfolio Packaging

Goal: Package the project for recruiter and hiring-manager review.

Deliverables:

- Polished README.
- Example report from toy or approved data.
- Architecture diagram.
- Methodology notes and limitations.
- Reproducible demo path.

Interview defendable:

- End-to-end story from data to decision-ready insight.
- Tradeoffs, limitations, and next steps.

Exit criteria:

- Portfolio presentation is coherent, honest, and reproducible.
