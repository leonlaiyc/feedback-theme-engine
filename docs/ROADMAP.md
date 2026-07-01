# Roadmap

## Phase 0: Project Foundation

Goal: Establish repository structure, documentation, safety policies, minimal
Python skeleton, and tests.

Deliverables:

- README, project spec, roadmap, architecture, data policy, and portfolio
  positioning docs.
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

Goal: Define safe ingestion patterns and validate review data without committing
raw datasets.

Deliverables:

- Local ingestion script for `McAuley-Lab/Amazon-Reviews-2023`.
- Starting category: `raw_review_All_Beauty`.
- Explicit `trust_remote_code` opt-in for Hugging Face dataset loading.
- Normalized review schema with review text and rating outcome.
- Required-column, rating-range, empty-text, duplicate-ID, and distribution
  validation.
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
- Phase 1 does not implement embeddings, clustering, statistical tests, or LLM
  labeling.

## Phase 2: Sentence Embeddings and Semantic Representation

Status: complete.

Goal: Represent review text with sentence embeddings.

Deliverables:

- Embedding interface using `sentence-transformers`.
- Local default model: `sentence-transformers/all-MiniLM-L6-v2`.
- CLI script for embedding a prepared local Phase 1 sample.
- Embedding artifact policy for ignored local `.npy` and ID files.
- Similarity sanity helper for toy examples.
- Tests with fake models only; no model downloads required.

Interview defendable:

- Why sentence embeddings are better suited than bag-of-words for semantic
  grouping.
- How embedding artifacts are handled safely.
- Why embeddings are a representation layer rather than business conclusions.
- Why TF-IDF or c-TF-IDF remains useful later for explainability.

Exit criteria:

- Toy reviews can be embedded locally.
- No large embedding artifacts are committed.
- Phase 2 does not implement UMAP, HDBSCAN, clustering, statistical tests, or
  LLM labeling.

## Phase 3: UMAP + HDBSCAN Theme Discovery

Status: complete.

Goal: Discover semantic feedback themes using dimensionality reduction and
density-based clustering.

Deliverables:

- UMAP projection module.
- HDBSCAN clustering module.
- Cluster diagnostics.
- Noise handling strategy for HDBSCAN label `-1`.
- Representative review examples per non-noise cluster.
- TF-IDF keyword extraction for cluster representation and explainability.
- CLI script for local-only theme discovery from Phase 2 artifacts.

Interview defendable:

- Why UMAP is used before clustering.
- Why HDBSCAN is appropriate for unknown, uneven theme shapes.
- How outliers and noise are interpreted.
- Why TF-IDF keywords explain clusters but do not drive semantic clustering.

Exit criteria:

- Toy or local data can produce clusters.
- Cluster outputs include evidence examples.
- Generated UMAP and cluster outputs are ignored local artifacts.
- Phase 3 does not implement statistical tests, rating association analysis, or
  LLM labeling.

## Phase 4: Statistical Signal Layer

Status: complete.

Goal: Quantify which themes matter and how they relate to rating outcomes.

Deliverables:

- Theme prevalence metrics.
- Rating association measures.
- Wilson confidence intervals for prevalence.
- Mann-Whitney U rating association tests.
- Rank-biserial effect sizes.
- Benjamini-Hochberg FDR multiple testing correction.
- Exploratory versus confirmatory interpretation notes.
- CLI script for local-only statistical signal outputs.

Interview defendable:

- How prevalence differs from importance.
- Why uncertainty matters in theme prioritization.
- How to avoid overstating exploratory results.
- Why post-discovery theme tests are exploratory signals.
- How a future discovery/validation split could strengthen inference.

Exit criteria:

- Theme metrics are computed and tested on toy data.
- Outputs include appropriate caveats.
- Generated statistical signal outputs are ignored local artifacts.
- Phase 4 does not implement LLM labeling or LLM summaries.

## Phase 5: LLM-Assisted Labeling and Insight Generation

Status: next.

Goal: Use LLMs to label clusters and generate readable insights while preserving
evidence traceability.

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
