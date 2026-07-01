# Customer Feedback Theme Engine

An embedding-first analytics workflow for turning unstructured customer feedback into traceable, evidence-backed theme insights.

## Why This Project Exists

Customer feedback often contains repeated product concerns, purchase drivers, and usability signals, but the raw text is difficult to measure or defend in business discussions.

This project shows how to transform review text into candidate themes, quantify exploratory signals, and draft evidence-backed insights without claiming production readiness or real company conclusions.

## What This Project Demonstrates

- Customer feedback analytics from unstructured review text.
- Data validation and local-only data handling.
- Sentence embeddings for semantic representation.
- UMAP + HDBSCAN theme discovery.
- TF-IDF / c-TF-IDF for explainability.
- Exploratory statistical signal layer with prevalence, rating association, effect size, uncertainty, and FDR correction.
- Mock-default LLM-assisted labeling and summarization.
- Evidence-backed reporting from representative reviews, keywords, and theme statistics.
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

Install dependencies first:

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

## Example Output

Synthetic example only, not real Amazon review results.

| theme_label | prevalence | rating_gap | adjusted_p_value | interpretation | suggested_action |
| --- | ---: | ---: | ---: | --- | --- |
| Damaged packaging | 0.18 | -0.72 | 0.012 | Frequent theme with lower associated ratings in the toy sample. | Review fulfillment handoff points and packaging QA notes. |
| Fragrance preference | 0.14 | 0.31 | 0.084 | Moderate prevalence with weak positive association in the toy sample. | Keep fragrance language clear in product copy and examples. |
| Product size mismatch | 0.09 | -0.58 | 0.041 | Smaller theme with lower associated ratings in the toy sample. | Improve size guidance and expectation-setting content. |
| Repeat purchase driver | 0.22 | 0.64 | 0.009 | Frequent theme with higher associated ratings in the toy sample. | Preserve the attributes customers cite as repeat-purchase reasons. |

Additional synthetic-only example files are in `examples/`.

## Current Status

Phase 6 is complete after this portfolio packaging commit. The repository now includes a polished README, safe synthetic examples, interview-defense notes, LinkedIn-ready project assets, and portfolio positioning materials.

## Data Policy

Raw external datasets and processed real datasets are not redistributed in this repository. Generated artifacts from real data are ignored by default, including local review samples, embeddings, theme outputs, statistical signal outputs, and generated insight reports.

Secrets and API keys must never be committed. The default LLM workflow uses a deterministic local mock provider and does not require an external API call. Tracked examples are synthetic only and are clearly labeled as synthetic.

## Scope and Limitations

This is a portfolio project, not production SaaS. The statistical layer is exploratory, not causal. Theme discovery and rating association are useful for prioritization, but they should not be treated as confirmatory proof.

The LLM provider is mock by default. A live API integration would require environment variables, secret handling, provider-specific review, and additional validation. A real deployment would also require data governance, monitoring, evaluation, quality review, privacy review, and stakeholder sign-off.

This repository does not claim that findings represent Amazon, any company, or any real business conclusion.

## How to Talk About This Project

One-sentence version:

I built an embedding-first customer feedback analytics pipeline that discovers review themes, quantifies exploratory rating signals, and drafts evidence-backed insight reports with a mock-default LLM layer.

中文翻译：

我建立了一个以语义嵌入为核心的客户反馈分析流程，可以发现评论主题、量化探索性的评分信号，并用默认的模拟 LLM 层生成有证据支撑的洞察报告。

Three-sentence version:

This project turns unstructured product reviews into candidate customer themes using sentence embeddings, UMAP, and HDBSCAN. It adds explainability through representative reviews and TF-IDF keywords, then quantifies prevalence, rating association, effect size, uncertainty, and FDR-adjusted exploratory signals. The final layer uses a deterministic mock LLM provider to label and summarize evidence-backed clusters without making live API calls or unsupported business claims.

中文翻译：

这个项目使用句向量、UMAP 和 HDBSCAN，把非结构化产品评论转成候选客户主题。它通过代表性评论和 TF-IDF 关键词提供可解释性，并量化主题占比、评分关联、效果量、不确定性以及经过 FDR 校正的探索性信号。最后一层使用确定性的模拟 LLM 提供者，对有证据支撑的聚类进行命名和总结，不调用真实 API，也不做没有依据的商业结论。

Interview version:

I designed this as an analytics engineering project rather than a simple LLM wrapper. The pipeline validates local review data, cleans text conservatively, embeds review text with a local sentence-transformer model, uses UMAP + HDBSCAN to discover candidate semantic themes, and represents each theme with review examples and TF-IDF keywords. After discovery, it computes exploratory prevalence and rating-association signals with effect sizes, confidence intervals, and FDR correction, then passes only structured evidence into a mock-default LLM labeling layer. That separation lets me explain what the model discovered, what the statistics suggest, what the LLM summarized, and where the limitations remain.

中文翻译：

我把这个项目设计成一个分析工程项目，而不是简单的 LLM 包装器。整个流程会验证本地评论数据，保守地清洗文本，用本地句向量模型生成语义嵌入，再用 UMAP + HDBSCAN 发现候选语义主题，并用代表性评论和 TF-IDF 关键词解释每个主题。主题发现之后，系统会计算探索性的主题占比和评分关联信号，包括效果量、置信区间和 FDR 校正，然后只把结构化证据传给默认的模拟 LLM 命名层。这样的分层让我可以清楚说明哪些内容来自模型发现，哪些来自统计信号，哪些只是 LLM 的总结，以及项目仍有哪些限制。
