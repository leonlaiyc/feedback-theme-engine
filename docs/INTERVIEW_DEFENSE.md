# Interview Defense

## Project One-Liner

English answer:

I built an embedding-first customer feedback analytics pipeline that discovers review themes, quantifies exploratory rating signals, and drafts evidence-backed insight reports with a mock-default LLM layer.

中文翻译：

我建立了一个以语义嵌入为核心的客户反馈分析流程，可以发现评论主题、量化探索性的评分信号，并用默认的模拟 LLM 层生成有证据支撑的洞察报告。

## Architecture Explanation

English answer:

The architecture separates discovery, measurement, and communication. Local review data is validated and cleaned first, then sentence embeddings represent review meaning. UMAP projects embeddings into a clustering-friendly space, HDBSCAN discovers candidate themes, TF-IDF keywords and representative reviews explain those themes, the statistical layer measures exploratory prevalence and rating association, and the mock-default LLM layer turns structured evidence into readable labels and summaries.

中文翻译：

这个架构把发现、衡量和表达分开处理。本地评论数据先经过验证和保守清洗，然后用句向量表示评论语义。UMAP 把嵌入向量投影到更适合聚类的空间，HDBSCAN 发现候选主题，TF-IDF 关键词和代表性评论解释这些主题，统计层衡量探索性的主题占比和评分关联，默认的模拟 LLM 层再把结构化证据转成可读的标签和摘要。

## Phase-by-Phase Explanation

English answer:

- Phase 1 builds safe local ingestion, schema normalization, deterministic sampling, and validation.
- Phase 2 adds local sentence embeddings as the semantic representation layer.
- Phase 3 uses UMAP + HDBSCAN to discover candidate themes and represent them with examples and keywords.
- Phase 4 adds exploratory statistical signals: prevalence, confidence intervals, rating gaps, effect sizes, and FDR-adjusted p-values.
- Phase 5 adds mock-default LLM-assisted labeling and insight drafting from structured evidence.
- Phase 6 packages the project with recruiter-readable documentation, synthetic examples, and interview assets.

中文翻译：

- 第 1 阶段建立安全的本地数据导入、统一 schema、确定性抽样和数据验证。
- 第 2 阶段加入本地句向量，作为语义表示层。
- 第 3 阶段用 UMAP + HDBSCAN 发现候选主题，并用例子和关键词解释主题。
- 第 4 阶段加入探索性统计信号，包括主题占比、置信区间、评分差异、效果量和 FDR 校正后的 p 值。
- 第 5 阶段加入默认模拟 LLM 的主题命名和洞察草稿生成，输入只来自结构化证据。
- 第 6 阶段把项目包装成适合招聘方阅读、也适合面试答辩的作品集项目。

## Why Embeddings Instead of Only TF-IDF

English answer:

Embeddings are better for semantic grouping because customers can describe the same issue with different words. TF-IDF is still useful, but I use it for cluster explanation and keyword representation rather than primary discovery. That keeps semantic discovery and human-readable explanation in separate roles.

中文翻译：

句向量更适合语义分组，因为客户可能用不同词语描述同一个问题。TF-IDF 仍然有价值，但我把它用于解释聚类和提取关键词，而不是作为主要发现方法。这样可以把语义发现和可读解释分成两个清楚的角色。

## Why UMAP + HDBSCAN

English answer:

UMAP reduces high-dimensional embeddings into a space that is easier for density-based clustering to use. HDBSCAN is a good fit because customer themes may have uneven sizes, irregular shapes, and outliers, and it does not require choosing the number of clusters in advance.

中文翻译：

UMAP 会把高维嵌入向量降到更适合密度聚类的空间。HDBSCAN 适合这个问题，因为客户主题的大小可能不均匀、形状不规则，也会有离群点，而且不需要事先指定聚类数量。

## Why the Statistical Signal Layer Matters

English answer:

Clustering tells us what people are talking about, but it does not tell us which themes are frequent, uncertain, or associated with outcomes. The statistical layer helps prioritize themes by showing prevalence, confidence intervals, rating gaps, effect sizes, and adjusted p-values. I frame these as exploratory signals because themes are discovered before they are tested.

中文翻译：

聚类告诉我们客户在谈什么，但不会告诉我们哪些主题更常见、更不确定，或与结果变量有关。统计层通过主题占比、置信区间、评分差异、效果量和校正后的 p 值来帮助排序主题。我把这些结果称为探索性信号，因为主题是先从同一批数据中发现，再进行检验。

## What Effect Size Means

English answer:

Effect size measures the practical strength of a difference or association, not just whether a p-value is small. In this project, rank-biserial effect size helps describe how strongly ratings inside a theme differ from ratings outside the theme. That matters because a statistically detectable difference can still be too small to matter in practice.

中文翻译：

效果量衡量差异或关联在实际上的强度，而不只是 p 值是否很小。在这个项目中，rank-biserial 效果量用来描述主题内评分和主题外评分的差异强度。这很重要，因为统计上可检测到的差异不一定有实际业务意义。

## Why FDR Correction Is Used

English answer:

The project tests multiple themes, so some small p-values can appear by chance. Benjamini-Hochberg FDR correction reduces the risk of overreacting to false discoveries while still being practical for exploratory analysis. It is a guardrail, not a guarantee that a signal is causal or final.

中文翻译：

这个项目会同时检验多个主题，所以有些小 p 值可能只是偶然出现。Benjamini-Hochberg FDR 校正可以降低把假阳性当成发现的风险，同时仍然适合探索性分析。它是一个防护措施，不代表信号就是因果或最终结论。

## How Hallucination Risk Is Controlled

English answer:

The LLM layer is constrained to structured evidence: theme IDs, keywords, representative snippets, prevalence, rating gaps, adjusted p-values, interpretation labels, and caution flags. The default provider is a deterministic mock, so tests and local runs do not require paid APIs. The LLM is used for labeling and summarization only, not discovery or statistical judgment.

中文翻译：

LLM 层只接收结构化证据，包括主题 ID、关键词、代表性片段、主题占比、评分差异、校正后的 p 值、解释标签和 caution flags。默认提供者是确定性的模拟 provider，所以测试和本地运行不需要付费 API。LLM 只用于命名和总结，不用于发现主题或替代统计判断。

## What Limitations Remain

English answer:

This is a portfolio project, not production SaaS. The signals are exploratory because discovery and testing happen on the same data. Real deployment would need stronger data governance, privacy review, model evaluation, monitoring, holdout validation, stakeholder review, and a secret-safe live API integration if external LLMs are used.

中文翻译：

这是一个作品集项目，不是生产级 SaaS。因为主题发现和统计检验来自同一批数据，所以信号是探索性的。真实部署需要更完整的数据治理、隐私审查、模型评估、监控、留出集验证、业务方审核，以及在使用外部 LLM 时安全处理密钥和 API 集成。

## Likely Interview Questions

### 1. What problem does this project solve?

English answer:

It helps turn unstructured customer reviews into structured themes that can be measured, explained, and discussed with evidence. The project is designed to show the full analytics path from raw feedback to cautious insight reporting.

中文翻译：

它帮助把非结构化客户评论转成可以衡量、解释和用证据讨论的结构化主题。这个项目展示的是从原始反馈到谨慎洞察报告的完整分析流程。

### 2. Why not just ask an LLM to summarize all reviews?

English answer:

A direct LLM summary is hard to audit and can blur evidence, frequency, and uncertainty. This project discovers themes with embeddings and clustering, measures signals with statistics, and only then uses an LLM-style layer to summarize supplied evidence.

中文翻译：

直接让 LLM 总结所有评论很难审计，也容易混淆证据、频率和不确定性。这个项目先用嵌入和聚类发现主题，再用统计方法衡量信号，最后才用 LLM 风格的层总结已经提供的证据。

### 3. What makes this more than a dashboard?

English answer:

It includes data validation, semantic representation, unsupervised discovery, explainability outputs, exploratory statistical testing, evidence-bounded labeling, and data safety rules. The value is in the reproducible analytical pipeline, not just the final display.

中文翻译：

它包含数据验证、语义表示、无监督发现、可解释性输出、探索性统计检验、有证据约束的命名，以及数据安全规则。项目价值在可复现的分析流程，而不仅是最终展示界面。

### 4. How do you know the clusters are meaningful?

English answer:

The project checks clusters through representative reviews, TF-IDF keywords, cluster diagnostics, and downstream signal tables. That does not prove every cluster is perfect, but it makes cluster quality reviewable instead of opaque.

中文翻译：

项目通过代表性评论、TF-IDF 关键词、聚类诊断和后续信号表来检查主题。这不能证明每个聚类都完美，但可以让聚类质量变得可审查，而不是黑箱。

### 5. What is the role of ratings?

English answer:

Ratings are used as an outcome for exploratory association analysis after themes are discovered. The project compares ratings inside and outside each theme, but it does not claim that a theme causes rating changes.

中文翻译：

评分在主题发现之后用于探索性关联分析。项目会比较主题内和主题外的评分，但不会声称某个主题导致了评分变化。

### 6. Why use a mock LLM provider?

English answer:

The mock provider keeps the project reproducible, testable, and secret-safe. It demonstrates the integration boundary and prompt structure without requiring paid APIs, keys, or network calls.

中文翻译：

模拟 provider 可以让项目保持可复现、可测试，并且避免密钥风险。它展示了 LLM 集成边界和提示结构，但不需要付费 API、密钥或网络调用。

### 7. What would you improve next?

English answer:

I would add holdout validation, stronger cluster evaluation, richer synthetic demos, optional secret-safe live provider integration, and a lightweight review UI for human validation. I would still keep raw and processed real data out of git.

中文翻译：

下一步我会加入留出集验证、更强的聚类评估、更完整的合成示例、安全的可选真实 provider 集成，以及一个轻量的人审界面。我仍然会把原始和处理后的真实数据排除在 git 之外。

### 8. How would this change for production?

English answer:

Production would require data contracts, access control, privacy review, observability, model and cluster evaluation, monitoring for drift, human review workflows, deployment infrastructure, and formal stakeholder sign-off.

中文翻译：

生产环境需要数据契约、访问控制、隐私审查、可观测性、模型和聚类评估、漂移监控、人工审核流程、部署基础设施，以及正式的业务方确认。

### 9. What is the biggest statistical caveat?

English answer:

The main caveat is that themes are discovered and tested on the same dataset, so the p-values are exploratory prioritization signals. A stronger design would discover themes on one split and validate associations on a holdout split.

中文翻译：

最大的统计限制是主题发现和检验来自同一批数据，所以 p 值只是探索性排序信号。更强的设计会在一个数据切分上发现主题，再在留出数据上验证关联。

### 10. How do you prevent claims from going beyond the evidence?

English answer:

The docs, data policy, prompts, examples, and report language all state that outputs are exploratory and evidence-backed. The project avoids causal language, avoids real company conclusions, and keeps generated real-data artifacts out of git.

中文翻译：

项目的文档、数据政策、提示词、示例和报告措辞都明确说明输出是探索性的、需要证据支撑。项目避免因果措辞，避免对真实公司下结论，并且不把真实数据生成物提交到 git。
