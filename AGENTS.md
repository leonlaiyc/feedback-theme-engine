# Repository Instructions for Codex

This repository is a portfolio-grade analytics engineering project for turning
unstructured customer feedback into structured, evidence-backed themes and
business insights.

## Working Principles

- Treat this as a professional portfolio project, not a toy demo.
- Prefer simple, readable, testable Python over clever abstractions.
- Keep functions small and behavior explicit.
- Use type hints where they improve clarity.
- Keep business framing honest, traceable, and evidence-backed.
- Do not claim production readiness unless the repository actually supports it.
- Do not claim findings represent Amazon, any company, or any real business conclusion.
- Keep abstract sections concise and strong.
- Keep detailed sections complete and clear.
- Avoid the sentence pattern "The hard part is not X. The hard part is Y."

## Data and Secrets

- Do not commit raw external datasets.
- Do not commit processed data generated from real datasets.
- Do not commit API keys, tokens, credentials, `.env`, or other secrets.
- Keep only toy samples, schema examples, and approved derived summaries in git.
- Ask before adding dataset download logic, API integrations, or major new dependencies.

## Code and Validation

- Run tests when code changes.
- Run lightweight validation before reporting completion.
- Update documentation when architecture, methodology, or scope changes.
- Keep generated artifacts out of git unless explicitly approved.

## Phase Completion Checklist

Every phase should end with:

- Clear summary of work completed.
- Test and validation results.
- Git status.
- Commit hash.
- Any skipped steps or blockers.
