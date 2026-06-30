---
name: data-safety-and-secrets
description: Use whenever touching data paths, dataset handling, API keys, environment variables, or gitignore rules.
---

# Data Safety and Secrets

- Never commit raw external datasets.
- Never commit real processed datasets unless small, license-safe, and explicitly approved.
- Never commit API keys, tokens, credentials, `.env`, or secrets.
- Keep data directories guarded by `.gitignore`.
- Commit only toy samples, schema examples, synthetic fixtures, or approved derived summaries.
- Document dataset download steps instead of embedding raw data in the repo.
- Check `git status` and staged files before committing.

