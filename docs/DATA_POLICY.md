# Data Policy

This repository must keep data handling conservative and explicit.

## What Must Not Be Committed

- Raw external datasets.
- Real processed datasets generated from external data.
- API keys, tokens, credentials, `.env` files, or other secrets.
- Large model artifacts or embedding outputs.
- Generated outputs from real data unless explicitly approved.

## What May Be Committed

- Toy samples.
- Schema examples.
- Small synthetic fixtures for tests.
- Derived summary outputs only when license-safe, small, and explicitly approved.

## Processed Data

Real processed datasets should not be committed unless they are small, license-safe, and explicitly approved. The default is to regenerate processed data locally from user-provided raw data.

## Environment Files

Do not add `.env` in Phase 0. A `.env.example` may be added later only if it is useful and contains no real secrets.

## Dataset Acquisition

Dataset download steps will be documented in a later phase. Phase 0 does not download external datasets.

## Review Before Commit

Before committing, check git status and staged files for data or secrets. If there is any doubt, do not commit the file.

