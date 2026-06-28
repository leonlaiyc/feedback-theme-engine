"""
data_prep.py — Phase 0: Data Acquisition, Cleaning, and Sanity Checks

Source  : McAuley-Lab/Amazon-Reviews-2023 (HuggingFace datasets, streaming mode)
Category: Home_and_Kitchen
  Rationale: mid-size category with broad complaint diversity — appliances,
  cookware, storage, cleaning tools — giving future cluster phases varied themes
  to work with. More specific categories (e.g. "Headphones") would produce
  narrower, less interesting clusters for a general-engine demo.

Text field: title + body (concatenated with a space separator)
  Rationale: the title often contains the reviewer's core verdict or complaint
  in a few words; the body expands on it. Discarding the title would lose
  high-signal signal that summarises the whole review. The tradeoff is that
  some titles repeat the body's first sentence, adding slight redundancy —
  acceptable at this stage.

Design decisions that affect downstream phases:
  - MIN_WORDS = 5: filters "Great product" style reviews that carry no theme
    information. Lower threshold → more rows but noisier clusters.
    Tradeoff: reduces raw recall; increases cluster purity.
  - STREAM_LIMIT = 50_000: we stream only the first 50 k rows from the HF
    dataset to avoid downloading the full category (multi-GB). The first 50 k
    rows are NOT a random sample of the category — they reflect HF's internal
    ordering. This is acceptable for a demo; a production version would sample
    across shards.
  - FINAL_TARGET = 8_000: large enough for statistically stable clusters,
    small enough for free-tier LLM labeling in later phases.
  - RANDOM_SEED = 42: conventional default; ensures reproducibility.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from langdetect import DetectorFactory, detect, LangDetectException
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY = "raw_review_Home_and_Kitchen"
STREAM_LIMIT = 50_000   # rows to pull via streaming (avoids full download)
MIN_WORDS = 5           # design decision — see module docstring
FINAL_TARGET = 8_000    # target rows after downsampling
RANDOM_SEED = 42

# Fix langdetect's internal random state so language detection is reproducible
DetectorFactory.seed = RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_text(row: dict) -> str:
    """Concatenate review title and body into a single text field."""
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def is_english(text: str) -> bool:
    """Return True if langdetect identifies the text as English.

    langdetect can mis-classify very short strings, but our MIN_WORDS filter
    runs before this, so inputs here are at least 5 words.
    """
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def word_count(text: str) -> int:
    return len(text.split())


def log_step(label: str, count: int, prev: int) -> None:
    pct = 100.0 * count / prev if prev else 0.0
    log.info("  %-35s %7d rows  (%.1f %% of previous)", label, count, pct)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def stream_raw(limit: int) -> pd.DataFrame:
    """Pull `limit` rows from the HF dataset in streaming mode and return a DataFrame."""
    log.info("Streaming first %d rows from %s / %s ...", limit, DATASET_NAME, CATEGORY)
    t0 = time.time()

    dataset = load_dataset(
        DATASET_NAME,
        name=CATEGORY,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    records: list[dict] = []
    for row in tqdm(dataset, total=limit, desc="streaming", unit="row"):
        records.append({
            "id": row.get("parent_asin", "") + "_" + row.get("user_id", ""),
            "rating": row.get("rating"),
            "raw_text": build_text(row),
        })
        if len(records) >= limit:
            break

    elapsed = time.time() - t0
    log.info("Streamed %d rows in %.1f s", len(records), elapsed)
    return pd.DataFrame(records)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps with row-count logging at every step."""
    start = len(df)
    log.info("─── Cleaning log ─────────────────────────────────────")
    log.info("  %-35s %7d rows  (starting point)", "raw streamed", start)

    # 1. Drop rows with null or empty text
    df = df[df["raw_text"].notna() & (df["raw_text"].str.strip() != "")]
    log_step("after drop null/empty text", len(df), start)

    # 2. Drop exact-duplicate text (same review text submitted multiple times)
    #    Tradeoff: keeps the first occurrence; arbitrary but deterministic.
    df = df.drop_duplicates(subset="raw_text", keep="first")
    log_step("after drop exact duplicates", len(df), start)

    # 3. Drop reviews shorter than MIN_WORDS
    #    Design decision — see module docstring.
    df = df[df["raw_text"].apply(word_count) >= MIN_WORDS]
    log_step(f"after drop < {MIN_WORDS} words", len(df), start)

    # 4. Keep English-only reviews
    #    langdetect adds ~1–3 ms per row; acceptable for 50 k rows.
    #    A faster alternative is fasttext-langdetect, but it requires a model
    #    download — langdetect is pip-installable and sufficient here.
    log.info("  Running language detection (may take a minute) ...")
    df = df[df["raw_text"].apply(is_english)]
    log_step("after keep English only", len(df), start)

    # 5. Downsample to FINAL_TARGET with a fixed seed
    if len(df) > FINAL_TARGET:
        df = df.sample(n=FINAL_TARGET, random_state=RANDOM_SEED)
        log_step(f"after downsample to {FINAL_TARGET}", len(df), start)
    else:
        log.info("  Sample smaller than target — no downsampling applied (%d rows)", len(df))

    log.info("─── End cleaning log ─────────────────────────────────")
    return df.reset_index(drop=True)


def sanity_checks(df: pd.DataFrame) -> None:
    """Print rating distribution, text-length stats, and 5 sample rows."""
    log.info("")
    log.info("══ Sanity checks ═══════════════════════════════════════")

    # Rating distribution
    rating_counts = df["rating"].value_counts().sort_index()
    total = len(df)
    log.info("  Rating distribution:")
    for stars, count in rating_counts.items():
        bar = "█" * int(count / total * 40)
        log.info("    %s★  %5d  (%4.1f %%)  %s", int(stars), count, 100 * count / total, bar)

    top_heavy = rating_counts.get(5, 0) / total > 0.50
    if top_heavy:
        log.info(
            "  [SKEW WARNING] >50%% of reviews are 5-star. "
            "Complaint-theme discovery will be driven by the minority low-star reviews. "
            "Consider oversampling 1-2 star reviews in a later phase."
        )

    # Text-length distribution
    lengths = df["raw_text"].apply(word_count)
    log.info("  Text length (words):")
    log.info("    min=%d  p25=%d  median=%d  p75=%d  max=%d",
             lengths.min(), int(lengths.quantile(0.25)),
             int(lengths.median()), int(lengths.quantile(0.75)), lengths.max())

    # 5 sample rows
    log.info("  Sample rows (5 random):")
    sample = df[["id", "rating", "raw_text"]].sample(5, random_state=RANDOM_SEED)
    for _, row in sample.iterrows():
        preview = row["raw_text"][:120].replace("\n", " ")
        log.info("    [%s★] %s ...", int(row["rating"]), preview)

    log.info("══ End sanity checks ═══════════════════════════════════")
    log.info("")


def save(df: pd.DataFrame) -> Path:
    out_path = OUTPUT_DIR / "reviews_cleaned.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(df), out_path)
    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Phase 0: Data Prep — %s ===", CATEGORY)

    raw_df = stream_raw(STREAM_LIMIT)
    cleaned_df = clean(raw_df)
    sanity_checks(cleaned_df)
    save(cleaned_df)

    log.info("Done. Output: data/processed/reviews_cleaned.parquet  (%d rows)", len(cleaned_df))


if __name__ == "__main__":
    main()
