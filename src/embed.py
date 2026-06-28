"""
embed.py — Phase 1: Sentence Embeddings and Similarity Sanity Check

WHY EMBEDDINGS FOR SEMANTIC CLUSTERING
---------------------------------------
A sentence embedding maps a piece of text to a dense numeric vector (here: 384 numbers)
such that texts with similar *meaning* land close together in that vector space, measured by
cosine similarity. This is the key property that makes semantic clustering possible:

  "the food was cold" and "my meal arrived chilly"

share zero words, so bag-of-words / TF-IDF methods see them as unrelated. An embedding model
trained on millions of sentence pairs has learned that both phrases describe the same experience,
so they end up close in vector space. Cosine similarity between two vectors ranges from -1
(opposite meaning) to +1 (identical meaning); in practice, same-topic review pairs typically
score 0.7–0.95.

This property means that after we cluster the embeddings in Phase 2, we can expect each cluster
to contain reviews about one coherent theme rather than a bag of random words.

MODEL CHOICE: all-MiniLM-L6-v2
---------------------------------------
Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional output vectors
  - Max input: 256 word-piece tokens (longer text is truncated; see TRUNCATION note below)
  - ~23 M parameters — fast on CPU, no GPU required
  - Pre-trained on 1B+ sentence pairs; strong off-the-shelf semantic quality

Tradeoff: all-mpnet-base-v2 (768-dim, ~110 M params) gives marginally better semantic
separation on MTEB benchmarks (~68 vs ~63 average score) but takes 3–4x longer on CPU.
For an 8,000-row demo that runs on a laptop, MiniLM is the right speed/quality tradeoff.
A production system processing millions of documents would re-evaluate this choice against
domain-specific fine-tunes (e.g. a review-trained model from the MTEB leaderboard).

TRUNCATION NOTE
---------------------------------------
MiniLM silently truncates any input longer than 256 word-piece tokens. Our data has reviews
up to 1,212 words. We log the count of truncated reviews and proceed with simple truncation.

Deliberate tradeoff: for review-level sentiment analysis, the most salient information is
front-loaded (the reviewer's verdict, the main complaint or praise). Truncating tail content
loses nuance but rarely changes the core theme. The proper production fix is:
  1. Split long reviews into overlapping chunks → embed each chunk → mean-pool the vectors.
  2. Or use a long-context model (e.g. longformer-based sentence encoder).
Both are out of scope for this demo; this comment exists so the decision can be defended.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # see module docstring for rationale
BATCH_SIZE = 64                    # tuned for CPU; GPU users can raise to 256+
RANDOM_SEED = 42

# MiniLM's stated max is 256 word-piece tokens. Word pieces are shorter than
# words (English averages ~1.3 word-pieces/word), so the practical word limit
# is roughly 196 words. We use 150 words as a conservative threshold for
# flagging reviews that are likely to be truncated.
TRUNCATION_WORD_THRESHOLD = 150

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "reviews_cleaned.parquet"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
IDS_PATH = PROJECT_ROOT / "data" / "processed" / "embedding_ids.npy"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix_row(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity between query_vec and every row in matrix.

    Both inputs are assumed to be L2-normalised already (which sentence-transformers
    returns by default when normalize_embeddings=True), so cosine similarity reduces
    to a dot product — fast and numerically stable.
    """
    return matrix @ query_vec


def top_k_similar(
    query_idx: int,
    embeddings: np.ndarray,
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return indices and scores of the k most similar rows, excluding the query itself."""
    sims = cosine_similarity_matrix_row(embeddings[query_idx], embeddings)
    sims[query_idx] = -1.0          # exclude self
    top_indices = np.argpartition(sims, -k)[-k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
    return [(int(i), float(sims[i])) for i in top_indices]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_reviews() -> pd.DataFrame:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PARQUET}\n"
            "Run src/data_prep.py first to generate the cleaned parquet."
        )
    df = pd.read_parquet(INPUT_PARQUET)
    log.info("Loaded %d reviews from %s", len(df), INPUT_PARQUET)
    return df


def report_truncation(df: pd.DataFrame) -> None:
    """Log how many reviews are likely to be truncated by the model's token limit."""
    word_counts = df["raw_text"].str.split().str.len()
    n_long = int((word_counts > TRUNCATION_WORD_THRESHOLD).sum())
    pct = 100.0 * n_long / len(df)
    log.info(
        "Truncation report: %d / %d reviews (%.1f%%) exceed ~%d words "
        "and will be truncated to the model's 256-token limit.",
        n_long, len(df), pct, TRUNCATION_WORD_THRESHOLD,
    )
    log.info(
        "  Deliberate choice: core sentiment is front-loaded in most reviews. "
        "Chunking / mean-pooling is the production fix; out of scope here."
    )


def compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Encode all reviews and return an (N, 384) float32 array."""
    log.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = df["raw_text"].tolist()
    log.info("Encoding %d reviews (batch_size=%d) ...", len(texts), BATCH_SIZE)
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise so cosine sim = dot product
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    log.info(
        "Encoding complete: shape=%s  dtype=%s  time=%.1fs (%.1f reviews/s)",
        embeddings.shape, embeddings.dtype, elapsed, len(texts) / elapsed,
    )
    return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, ids: np.ndarray) -> None:
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(IDS_PATH, ids)
    size_mb = EMBEDDINGS_PATH.stat().st_size / 1024 / 1024
    log.info("Saved embeddings to %s (%.1f MB)", EMBEDDINGS_PATH, size_mb)
    log.info("Saved id alignment array to %s", IDS_PATH)


def load_embeddings_from_disk() -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True)
    log.info(
        "Loaded embeddings from disk: shape=%s  (skipping recompute)",
        embeddings.shape,
    )
    return embeddings, ids


def sanity_check(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    """
    For 3 hand-picked queries (including at least one negative review), print the
    top-3 nearest neighbors. Goal: visually confirm that similar texts cluster
    together — this validates the embedding quality before we hand the vectors
    to the clustering phase.
    """
    log.info("")
    log.info("== Nearest-neighbor sanity check ==================================")
    log.info("   Confirms that semantically similar reviews land close in vector space.")
    log.info("   Each block: query review, then its top-3 most similar neighbors.")
    log.info("===================================================================")

    # Pick 3 query reviews manually for interpretability.
    # We want variety: one clear negative complaint, one positive praise, one mixed.
    # Strategy: search by star rating + a keyword in the text.
    def find_row(rating: float, keyword: str) -> int | None:
        mask = (df["rating"] == rating) & df["raw_text"].str.contains(keyword, case=False, na=False)
        hits = df[mask]
        if hits.empty:
            return None
        # Pick deterministically (first match after sorting by index)
        return int(hits.index[0])

    queries: list[tuple[str, int | None]] = [
        ("1-star complaint (leaking / broken)",    find_row(1.0, "leak")),
        ("5-star praise (easy to clean)",          find_row(5.0, "easy to clean")),
        ("3-star mixed (good but too small)",      find_row(3.0, "too small")),
    ]

    # Fallback: if a keyword search misses, pick the first row of that rating
    def fallback(rating: float) -> int:
        hits = df[df["rating"] == rating]
        return int(hits.index[0]) if not hits.empty else 0

    fallbacks = {1.0: None, 5.0: None, 3.0: None}

    for label, idx in queries:
        if idx is None:
            # keyword not found — grab any row with the target rating
            rating_target = float(label.split("-")[0].replace("star", "").strip()
                                  if "-star" in label else "5")
            idx = fallback(rating_target)

        query_row = df.iloc[idx]
        query_preview = query_row["raw_text"][:200].replace("\n", " ")

        log.info("")
        log.info("QUERY [%s]", label)
        log.info("  Rating : %g star(s)", query_row["rating"])
        log.info("  Text   : %s ...", query_preview)
        log.info("  ---")

        neighbors = top_k_similar(idx, embeddings, k=3)
        for rank, (n_idx, score) in enumerate(neighbors, start=1):
            n_row = df.iloc[n_idx]
            n_preview = n_row["raw_text"][:160].replace("\n", " ")
            log.info(
                "  #%d  sim=%.4f  [%g star]  %s ...",
                rank, score, n_row["rating"], n_preview,
            )

    log.info("")
    log.info("===================================================================")
    log.info("== End sanity check ================================================")
    log.info("")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Phase 1: Embedding — model: %s ===", MODEL_NAME)

    df = load_reviews()
    report_truncation(df)

    if EMBEDDINGS_PATH.exists() and IDS_PATH.exists():
        log.info("Embeddings already on disk — loading (delete to force recompute).")
        embeddings, _ = load_embeddings_from_disk()
    else:
        embeddings = compute_embeddings(df)
        ids = df["id"].to_numpy(dtype=object)
        save_embeddings(embeddings, ids)

    log.info("Embedding matrix shape: %s", embeddings.shape)

    sanity_check(df, embeddings)

    log.info("Done. Embeddings ready for Phase 2 (clustering).")


if __name__ == "__main__":
    main()
