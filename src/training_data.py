"""
Phase 5 - Supervised training dataset construction for Amazon ML Challenge 2026:
Business Entity Resolution.

Turns ground-truth positive matches + Phase 3's blocking-stage candidate pairs into a
balanced, leakage-free, feature-ready dataset for LightGBM: positive pairs from ground
truth, hard negatives sampled from candidate pairs that passed blocking but aren't true
matches, split by source1 entity (never by individual pair) into train_pairs.parquet /
validation_pairs.parquet.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_tsv
from src.features import generate_features
from src.preprocessing import add_clean_columns

PAIR_ID_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source"]


# --------------------------------------------------------------------------- #
# Positive pairs
# --------------------------------------------------------------------------- #

def build_positive_pairs(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Explode ground_truth's comma-separated matched_entity_ids into one row per true
    (source1_entity_id, candidate_entity_id, source, label=1) pair."""
    exploded = ground_truth.assign(
        candidate_entity_id=ground_truth["matched_entity_ids"].str.split(",")
    ).explode("candidate_entity_id")
    exploded = exploded[["source1_entity_id", "candidate_entity_id"]].dropna()
    exploded = exploded[exploded["candidate_entity_id"].str.len() > 0]
    exploded["source"] = np.where(exploded["candidate_entity_id"].str.startswith("S2"), "S2", "S3")
    exploded["label"] = 1
    return (
        exploded.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Hard negatives
# --------------------------------------------------------------------------- #

def sample_hard_negatives(
    candidate_pairs: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    negatives_per_positive: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Hard negatives = candidate pairs that passed blocking (share a block: similar
    prefix, overlapping tokens, ...) but are NOT in the ground truth. These are far
    more informative than random pairs, since blocking would never have proposed a
    truly dissimilar pair in the first place - the model needs exactly this "looks
    plausible but isn't a match" signal to learn a useful decision boundary.

    Sampled per source1 entity, capped at `negatives_per_positive` times that entity's
    positive count (minimum 1), so one entity with a huge block can't flood the
    negative pool and unbalance the dataset. Fully vectorized: shuffle once, then keep
    each group's first `quota` rows via groupby().cumcount() - no per-group Python loop.
    """
    true_pairs = set(zip(positive_pairs["source1_entity_id"], positive_pairs["candidate_entity_id"]))
    pair_keys = zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"])
    is_positive = np.fromiter((k in true_pairs for k in pair_keys), dtype=bool, count=len(candidate_pairs))
    negative_pool = candidate_pairs.loc[~is_positive].sample(frac=1.0, random_state=random_state)
    negative_pool = negative_pool.reset_index(drop=True)

    if negative_pool.empty:
        return negative_pool.reindex(columns=PAIR_ID_COLUMNS + ["label"])

    rank_within_entity = negative_pool.groupby("source1_entity_id").cumcount()

    n_positives_per_entity = positive_pairs.groupby("source1_entity_id").size()
    quota = (
        negative_pool["source1_entity_id"].map(n_positives_per_entity).fillna(1).astype(int)
        * negatives_per_positive
    ).clip(lower=negatives_per_positive)

    negatives = negative_pool.loc[rank_within_entity.to_numpy() < quota.to_numpy(), PAIR_ID_COLUMNS].copy()
    negatives["label"] = 0
    return negatives.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Combine + split
# --------------------------------------------------------------------------- #

def build_labeled_pairs(
    ground_truth: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    negatives_per_positive: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Union of positive pairs and sampled hard negatives, shuffled, with a `label` column."""
    positives = build_positive_pairs(ground_truth)
    negatives = sample_hard_negatives(candidate_pairs, positives, negatives_per_positive, random_state)
    labeled = pd.concat(
        [positives[PAIR_ID_COLUMNS + ["label"]], negatives[PAIR_ID_COLUMNS + ["label"]]],
        ignore_index=True,
    )
    return labeled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def split_train_validation(
    labeled_pairs: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by source1_entity_id, not by individual pair, so every pair for a given
    source1 entity - its positive match(es) and its hard negatives alike - lands
    entirely in train or entirely in validation.

    Splitting at the pair level would leak: the model could see entity X's true match
    in train and a different, very similar hard negative for that SAME entity X in
    validation, letting validation score reflect memorized entity-specific quirks
    instead of genuine generalization to unseen entities.
    """
    entities = labeled_pairs["source1_entity_id"].unique()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(entities)
    n_val = int(round(len(shuffled) * val_size))
    val_entities = set(shuffled[:n_val])

    is_val = labeled_pairs["source1_entity_id"].isin(val_entities)
    train_pairs = labeled_pairs.loc[~is_val].reset_index(drop=True)
    val_pairs = labeled_pairs.loc[is_val].reset_index(drop=True)
    return train_pairs, val_pairs


def build_training_dataset(
    ground_truth: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    negatives_per_positive: int = 1,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End to end: balanced positive/hard-negative pairs -> similarity features (Phase 4)
    -> source1-entity-level train/validation split. The two DataFrames returned are
    exactly what gets written to train_pairs.parquet / validation_pairs.parquet.
    """
    labeled_pairs = build_labeled_pairs(ground_truth, candidate_pairs, negatives_per_positive, random_state)

    features = generate_features(labeled_pairs, source1, targets, verbose=False)
    # generate_features preserves labeled_pairs' row order exactly, so a positional
    # assignment is safe here and avoids a join that could silently duplicate rows.
    features["label"] = labeled_pairs["label"].to_numpy()

    return split_train_validation(features, val_size=val_size, random_state=random_state)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 5 supervised training dataset.")
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv")
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    t0 = time.time()
    source1 = add_clean_columns(load_tsv(os.path.join(args.data_dir, "train_source1.tsv")))
    s2 = add_clean_columns(load_tsv(os.path.join(args.data_dir, "train_source2.tsv")))
    s3 = add_clean_columns(load_tsv(os.path.join(args.data_dir, "train_source3.tsv")))
    ground_truth = load_tsv(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    candidate_pairs = pd.read_csv(args.candidate_pairs, sep="\t")
    print(f"Loaded inputs in {time.time() - t0:.1f}s ({len(candidate_pairs):,} candidate pairs)")

    train_pairs, val_pairs = build_training_dataset(
        ground_truth,
        candidate_pairs,
        source1,
        {"S2": s2, "S3": s3},
        negatives_per_positive=args.negatives_per_positive,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    print(f"train_pairs: {len(train_pairs):,} rows, positive rate={train_pairs['label'].mean():.3f}")
    print(f"validation_pairs: {len(val_pairs):,} rows, positive rate={val_pairs['label'].mean():.3f}")

    overlap = set(train_pairs["source1_entity_id"]) & set(val_pairs["source1_entity_id"])
    print(f"source1 entity overlap between train/val: {len(overlap)} (must be 0)")
    assert not overlap, "Data leakage: some source1 entities appear in both train and validation."

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_pairs.parquet")
    val_path = os.path.join(args.output_dir, "validation_pairs.parquet")
    train_pairs.to_parquet(train_path, index=False)
    val_pairs.to_parquet(val_path, index=False)
    print(f"Wrote {train_path} and {val_path} in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
