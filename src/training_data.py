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

DEFAULT_NEGATIVES_PER_POSITIVE = 6
"""A 1:1 pos:neg training ratio (the old default) teaches a model to over-predict
matches: it never sees how heavily negatives outnumber positives in the real
candidate pool, so its decision boundary ends up too permissive and precision
collapses at inference time on the true, heavily-imbalanced distribution (this is
exactly what happened in this project's own Phase 8 smoke test: ~99.7% validation
precision at 1:1 sampling, ~69% precision once scored against the real candidate
pool). 5:1-8:1 is a more realistic ratio without making the positive class so rare
that training becomes unstable at this dataset's scale."""


def sample_hard_negatives(
    candidate_pairs: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
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
    # Vectorized membership check via integer-packed keys, instead of a Python-level
    # `k in set(...)` loop over every candidate pair (which took over an hour and
    # exhausted 64GB building hundreds of millions of tuple objects) or a naive string
    # concatenation + .isin() (which still exhausted 64GB, since building the combined
    # "id1|id2" string column materializes 400M+ new Python string objects on top of
    # the already-loaded data). factorize() maps each entity_id string to a compact
    # int32 code once; packing (code1, code2) into a single uint64 - the same technique
    # src/blocking.py already uses for its own full-scale dedup - lets .isin() run
    # over plain integer arrays with no large string allocations.
    s1_codes, _ = pd.factorize(
        pd.concat([positive_pairs["source1_entity_id"], candidate_pairs["source1_entity_id"]], ignore_index=True)
    )
    cand_codes, _ = pd.factorize(
        pd.concat([positive_pairs["candidate_entity_id"], candidate_pairs["candidate_entity_id"]], ignore_index=True)
    )
    n_pos = len(positive_pairs)

    def pack(s1: np.ndarray, cand: np.ndarray) -> np.ndarray:
        return (s1.astype(np.uint64) << np.uint64(32)) | cand.astype(np.uint64)

    true_pair_keys = pack(s1_codes[:n_pos], cand_codes[:n_pos])
    candidate_pair_keys = pack(s1_codes[n_pos:], cand_codes[n_pos:])
    is_positive = np.isin(candidate_pair_keys, true_pair_keys)
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
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
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


def _entity_strata(labeled_pairs: pd.DataFrame, source1: pd.DataFrame, entities: np.ndarray) -> pd.Series:
    """Per-entity stratification key: match-count bucket (singleton / single_match /
    multi_match) x country. `labeled_pairs` already contains every true positive (see
    build_positive_pairs), so counting label==1 rows per entity recovers the real
    match count even though negatives are subsampled."""
    positive_counts = (
        labeled_pairs.loc[labeled_pairs["label"] == 1]
        .groupby("source1_entity_id").size()
    )
    bucket = pd.Series(positive_counts, index=entities).reindex(entities).fillna(0).astype(int).clip(upper=2)
    bucket_labels = bucket.map({0: "singleton", 1: "single_match", 2: "multi_match"})

    country_map = source1.drop_duplicates("entity_id").set_index("entity_id")["country"]
    country = pd.Series(entities, index=entities).map(country_map).fillna("UNKNOWN")

    return bucket_labels.astype(str) + "|" + country.astype(str)


def stratified_split_train_validation(
    labeled_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entity-level split (same leakage guarantee as split_train_validation) additionally
    stratified by (match-count bucket, country), so validation's mix of singletons /
    single-matches / multi-matches and its country distribution mirror the real
    Source1 population instead of being whatever a plain random draw happens to give.
    This matters because with only 5 submissions/day, local CV needs to actually be
    trustworthy - an unlucky random split can look far better or worse than reality.

    Falls back to the plain (unstratified) split with a printed warning if any
    stratum ends up too small for sklearn's stratified split to run (tiny datasets).
    """
    from sklearn.model_selection import train_test_split

    # .unique() on pandas 3.0's Arrow-backed string dtype returns an ExtensionArray
    # that sklearn's fancy-indexing chokes on ("only integer scalar arrays can be
    # converted to a scalar index") - force a plain numpy object array up front.
    entities = labeled_pairs["source1_entity_id"].unique().astype(object)
    strata = _entity_strata(labeled_pairs, source1, entities)

    # sklearn's stratify requires every group to have >= 2 members; collapse rarer
    # strata into one bucket rather than fail outright on a small/uneven dataset.
    counts = strata.value_counts()
    rare = counts[counts < 2].index
    strata_effective = strata.where(~strata.isin(rare), "OTHER")
    if strata_effective.value_counts().min() < 2:
        print("stratified_split_train_validation: too few entities per stratum, "
              "falling back to an unstratified split.")
        return split_train_validation(labeled_pairs, val_size=val_size, random_state=random_state)

    train_entities, val_entities = train_test_split(
        entities, test_size=val_size, random_state=random_state, stratify=strata_effective.to_numpy(),
    )
    val_entities = set(val_entities)
    is_val = labeled_pairs["source1_entity_id"].isin(val_entities)
    train_pairs = labeled_pairs.loc[~is_val].reset_index(drop=True)
    val_pairs = labeled_pairs.loc[is_val].reset_index(drop=True)
    return train_pairs, val_pairs


def build_training_dataset(
    ground_truth: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End to end: balanced positive/hard-negative pairs -> similarity features (Phase 4)
    -> source1-entity-level train/validation split (stratified by match-count bucket x
    country when `stratify=True`). The two DataFrames returned are exactly what gets
    written to train_pairs.parquet / validation_pairs.parquet.
    """
    labeled_pairs = build_labeled_pairs(ground_truth, candidate_pairs, negatives_per_positive, random_state)

    features = generate_features(labeled_pairs, source1, targets, verbose=False)
    # generate_features preserves labeled_pairs' row order exactly, so a positional
    # assignment is safe here and avoids a join that could silently duplicate rows.
    features["label"] = labeled_pairs["label"].to_numpy()

    if stratify:
        return stratified_split_train_validation(features, source1, val_size=val_size, random_state=random_state)
    return split_train_validation(features, val_size=val_size, random_state=random_state)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 5 supervised training dataset.")
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv")
    parser.add_argument("--negatives-per-positive", type=int, default=DEFAULT_NEGATIVES_PER_POSITIVE)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-stratify", action="store_true",
                         help="Use a plain random entity split instead of stratifying by match-count bucket x country.")
    args = parser.parse_args()

    t0 = time.time()
    source1 = add_clean_columns(load_tsv(os.path.join(args.data_dir, "train_source1.tsv")))
    s2 = add_clean_columns(load_tsv(os.path.join(args.data_dir, "train_source2.tsv")))
    s3 = add_clean_columns(load_tsv(os.path.join(args.data_dir, "train_source3.tsv")))
    ground_truth = load_tsv(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    candidate_pairs = pd.read_csv(args.candidate_pairs, sep="\t", dtype=str)
    print(f"Loaded inputs in {time.time() - t0:.1f}s ({len(candidate_pairs):,} candidate pairs)")

    train_pairs, val_pairs = build_training_dataset(
        ground_truth,
        candidate_pairs,
        source1,
        {"S2": s2, "S3": s3},
        negatives_per_positive=args.negatives_per_positive,
        val_size=args.val_size,
        random_state=args.random_state,
        stratify=not args.no_stratify,
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
