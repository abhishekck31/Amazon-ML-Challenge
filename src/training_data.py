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


def load_candidate_pairs(path: str) -> pd.DataFrame:
    """
    Load candidate_pairs.tsv with categorical id columns, dictionary-encoded by Arrow
    while parsing. At full scale the file is 400M+ rows: pd.read_csv(dtype=str) holds
    ~25-30GB of Python string objects, and pd.read_csv(dtype="category") still builds
    those strings before encoding them (it peaked above 48GB before even finishing the
    load on a 64GB machine). Arrow's dictionary columns never materialize per-row
    strings, so the result is a few GB of int32 codes plus the unique ids.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    dictionary_type = pa.dictionary(pa.int32(), pa.string())
    table = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            include_columns=PAIR_ID_COLUMNS,
            column_types={col: dictionary_type for col in PAIR_ID_COLUMNS},
        ),
    )
    return table.unify_dictionaries().to_pandas()


def _as_categorical(series: pd.Series) -> pd.Series:
    return series if isinstance(series.dtype, pd.CategoricalDtype) else series.astype("category")


def _pack_keys(s1_codes: np.ndarray, cand_codes: np.ndarray) -> np.ndarray:
    keys = s1_codes.astype(np.int64)
    keys <<= 32
    keys |= cand_codes.astype(np.int64)
    return keys


def _membership_mask(keys: np.ndarray, true_keys: np.ndarray, chunk_size: int = 50_000_000) -> np.ndarray:
    """keys-in-true_keys via binary search against the (small) sorted true set, in chunks,
    so temporaries stay bounded at chunk_size rather than len(keys). np.isin's sort path
    would concatenate and argsort all 400M+ keys at once."""
    true_sorted = np.unique(true_keys)
    mask = np.zeros(len(keys), dtype=bool)
    if true_sorted.size == 0:
        return mask
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start:start + chunk_size]
        idx = np.searchsorted(true_sorted, chunk)
        np.minimum(idx, true_sorted.size - 1, out=idx)
        mask[start:start + chunk_size] = true_sorted[idx] == chunk
    return mask


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
    negative pool and unbalance the dataset: shuffle, then keep each entity's first
    `quota` rows.

    Works on categorical codes and row positions only. At full scale candidate_pairs is
    400M+ rows; every earlier version that touched it as Python strings or copied the
    whole frame (a Python set of tuples, a concatenated string key + .isin(), a
    .loc[~mask].sample(frac=1.0) shuffle) exhausted 64GB. Pass candidate_pairs with
    categorical id columns (see main()) to avoid holding 400M+ string objects at all;
    object columns still work but are converted here.
    """
    s1_col = _as_categorical(candidate_pairs["source1_entity_id"])
    cand_col = _as_categorical(candidate_pairs["candidate_entity_id"])
    cand_s1 = s1_col.cat.codes.to_numpy()
    cand_c = cand_col.cat.codes.to_numpy()

    # Map positives onto the candidates' category codes; -1 = id never appears in any
    # candidate pair, so that positive can't match (or block-sample) anything.
    pos_s1 = s1_col.cat.categories.get_indexer(positive_pairs["source1_entity_id"])
    pos_c = cand_col.cat.categories.get_indexer(positive_pairs["candidate_entity_id"])
    both_known = (pos_s1 >= 0) & (pos_c >= 0)

    candidate_keys = _pack_keys(cand_s1, cand_c)
    is_positive = _membership_mask(candidate_keys, _pack_keys(pos_s1[both_known], pos_c[both_known]))
    del candidate_keys

    neg_positions = np.flatnonzero(~is_positive & (cand_s1 >= 0))
    del is_positive
    if neg_positions.size == 0:
        empty = candidate_pairs.iloc[:0][PAIR_ID_COLUMNS].astype(str)
        return empty.assign(label=pd.Series(dtype=int))

    rng = np.random.default_rng(random_state)
    rng.shuffle(neg_positions)

    # Rank each shuffled negative within its source1 entity: stable-sort by entity code
    # (keeps the shuffled order inside each entity), then rank = position - group start.
    neg_codes = cand_s1[neg_positions]
    order = np.argsort(neg_codes, kind="stable")
    sorted_codes = neg_codes[order]
    del neg_codes

    n = sorted_codes.size
    group_start = np.zeros(n, dtype=bool)
    group_start[0] = True
    np.not_equal(sorted_codes[1:], sorted_codes[:-1], out=group_start[1:])
    start_idx = np.flatnonzero(group_start)
    del group_start
    rank = np.arange(n, dtype=np.int64)
    rank -= np.repeat(start_idx, np.diff(np.append(start_idx, n)))

    positives_per_code = np.bincount(pos_s1[pos_s1 >= 0], minlength=len(s1_col.cat.categories))
    quota_per_code = np.maximum(positives_per_code, 1) * negatives_per_positive
    keep = rank < quota_per_code[sorted_codes]
    del rank, sorted_codes

    final_positions = neg_positions[order[keep]]
    negatives = candidate_pairs.iloc[final_positions][PAIR_ID_COLUMNS].astype(str)
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
    candidate_pairs = load_candidate_pairs(args.candidate_pairs)
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
