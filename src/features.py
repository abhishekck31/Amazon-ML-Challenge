"""
Phase 4 - Feature engineering for Amazon ML Challenge 2026: Business Entity Resolution.

Turns each blocking-stage candidate pair (src.blocking.generate_candidate_pairs) into a
row of numeric similarity features a LightGBM ranker/classifier can train on.

Every string-similarity feature is computed with RapidFuzz's C-accelerated scorers.
RapidFuzz has no vectorized "many arbitrary pairs" API (rapidfuzz.process.cdist computes
a full cross product, which is not what a sparse candidate-pair list needs), so each
scorer is applied pairwise over aligned Python lists built from a single O(n) hash-index
join keyed by entity_id - no iterrows(), no per-pair DataFrame lookups.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.data_loader import load_tsv
from src.preprocessing import add_clean_columns

_NUMERIC_RE = re.compile(r"\d+")

ID_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source"]

FEATURE_COLUMNS = [
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "address_token_similarity",
    "address_partial_similarity",
    "exact_country_match",
    "first_word_match",
    "prefix_match",
    "name_length_diff",
    "address_length_diff",
    "numeric_overlap",
    "jaccard_token_similarity",
]


# --------------------------------------------------------------------------- #
# Small pure-Python helpers (operate on already-normalized name_clean /
# address_clean strings from src.preprocessing)
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> List[str]:
    return text.split() if text else []


def _numeric_tokens(text: str) -> set:
    return set(_NUMERIC_RE.findall(text)) if text else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _build_lookup(df: pd.DataFrame, id_col: str = "entity_id") -> Dict[str, Tuple[str, str, str]]:
    """entity_id -> (name_clean, address_clean, country), for O(1) per-pair joins."""
    required = ["name_clean", "address_clean", "country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns {missing}; "
            "run src.preprocessing.add_clean_columns() first"
        )
    ids = df[id_col].to_numpy()
    names = df["name_clean"].fillna("").to_numpy()
    addrs = df["address_clean"].fillna("").to_numpy()
    countries = df["country"].fillna("").to_numpy()
    return {i: (n, a, c) for i, n, a, c in zip(ids, names, addrs, countries)}


# --------------------------------------------------------------------------- #
# Feature generation
# --------------------------------------------------------------------------- #

def generate_features(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    prefix_len: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build one feature row per (source1_entity_id, candidate_entity_id) pair in
    `candidate_pairs` (as produced by src.blocking.generate_candidate_pairs).

    `source1` and every DataFrame in `targets` (e.g. {"S2": s2_df, "S3": s3_df}) must
    already carry name_clean / address_clean / country - run
    src.preprocessing.add_clean_columns() on them first if they don't.

    Returns id columns (source1_entity_id, candidate_entity_id, source) plus the
    twelve numeric similarity features listed in FEATURE_COLUMNS. Use
    get_feature_matrix() to drop the id columns before handing X to LightGBM.
    """
    t0 = time.time()
    s1_lookup = _build_lookup(source1)
    target_lookups = {label: _build_lookup(df) for label, df in targets.items()}

    s1_ids = candidate_pairs["source1_entity_id"].to_numpy()
    cand_ids = candidate_pairs["candidate_entity_id"].to_numpy()
    sources = candidate_pairs["source"].to_numpy()

    n = len(candidate_pairs)
    token_sort_ratio = np.empty(n, dtype=np.float32)
    token_set_ratio = np.empty(n, dtype=np.float32)
    partial_ratio = np.empty(n, dtype=np.float32)
    address_token_similarity = np.empty(n, dtype=np.float32)
    address_partial_similarity = np.empty(n, dtype=np.float32)
    exact_country_match = np.empty(n, dtype=np.int8)
    first_word_match = np.empty(n, dtype=np.int8)
    prefix_match = np.empty(n, dtype=np.int8)
    name_length_diff = np.empty(n, dtype=np.int32)
    address_length_diff = np.empty(n, dtype=np.int32)
    numeric_overlap = np.empty(n, dtype=np.float32)
    jaccard_token_similarity = np.empty(n, dtype=np.float32)

    empty_record = ("", "", "")
    for i, (s1_id, cand_id, src) in enumerate(zip(s1_ids, cand_ids, sources)):
        name1, addr1, country1 = s1_lookup.get(s1_id, empty_record)
        name2, addr2, country2 = target_lookups[src].get(cand_id, empty_record)

        token_sort_ratio[i] = fuzz.token_sort_ratio(name1, name2)
        token_set_ratio[i] = fuzz.token_set_ratio(name1, name2)
        partial_ratio[i] = fuzz.partial_ratio(name1, name2)

        address_token_similarity[i] = fuzz.token_sort_ratio(addr1, addr2)
        address_partial_similarity[i] = fuzz.partial_ratio(addr1, addr2)

        exact_country_match[i] = int(bool(country1) and country1 == country2)

        t1, t2 = _tokens(name1), _tokens(name2)
        first_word_match[i] = int(bool(t1) and bool(t2) and t1[0] == t2[0])
        prefix_match[i] = int(bool(name1) and name1[:prefix_len] == name2[:prefix_len])
        name_length_diff[i] = abs(len(name1) - len(name2))
        jaccard_token_similarity[i] = _jaccard(set(t1), set(t2))

        address_length_diff[i] = abs(len(addr1) - len(addr2))
        numeric_overlap[i] = _jaccard(_numeric_tokens(addr1), _numeric_tokens(addr2))

    features = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "source": sources,
        "token_sort_ratio": token_sort_ratio,
        "token_set_ratio": token_set_ratio,
        "partial_ratio": partial_ratio,
        "address_token_similarity": address_token_similarity,
        "address_partial_similarity": address_partial_similarity,
        "exact_country_match": exact_country_match,
        "first_word_match": first_word_match,
        "prefix_match": prefix_match,
        "name_length_diff": name_length_diff,
        "address_length_diff": address_length_diff,
        "numeric_overlap": numeric_overlap,
        "jaccard_token_similarity": jaccard_token_similarity,
    })

    if verbose:
        print(f"Generated {n:,} feature rows in {time.time() - t0:.1f}s")

    return features


def attach_labels(features: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary `label` column (1 if the pair is a true match per `ground_truth`,
    0 otherwise) for supervised LightGBM training. Only meaningful when `features`
    was built from candidate pairs covering the same source1 rows as `ground_truth`.
    """
    exploded = ground_truth.assign(
        candidate_entity_id=ground_truth["matched_entity_ids"].str.split(",")
    )
    exploded = exploded.explode("candidate_entity_id")
    exploded = exploded[["source1_entity_id", "candidate_entity_id"]].dropna()
    true_pairs = set(zip(exploded["source1_entity_id"], exploded["candidate_entity_id"]))

    labels = np.fromiter(
        (
            int((s1, c) in true_pairs)
            for s1, c in zip(features["source1_entity_id"], features["candidate_entity_id"])
        ),
        dtype=np.int8,
        count=len(features),
    )
    return features.assign(label=labels)


def get_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Drop id/source/label columns, leaving only the numeric matrix LightGBM trains on."""
    drop_cols = [c for c in ID_COLUMNS + ["label"] if c in features.columns]
    return features.drop(columns=drop_cols)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 feature engineering on a candidate-pairs file.")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv")
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output", default="output/features.tsv")
    parser.add_argument("--attach-labels", action="store_true",
                         help="Add a `label` column from train_ground_truth.tsv (train split only).")
    args = parser.parse_args()

    t0 = time.time()
    prefix = args.split
    source1 = add_clean_columns(load_tsv(os.path.join(args.data_dir, f"{prefix}_source1.tsv")))
    s2 = add_clean_columns(load_tsv(os.path.join(args.data_dir, f"{prefix}_source2.tsv")))
    s3 = add_clean_columns(load_tsv(os.path.join(args.data_dir, f"{prefix}_source3.tsv")))
    candidate_pairs = pd.read_csv(args.candidate_pairs, sep="\t")
    print(f"Loaded sources + candidate pairs in {time.time() - t0:.1f}s "
          f"({len(candidate_pairs):,} pairs)")

    features = generate_features(candidate_pairs, source1, {"S2": s2, "S3": s3})

    if args.attach_labels:
        gt_path = os.path.join(args.data_dir, f"{prefix}_ground_truth.tsv")
        ground_truth = load_tsv(gt_path)
        features = attach_labels(features, ground_truth)
        print(f"Attached labels: {features['label'].sum():,} positive / {len(features):,} total")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    features.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(features):,} feature rows to {args.output} in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
