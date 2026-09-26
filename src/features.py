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
from rapidfuzz.distance import JaroWinkler, LCSseq

from src.data_loader import load_tsv
from src.preprocessing import add_clean_columns

_NUMERIC_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"\d{5,6}\b")

ID_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source"]

FEATURE_COLUMNS = [
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "name_ratio",
    "name_wratio",
    "name_jaro_winkler",
    "address_token_similarity",
    "address_partial_similarity",
    "address_jaro_winkler",
    "address_lcs_ratio",
    "exact_country_match",
    "first_word_match",
    "prefix_match",
    "name_length_diff",
    "address_length_diff",
    "numeric_overlap",
    "jaccard_token_similarity",
    "name_bigram_jaccard",
    "name_trigram_dice",
    "postal_code_match",
    "rank_within_entity",
    "is_top1_for_entity",
    "score_margin_to_next",
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


def _dice(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    total = len(a) + len(b)
    return 2 * len(a & b) / total if total else 0.0


def _char_ngrams(text: str, n: int) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _postal_code(text: str):
    matches = _POSTAL_RE.findall(text)
    return matches[-1] if matches else None


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

    empty_record = ("", "", "")
    left = [s1_lookup.get(s1_id, empty_record) for s1_id in s1_ids]
    right = [target_lookups[src].get(cand_id, empty_record) for cand_id, src in zip(cand_ids, sources)]
    pair_features = compute_pair_features(
        [r[0] for r in left], [r[1] for r in left], [r[2] for r in left],
        [r[0] for r in right], [r[1] for r in right], [r[2] for r in right],
        prefix_len=prefix_len,
    )

    features = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "source": sources,
        **pair_features,
    })
    features = _add_relative_rank_features(features)

    if verbose:
        print(f"Generated {len(features):,} feature rows in {time.time() - t0:.1f}s")

    return features


def compute_pair_features(names1, addrs1, countries1, names2, addrs2, countries2, prefix_len: int = 3):
    """
    The per-pair (non entity-relative) features, from aligned sequences of already
    normalized name/address/country strings for each side of each pair. Returns
    {column: array} in FEATURE_COLUMNS order. Takes plain strings rather than entity
    ids so it can run in a worker process without the full source lookup tables
    (see src.scoring).
    """
    n = len(names1)
    token_sort_ratio = np.empty(n, dtype=np.float32)
    token_set_ratio = np.empty(n, dtype=np.float32)
    partial_ratio = np.empty(n, dtype=np.float32)
    name_ratio = np.empty(n, dtype=np.float32)
    name_wratio = np.empty(n, dtype=np.float32)
    name_jaro_winkler = np.empty(n, dtype=np.float32)
    address_token_similarity = np.empty(n, dtype=np.float32)
    address_partial_similarity = np.empty(n, dtype=np.float32)
    address_jaro_winkler = np.empty(n, dtype=np.float32)
    address_lcs_ratio = np.empty(n, dtype=np.float32)
    exact_country_match = np.empty(n, dtype=np.int8)
    first_word_match = np.empty(n, dtype=np.int8)
    prefix_match = np.empty(n, dtype=np.int8)
    name_length_diff = np.empty(n, dtype=np.int32)
    address_length_diff = np.empty(n, dtype=np.int32)
    numeric_overlap = np.empty(n, dtype=np.float32)
    jaccard_token_similarity = np.empty(n, dtype=np.float32)
    name_bigram_jaccard = np.empty(n, dtype=np.float32)
    name_trigram_dice = np.empty(n, dtype=np.float32)
    postal_code_match = np.empty(n, dtype=np.int8)

    for i, (name1, addr1, country1, name2, addr2, country2) in enumerate(
        zip(names1, addrs1, countries1, names2, addrs2, countries2)
    ):
        token_sort_ratio[i] = fuzz.token_sort_ratio(name1, name2)
        token_set_ratio[i] = fuzz.token_set_ratio(name1, name2)
        partial_ratio[i] = fuzz.partial_ratio(name1, name2)
        name_ratio[i] = fuzz.ratio(name1, name2)
        name_wratio[i] = fuzz.WRatio(name1, name2)
        name_jaro_winkler[i] = JaroWinkler.normalized_similarity(name1, name2) * 100.0

        address_token_similarity[i] = fuzz.token_sort_ratio(addr1, addr2)
        address_partial_similarity[i] = fuzz.partial_ratio(addr1, addr2)
        address_jaro_winkler[i] = JaroWinkler.normalized_similarity(addr1, addr2) * 100.0
        address_lcs_ratio[i] = LCSseq.normalized_similarity(addr1, addr2) * 100.0

        exact_country_match[i] = int(bool(country1) and country1 == country2)

        t1, t2 = _tokens(name1), _tokens(name2)
        first_word_match[i] = int(bool(t1) and bool(t2) and t1[0] == t2[0])
        prefix_match[i] = int(bool(name1) and name1[:prefix_len] == name2[:prefix_len])
        name_length_diff[i] = abs(len(name1) - len(name2))
        jaccard_token_similarity[i] = _jaccard(set(t1), set(t2))
        name_bigram_jaccard[i] = _jaccard(_char_ngrams(name1, 2), _char_ngrams(name2, 2))
        name_trigram_dice[i] = _dice(_char_ngrams(name1, 3), _char_ngrams(name2, 3))

        address_length_diff[i] = abs(len(addr1) - len(addr2))
        numeric_overlap[i] = _jaccard(_numeric_tokens(addr1), _numeric_tokens(addr2))
        postal1, postal2 = _postal_code(addr1), _postal_code(addr2)
        postal_code_match[i] = int(postal1 is not None and postal1 == postal2)

    return {
        "token_sort_ratio": token_sort_ratio,
        "token_set_ratio": token_set_ratio,
        "partial_ratio": partial_ratio,
        "name_ratio": name_ratio,
        "name_wratio": name_wratio,
        "name_jaro_winkler": name_jaro_winkler,
        "address_token_similarity": address_token_similarity,
        "address_partial_similarity": address_partial_similarity,
        "address_jaro_winkler": address_jaro_winkler,
        "address_lcs_ratio": address_lcs_ratio,
        "exact_country_match": exact_country_match,
        "first_word_match": first_word_match,
        "prefix_match": prefix_match,
        "name_length_diff": name_length_diff,
        "address_length_diff": address_length_diff,
        "numeric_overlap": numeric_overlap,
        "jaccard_token_similarity": jaccard_token_similarity,
        "name_bigram_jaccard": name_bigram_jaccard,
        "name_trigram_dice": name_trigram_dice,
        "postal_code_match": postal_code_match,
    }


def _add_relative_rank_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Entity-relative ranking signals: among all candidates blocked for the SAME
    Source1 entity, how does this one compare? A candidate that looks similar in
    isolation but is clearly second-best within its own entity's pool is much weaker
    evidence than the single best-scoring candidate for that entity - these features
    let the model use that group structure instead of scoring every pair as if it
    existed alone. Fully vectorized (groupby rank/transform), no per-entity Python loop.
    """
    basis = (features["token_sort_ratio"].astype(np.float64) + features["address_token_similarity"]) / 2.0
    groups = features["source1_entity_id"]

    rank = basis.groupby(groups, sort=False).rank(method="first", ascending=False)
    top1_score = basis.groupby(groups, sort=False).transform("max")
    second_place_score = (
        basis.where(rank == 2).groupby(groups, sort=False).transform("max")
    )
    # No runner-up (this entity has only one candidate) -> margin defaults to the
    # candidate's own score, i.e. "no competition, maximally uncontested".
    margin = (top1_score - second_place_score).fillna(top1_score)

    features = features.copy()
    features["rank_within_entity"] = rank.astype(np.int32)
    features["is_top1_for_entity"] = (rank == 1).astype(np.int8)
    features["score_margin_to_next"] = np.where(rank == 1, margin, 0.0).astype(np.float32)
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
    candidate_pairs = pd.read_csv(args.candidate_pairs, sep="\t", dtype=str)
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
