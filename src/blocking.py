"""
Phase 3 - Candidate generation (blocking) for Amazon ML Challenge 2026: Business Entity Resolution.

Builds a memory-efficient, fully vectorized blocking pipeline that turns the naive
O(|source1| x |source2 + source3|) matching problem into a small set of high-recall
candidate pairs, by grouping records into "blocks" that are cheap to compute (country,
name/address prefixes, shared tokens, shared numeric substrings) and only pairing
records that share at least one block.

No row is ever visited with iterrows(). Every rule is built with vectorized pandas
string ops plus groupby(...).indices, which is the standard O(n) / Cython-backed way
to turn a column into an inverted index {block_key: array_of_row_positions}.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing import add_clean_columns

EMPTY_INDEX = np.empty(0, dtype=np.int64)

# Tokens that appear in a huge fraction of business names and add no discriminative
# power to name-token blocking (legal suffixes are already stripped by
# src.preprocessing.normalize_name, this list covers common generic business words).
DEFAULT_STOP_TOKENS = {
    "the", "and", "of", "a", "an", "for", "group", "international", "global",
    "company", "enterprise", "enterprises", "solutions", "services", "trading",
    "store", "shop", "holdings",
}


# --------------------------------------------------------------------------- #
# Text normalization (fallback in case name_clean / address_clean were not
# already produced upstream by Phase 2's src.preprocessing)
# --------------------------------------------------------------------------- #

def ensure_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add name_clean / address_clean (via src.preprocessing) if a caller passes raw
    Phase-1 data instead of Phase-2 output."""
    if "name_clean" in df.columns and "address_clean" in df.columns:
        return df
    return add_clean_columns(df)


# --------------------------------------------------------------------------- #
# Single-key blocking rules (rules 1-3, 6a-6b): one deterministic string key per row
# --------------------------------------------------------------------------- #

def _key_country_name_prefix(df: pd.DataFrame, n: int = 3) -> pd.Series:
    """Rule 1: country + first n characters of the cleaned business name."""
    return df["country"].fillna("").str.upper() + "|" + df["name_clean"].fillna("").str.slice(0, n)


def _key_country_name_firstword(df: pd.DataFrame) -> pd.Series:
    """Rule 2: country + first whitespace-delimited token of the cleaned name."""
    first_word = df["name_clean"].fillna("").str.split().str[0].fillna("")
    return df["country"].fillna("").str.upper() + "|" + first_word


def _key_country_address_prefix(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """Rule 3: country + prefix of the whitespace-stripped cleaned address."""
    compact = df["address_clean"].fillna("").str.replace(r"\s+", "", regex=True)
    return df["country"].fillna("").str.upper() + "|" + compact.str.slice(0, n)


def _key_country_name_lastword(df: pd.DataFrame) -> pd.Series:
    """Rule 6a: country + last token of the cleaned name (catches prefix drift, e.g. legal-form
    variants or "Saint"/"St" style abbreviations at the start of the name)."""
    last_word = df["name_clean"].fillna("").str.split().str[-1].fillna("")
    return df["country"].fillna("").str.upper() + "|" + last_word


def _key_country_name_initials(df: pd.DataFrame, max_tokens: int = 5) -> pd.Series:
    """Rule 6b: country + acronym formed from each name token's first letter, sorted.
    Catches token-reordering / abbreviation variants (e.g. "Global Trade Bank" vs
    "Trade Global Bank") that prefix/first-word rules miss."""
    def initials(tokens: List[str]) -> str:
        letters = sorted(t[0] for t in tokens[:max_tokens] if t)
        return "".join(letters)

    tok_lists = df["name_clean"].fillna("").str.split()
    initials_series = tok_lists.apply(initials)
    return df["country"].fillna("").str.upper() + "|" + initials_series


def build_block_indices(key_series: pd.Series) -> Dict[str, np.ndarray]:
    """
    Turn a per-row string key into an inverted index {block_key: array_of_row_positions},
    using groupby(...).indices (vectorized, Cython-backed grouping - no iterrows()).

    Rows whose key is the empty string are dropped: an empty key usually means missing
    data (e.g. blank name/address) and would otherwise form one giant, useless block.
    """
    positions = pd.Series(np.arange(len(key_series)))
    grouped = positions.groupby(key_series.values).indices
    return {k: v.astype(np.int64) for k, v in grouped.items() if k and not k.endswith("|")}


def get_candidate_indices(block_index: Dict[str, np.ndarray], keys) -> np.ndarray:
    """Union of row positions for one or more block keys; empty array if none match."""
    if isinstance(keys, str):
        keys = [keys]
    arrays = [block_index[k] for k in keys if k in block_index]
    if not arrays:
        return EMPTY_INDEX
    if len(arrays) == 1:
        return arrays[0]
    return np.unique(np.concatenate(arrays))


# --------------------------------------------------------------------------- #
# Token-based inverted indices (rules 4-5): a row can belong to many blocks
# --------------------------------------------------------------------------- #

def build_token_index(
    df: pd.DataFrame,
    column: str,
    min_token_len: int = 3,
    max_df_ratio: float = 0.01,
    stop_tokens: Optional[set] = None,
) -> Dict[str, np.ndarray]:
    """
    Rule 4: inverted index {token: array_of_row_positions} over whitespace tokens of
    `column`. Tokens shorter than `min_token_len`, in `stop_tokens`, or appearing in
    more than `max_df_ratio` of rows are dropped - those are exactly the tokens that
    are cheap to match but contribute no recall while blowing up candidate counts.
    """
    stop_tokens = stop_tokens or DEFAULT_STOP_TOKENS
    n = len(df)
    tok_lists = df[column].fillna("").str.split()
    tok_lists.index = np.arange(n)  # row position, reused as-is by explode()

    # explode() keeps the original (positional) index for every element it produces,
    # and emits a single NaN row for an empty list - dropna() removes those cleanly,
    # so the exploded index is always exactly the row position of each surviving token.
    exploded = tok_lists.explode().dropna()
    exploded = exploded[exploded.str.len() >= min_token_len]

    doc_freq = exploded.value_counts()
    max_df = max_df_ratio * max(n, 1)
    keep_tokens = set(doc_freq[doc_freq <= max_df].index) - stop_tokens
    exploded = exploded[exploded.isin(keep_tokens)]

    idx = pd.Series(exploded.index.values, index=exploded.values)
    return {k: v.to_numpy(dtype=np.int64) for k, v in idx.groupby(level=0)}


def build_numeric_token_index(
    df: pd.DataFrame,
    column: str,
    min_digits: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Rule 5: inverted index {number_string: array_of_row_positions} over numeric
    substrings (house numbers, zip codes, ...) with at least `min_digits` digits,
    extracted from `column`. Short numbers (unit numbers, single-digit noise) are
    skipped since they are far too common to be discriminative.
    """
    n = len(df)
    number_lists = df[column].fillna("").str.findall(r"\d+")
    number_lists = number_lists.apply(lambda lst: [t for t in lst if len(t) >= min_digits])
    number_lists.index = np.arange(n)  # row position, reused as-is by explode()

    exploded = number_lists.explode().dropna()
    idx = pd.Series(exploded.index.values, index=exploded.values)
    return {k: v.to_numpy(dtype=np.int64) for k, v in idx.groupby(level=0)}


# --------------------------------------------------------------------------- #
# Rule registry
# --------------------------------------------------------------------------- #

@dataclass
class BlockingRule:
    name: str
    kind: str  # "single_key" | "token" | "numeric_token"
    key_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    column: Optional[str] = None
    kwargs: dict = field(default_factory=dict)


DEFAULT_RULES: List[BlockingRule] = [
    BlockingRule("country_name_prefix3", "single_key", key_fn=lambda df: _key_country_name_prefix(df, 3)),
    BlockingRule("country_name_firstword", "single_key", key_fn=_key_country_name_firstword),
    BlockingRule("country_address_prefix10", "single_key", key_fn=lambda df: _key_country_address_prefix(df, 10)),
    BlockingRule("name_token", "token", column="name_clean",
                 kwargs={"min_token_len": 3, "max_df_ratio": 0.01}),
    BlockingRule("address_numeric_token", "numeric_token", column="address_clean",
                 kwargs={"min_digits": 3}),
    BlockingRule("country_name_lastword", "single_key", key_fn=_key_country_name_lastword),
    BlockingRule("country_name_initials", "single_key", key_fn=lambda df: _key_country_name_initials(df, 5)),
]


def build_source_block_indices(df: pd.DataFrame, rules: List[BlockingRule]) -> Dict[str, Dict[str, np.ndarray]]:
    """Materialize every rule's inverted index for one source dataframe."""
    indices: Dict[str, Dict[str, np.ndarray]] = {}
    for rule in rules:
        if rule.kind == "single_key":
            indices[rule.name] = build_block_indices(rule.key_fn(df))
        elif rule.kind == "token":
            indices[rule.name] = build_token_index(df, rule.column, **rule.kwargs)
        elif rule.kind == "numeric_token":
            indices[rule.name] = build_numeric_token_index(df, rule.column, **rule.kwargs)
        else:
            raise ValueError(f"Unknown rule kind: {rule.kind!r}")
    return indices


def _pairs_from_matching_blocks(
    s1_index: Dict[str, np.ndarray],
    target_index: Dict[str, np.ndarray],
    max_block_pairs: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every block key present on both sides, emit the full cross product of
    (source1_position, target_position) pairs, using np.repeat/np.tile (vectorized,
    no per-pair Python loop). A block whose cross product would exceed
    `max_block_pairs` is skipped - such blocks are almost always an under-specific
    key (e.g. a very common country + short prefix) and would dominate memory/output
    size without materially improving recall, since other rules cover the same rows.
    """
    common_keys = s1_index.keys() & target_index.keys()
    left_parts: List[np.ndarray] = []
    right_parts: List[np.ndarray] = []
    skipped = 0
    for key in common_keys:
        left = s1_index[key]
        right = target_index[key]
        if left.size * right.size > max_block_pairs:
            skipped += 1
            continue
        left_parts.append(np.repeat(left, right.size))
        right_parts.append(np.tile(right, left.size))

    if skipped:
        print(f"    skipped {skipped:,} oversized block(s) (> {max_block_pairs:,} pairs)")

    if not left_parts:
        return EMPTY_INDEX, EMPTY_INDEX
    return np.concatenate(left_parts), np.concatenate(right_parts)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_candidate_pairs(
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    rules: Optional[List[BlockingRule]] = None,
    max_block_pairs: int = 200_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate deduplicated candidate pairs between `source1` and one or more target
    sources (e.g. {"S2": source2_df, "S3": source3_df}) by unioning the pairs
    produced by every rule in `rules`.

    Country is treated as an open-set string throughout (no fixed vocabulary /
    category dtype), so countries unseen during training are handled the same way
    as any other value at inference time.

    Returns a DataFrame: source1_entity_id, candidate_entity_id, source, rules_matched.
    """
    rules = rules or DEFAULT_RULES
    t0 = time.time()
    s1_indices = build_source_block_indices(source1, rules)
    if verbose:
        print(f"Built source1 block indices in {time.time() - t0:.1f}s")

    all_pair_frames = []
    for source_label, target_df in targets.items():
        t_src = time.time()
        target_indices = build_source_block_indices(target_df, rules)

        rule_frames = []
        for rule in rules:
            left_pos, right_pos = _pairs_from_matching_blocks(
                s1_indices[rule.name], target_indices[rule.name], max_block_pairs
            )
            if left_pos.size == 0:
                continue
            frame = pd.DataFrame({
                "source1_entity_id": source1["entity_id"].values[left_pos],
                "candidate_entity_id": target_df["entity_id"].values[right_pos],
                "rule": rule.name,
            })
            rule_frames.append(frame)
            if verbose:
                print(f"  [{source_label}] rule={rule.name}: {len(frame):,} raw pairs")

        if not rule_frames:
            continue

        merged = pd.concat(rule_frames, ignore_index=True)
        agg = (
            merged.groupby(["source1_entity_id", "candidate_entity_id"])["rule"]
            .apply(lambda s: ",".join(sorted(set(s))))
            .reset_index()
            .rename(columns={"rule": "rules_matched"})
        )
        agg["source"] = source_label
        all_pair_frames.append(agg)

        if verbose:
            print(f"[{source_label}] {len(agg):,} deduped candidate pairs in {time.time() - t_src:.1f}s")

    if not all_pair_frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source", "rules_matched"])

    result = pd.concat(all_pair_frames, ignore_index=True)
    return result[["source1_entity_id", "candidate_entity_id", "source", "rules_matched"]]


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def compute_blocking_stats(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
) -> dict:
    """Average candidates per source1 entity + reduction ratio vs. the naive full cross product."""
    n_s1 = len(source1)
    total_target_rows = sum(len(df) for df in targets.values())
    n_pairs = len(candidate_pairs)
    full_cross_product = n_s1 * total_target_rows

    return {
        "n_source1_entities": n_s1,
        "n_candidate_pairs": n_pairs,
        "avg_candidates_per_entity": n_pairs / n_s1 if n_s1 else 0.0,
        "full_cross_product_size": full_cross_product,
        "reduction_ratio": (1 - n_pairs / full_cross_product) if full_cross_product else 0.0,
    }


def evaluate_blocking_recall(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    sample_size: Optional[int] = 20_000,
    random_state: int = 42,
) -> dict:
    """
    Recall = fraction of true (source1_entity_id, matched_entity_id) pairs from
    `ground_truth` that also appear in `candidate_pairs`. Evaluated on a random
    sample of ground-truth rows (`sample_size`) so this stays cheap on the full
    2M+ row training set; pass sample_size=None to evaluate exactly.
    """
    gt = ground_truth
    if sample_size is not None and len(gt) > sample_size:
        gt = gt.sample(n=sample_size, random_state=random_state)

    exploded = gt.assign(matched_entity_id=gt["matched_entity_ids"].str.split(","))
    exploded = exploded.explode("matched_entity_id")[["source1_entity_id", "matched_entity_id"]].dropna()

    candidate_set = set(zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"]))
    true_pairs = list(zip(exploded["source1_entity_id"], exploded["matched_entity_id"]))
    hits = sum(1 for pair in true_pairs if pair in candidate_set)

    return {
        "n_ground_truth_pairs_sampled": len(true_pairs),
        "n_hits": hits,
        "recall": hits / len(true_pairs) if true_pairs else 0.0,
    }


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def load_sources(
    data_dir: str, split: str = "train"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    prefix = split
    s1 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source3.tsv"), sep="\t")
    gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
    gt = pd.read_csv(gt_path, sep="\t") if os.path.exists(gt_path) else None
    return s1, s2, s3, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 blocking / candidate generation pipeline.")
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                         help="File prefix to load (train_*.tsv or test_*.tsv).")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-block-pairs", type=int, default=200_000)
    parser.add_argument("--recall-sample", type=int, default=20_000)
    args = parser.parse_args()

    t0 = time.time()
    s1, s2, s3, gt = load_sources(args.data_dir, args.split)
    s1, s2, s3 = (ensure_clean_columns(df) for df in (s1, s2, s3))
    print(f"Loaded sources in {time.time() - t0:.1f}s "
          f"(S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,})")

    pairs = generate_candidate_pairs(s1, {"S2": s2, "S3": s3}, max_block_pairs=args.max_block_pairs)

    stats = compute_blocking_stats(pairs, s1, {"S2": s2, "S3": s3})
    print("Blocking stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")

    if gt is not None:
        recall_stats = evaluate_blocking_recall(pairs, gt, sample_size=args.recall_sample)
        print("Recall stats:")
        for k, v in recall_stats.items():
            print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pairs.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(pairs):,} candidate pairs to {args.output} in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
