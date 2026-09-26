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

import jellyfish
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


def _key_country_name_phonetic(df: pd.DataFrame) -> pd.Series:
    """Rule 6c: country + Metaphone phonetic code of the name's first word. Catches
    transliteration variants that no exact-match rule can (e.g. "Kumar" vs "Coomar",
    "Sri" vs "Shree") - especially common where the same Indian business name gets
    romanized differently across sources. jellyfish has no vectorized batch API, so
    this runs one C-accelerated call per row via .apply(), same pattern as the
    initials rule above."""
    first_word = df["name_clean"].fillna("").str.split().str[0].fillna("")
    phonetic = first_word.apply(lambda w: jellyfish.metaphone(w) if w else "")
    return df["country"].fillna("").str.upper() + "|" + phonetic


def build_block_indices(key_series: pd.Series, max_df_ratio: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Turn a per-row string key into an inverted index {block_key: array_of_row_positions},
    using groupby(...).indices (vectorized, Cython-backed grouping - no iterrows()).

    Rows whose key is the empty string are dropped: an empty key usually means missing
    data (e.g. blank name/address) and would otherwise form one giant, useless block.

    `max_df_ratio`, if given, additionally drops any key shared by more than that
    fraction of rows. This matters at full dataset scale in a way it doesn't on a
    small sample: _pairs_from_matching_blocks's max_block_pairs cap only skips a block
    whose cross product is individually huge, but a broad key like a common first word
    ("the", "global", ...) forms thousands of merely medium-sized blocks across a
    multi-million-row dataset that each pass that cap yet sum to tens of millions of
    low-value pairs. Capping document frequency here - the same technique
    build_token_index already uses for token blocking - prunes those low-signal keys
    before pair generation instead of after, which is what actually keeps full-scale
    runs tractable.
    """
    positions = pd.Series(np.arange(len(key_series)))
    grouped = positions.groupby(key_series.values).indices
    index = {k: v.astype(np.int64) for k, v in grouped.items() if k and not k.endswith("|")}
    if max_df_ratio is not None and len(key_series):
        max_count = max_df_ratio * len(key_series)
        index = {k: v for k, v in index.items() if len(v) <= max_count}
    return index


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
    max_df_ratio: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Rule 5: inverted index {number_string: array_of_row_positions} over numeric
    substrings (house numbers, zip codes, ...) with at least `min_digits` digits,
    extracted from `column`. Short numbers (unit numbers, single-digit noise) are
    skipped since they are far too common to be discriminative. `max_df_ratio` drops
    numbers shared by more than that fraction of rows (e.g. a common zip code prefix),
    same doc-frequency capping build_token_index already applies to name tokens -
    without it this rule was the single largest uncapped contributor to full-scale
    candidate-pair blowup (observed: 69-70M raw pairs per source on the full training
    set, with no ceiling on how common a given digit string could be).
    """
    n = len(df)
    number_lists = df[column].fillna("").str.findall(r"\d+")
    number_lists = number_lists.apply(lambda lst: [t for t in lst if len(t) >= min_digits])
    number_lists.index = np.arange(n)  # row position, reused as-is by explode()

    exploded = number_lists.explode().dropna()
    doc_freq = exploded.value_counts()
    max_df = max_df_ratio * max(n, 1)
    keep_numbers = set(doc_freq[doc_freq <= max_df].index)
    exploded = exploded[exploded.isin(keep_numbers)]

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


# max_df_ratio on the broad single-key rules below is not optional polish - at full
# dataset scale (millions of rows), a low-cardinality key like a common first word
# ("the", "global", ...) forms thousands of individually medium-sized blocks that each
# pass the max_block_pairs cross-product cap yet sum to tens/hundreds of millions of
# low-signal pairs. Even at the previous 0.01 (1%) ratio, every broad rule still
# produced 40-95M raw pairs per target source on the full training set (S1=2.2M,
# S2/S3=5M+ rows each), pushing total deduped candidates to 635M and exhausting 64GB
# of RAM before the pipeline could even finish writing output. 0.002 (0.2%) caps a key
# to ~4-11K matching rows instead of ~22-53K, which is still far more permissive than
# any real business sharing one exact key needs, but cuts the low-signal long tail that
# was dominating candidate volume.
BROAD_KEY_MAX_DF_RATIO = 0.002

# Prefix/token-length knobs below were widened from their original values
# (name prefix 3->6 chars, address prefix 10->16 chars, min_token_len 3->5) after
# max_df_ratio tightening (0.01->0.002) proved to have ZERO effect on raw pair counts
# for every rule, including two that previously had no doc-frequency cap at all -
# meaning the long tail of medium-sized blocks driving 635M candidate pairs wasn't
# coming from a few overly-common keys, it was inherent to how coarse these keys are.
# Widening the key itself spreads rows across a far larger keyspace (e.g. 26^3=17.6K
# possible 3-char prefixes vs 26^6=308M for 6 chars), shrinking block sizes directly
# instead of skipping whole blocks outright, which preserves more true-match recall
# than capping ever could.
#
# country_name_initials and country_name_phonetic (rules 6b/6c) are dropped entirely:
# per their own docstrings they exist to catch rare edge cases (token-reordering,
# transliteration variants) but were jointly responsible for ~115M of the raw pairs
# on Source2 alone, a cost wildly disproportionate to the narrow recall they add at
# this data volume.
DEFAULT_RULES: List[BlockingRule] = [
    BlockingRule("country_name_prefix6", "single_key", key_fn=lambda df: _key_country_name_prefix(df, 6),
                 kwargs={"max_df_ratio": BROAD_KEY_MAX_DF_RATIO}),
    BlockingRule("country_name_firstword", "single_key", key_fn=_key_country_name_firstword,
                 kwargs={"max_df_ratio": BROAD_KEY_MAX_DF_RATIO}),
    BlockingRule("country_address_prefix16", "single_key", key_fn=lambda df: _key_country_address_prefix(df, 16),
                 kwargs={"max_df_ratio": BROAD_KEY_MAX_DF_RATIO}),
    BlockingRule("name_token", "token", column="name_clean",
                 kwargs={"min_token_len": 5, "max_df_ratio": BROAD_KEY_MAX_DF_RATIO}),
    BlockingRule("address_numeric_token", "numeric_token", column="address_clean",
                 kwargs={"min_digits": 3, "max_df_ratio": BROAD_KEY_MAX_DF_RATIO}),
    BlockingRule("country_name_lastword", "single_key", key_fn=_key_country_name_lastword,
                 kwargs={"max_df_ratio": BROAD_KEY_MAX_DF_RATIO}),
]


def build_source_block_indices(df: pd.DataFrame, rules: List[BlockingRule]) -> Dict[str, Dict[str, np.ndarray]]:
    """Materialize every rule's inverted index for one source dataframe."""
    indices: Dict[str, Dict[str, np.ndarray]] = {}
    for rule in rules:
        if rule.kind == "single_key":
            indices[rule.name] = build_block_indices(rule.key_fn(df), **rule.kwargs)
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
    max_block_pairs: int = 50_000,
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
    max_block_pairs: int = 50_000,
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

        left_list: List[np.ndarray] = []
        right_list: List[np.ndarray] = []
        for rule in rules:
            left_pos, right_pos = _pairs_from_matching_blocks(
                s1_indices[rule.name], target_indices[rule.name], max_block_pairs
            )
            if left_pos.size == 0:
                continue
            left_list.append(left_pos)
            right_list.append(right_pos)
            if verbose:
                print(f"  [{source_label}] rule={rule.name}: {len(left_pos):,} raw pairs")

        if not left_list:
            continue

        all_left = np.concatenate(left_list)
        all_right = np.concatenate(right_list)
        del left_list, right_list

        # Ultra-fast integer pair packing: (left << 32) | right
        # Deduplicates in C via np.unique, using only ~1.4GB peak RAM instead of 40GB+
        packed = (all_left.astype(np.uint64) << np.uint64(32)) | all_right.astype(np.uint64)
        del all_left, all_right

        unique_packed = np.unique(packed)
        del packed

        uniq_left = (unique_packed >> np.uint64(32)).astype(np.int64)
        uniq_right = (unique_packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
        del unique_packed

        agg = pd.DataFrame({
            "source1_entity_id": source1["entity_id"].values[uniq_left],
            "candidate_entity_id": target_df["entity_id"].values[uniq_right],
            "source": source_label,
            "rules_matched": "",
        })
        del uniq_left, uniq_right
        all_pair_frames.append(agg)

        if verbose:
            print(f"[{source_label}] {len(agg):,} deduped candidate pairs in {time.time() - t_src:.1f}s")

    if not all_pair_frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source", "rules_matched"])

    result = pd.concat(all_pair_frames, ignore_index=True)
    return result[["source1_entity_id", "candidate_entity_id", "source", "rules_matched"]]


# --------------------------------------------------------------------------- #
# Submission-format grouping (one row per Source1 entity)
#
# The competition requires BOTH candidate_pairs.tsv and matching_results.tsv in this
# shape - source1_entity_id, then a deduplicated/sorted/comma-joined id list, with
# every Source1 entity present exactly once (empty string if it has none). This is
# the same grouping operation for both files (candidate_pairs.tsv groups ALL scored
# candidates; matching_results.tsv groups only the ones that cleared the decision
# threshold), so it lives here once and src.inference reuses it rather than
# reimplementing the same groupby.
# --------------------------------------------------------------------------- #

def group_entity_ids(
    pairs: pd.DataFrame,
    all_entity_ids: Iterable[str],
    id_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    out_col: str = "candidate_entity_ids",
) -> pd.DataFrame:
    """One row per id in `all_entity_ids`, with the matching rows of `pairs` grouped
    into a deduplicated, sorted, comma-joined string in `out_col` (empty string if an
    id has no rows in `pairs` at all)."""
    if len(pairs):
        grouped = (
            pairs.groupby(id_col)[cand_col]
            .apply(lambda ids: ",".join(sorted(set(ids))))
            .rename(out_col)
            .reset_index()
        )
    else:
        grouped = pd.DataFrame(columns=[id_col, out_col])

    all_ids_df = pd.DataFrame({id_col: pd.unique(pd.Series(list(all_entity_ids)))})
    result = all_ids_df.merge(grouped, on=id_col, how="left")
    result[out_col] = result[out_col].fillna("")
    return result.drop_duplicates(subset=id_col).reset_index(drop=True)


def to_submission_format(candidate_pairs: pd.DataFrame, source1: pd.DataFrame) -> pd.DataFrame:
    """
    Convert generate_candidate_pairs()'s long format (one row per pair) into the
    competition's required candidate_pairs.tsv format: one row per Source1 entity,
    `candidate_entity_ids` as every candidate that survived blocking for that entity
    (deduplicated, sorted, comma-joined) - NOT filtered by any model threshold. This
    is "the exact candidate set fed into the model before final thresholding" per the
    problem statement, and matching_results.tsv's matches must be a subset of it.
    """
    return group_entity_ids(
        candidate_pairs, source1["entity_id"].unique(),
        id_col="source1_entity_id", cand_col="candidate_entity_id", out_col="candidate_entity_ids",
    )


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

    # Only the sampled source1 entities can ever produce a hit, so filter candidate_pairs
    # down to just those before building the lookup set. At full dataset scale
    # candidate_pairs can be hundreds of millions of rows - materializing a Python set of
    # every (source1_entity_id, candidate_entity_id) string-tuple in it (the previous
    # approach) took 50+ minutes and tens of GB of RAM for a sample that only ever needs
    # to check a few hundred thousand pairs. Restricting to the sampled entities first
    # keeps the set proportional to sample_size, not to the full candidate table.
    sampled_ids = set(gt["source1_entity_id"])
    relevant_pairs = candidate_pairs[candidate_pairs["source1_entity_id"].isin(sampled_ids)]

    candidate_set = set(zip(relevant_pairs["source1_entity_id"], relevant_pairs["candidate_entity_id"]))
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
    # dtype=str: skips pandas' memory-hungry dtype-inference pass - see
    # src.data_loader.load_tsv's docstring for why this matters on this dataset.
    s1 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source3.tsv"), sep="\t", dtype=str)
    gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str) if os.path.exists(gt_path) else None
    return s1, s2, s3, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 blocking / candidate generation pipeline.")
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                         help="File prefix to load (train_*.tsv or test_*.tsv).")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-block-pairs", type=int, default=50_000)
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
