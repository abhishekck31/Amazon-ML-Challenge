"""
Chunked, multi-process scoring of a full candidate-pair pool.

src.features.generate_features builds every feature for every pair in one
single-threaded pass and holds the whole feature table in memory. That is fine for the
few-million-row training sample, but the full candidate pool at inference / threshold
tuning time is hundreds of millions of pairs: ~3+ hours on one core and far more
memory than a 64GB machine has. score_pairs_chunked produces the same features and
probabilities by:

* grouping pairs by source1 entity and cutting chunks only at entity boundaries, so
  the entity-relative rank features (src.features._add_relative_rank_features) and
  entity-level decision rules (src.threshold_search.apply_dual_threshold) see each
  entity's complete candidate set, exactly as in a single pass;
* computing the per-pair string-similarity features in worker processes, fed plain
  string arrays per chunk (so workers never need the multi-GB source lookup tables);
* predicting per chunk and keeping only what the caller needs (`keep_fn`), so the full
  feature table never exists at once.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features import (
    ID_COLUMNS,
    _add_relative_rank_features,
    compute_pair_features,
    get_feature_matrix,
)

OUTPUT_COLUMNS = ID_COLUMNS + ["match_probability"]


def _codes_and_values(column: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """(int codes per row, unique values) for a categorical or plain column."""
    if isinstance(column.dtype, pd.CategoricalDtype):
        return column.cat.codes.to_numpy(), column.cat.categories.to_numpy(dtype=object)
    codes, uniques = pd.factorize(column)
    return codes, np.asarray(uniques, dtype=object)


def _text_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """name_clean / address_clean / country as object arrays with a trailing "" sentinel,
    so position -1 (id not found in the source) indexes to an empty string - the same
    fallback src.features.generate_features uses for unknown ids."""
    return tuple(
        np.append(df[col].fillna("").to_numpy(dtype=object), "")
        for col in ("name_clean", "address_clean", "country")
    )


def _entity_chunks(s1_codes: np.ndarray, chunk_size: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Row order grouping each source1 entity contiguously, plus (start, end) slices of
    that order of ~chunk_size rows each, cut only at entity boundaries."""
    order = np.argsort(s1_codes, kind="stable")
    sorted_codes = s1_codes[order]
    n = sorted_codes.size
    boundaries = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    del sorted_codes

    chunks = []
    start = 0
    while start < n:
        target = start + chunk_size
        if target >= n:
            end = n
        else:
            idx = np.searchsorted(boundaries, target)
            end = int(boundaries[idx]) if idx < boundaries.size else n
        chunks.append((start, end))
        start = end
    return order, chunks


def score_pairs_chunked(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    keep_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    n_workers: Optional[int] = None,
    chunk_size: int = 1_000_000,
    prefix_len: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score every (source1_entity_id, candidate_entity_id, source) row of
    `candidate_pairs` with `predict_fn`, returning OUTPUT_COLUMNS for the rows
    `keep_fn` keeps (all rows if keep_fn is None). keep_fn receives one chunk at a time,
    each chunk holding complete source1 entities, so per-entity rules give the same
    result as on the full table.

    `source1` / every target must already have name_clean / address_clean / country
    (src.preprocessing.add_clean_columns). The id columns of `candidate_pairs` may be
    plain strings or categorical (src.training_data.load_candidate_pairs).

    n_workers: processes for the per-pair features; defaults to cpu_count - 1. Use 0 to
    compute in-process (tests, small inputs, or platforms without fork).
    """
    t0 = time.time()
    n_pairs = len(candidate_pairs)
    if n_pairs == 0:
        return pd.DataFrame({col: pd.Series(dtype=object) for col in ID_COLUMNS}).assign(
            match_probability=pd.Series(dtype=np.float32)
        )

    s1_codes, s1_values = _codes_and_values(candidate_pairs["source1_entity_id"])
    cand_codes, cand_values = _codes_and_values(candidate_pairs["candidate_entity_id"])
    src_codes, src_values = _codes_and_values(candidate_pairs["source"])

    # Map each distinct id once to its row position in its source table.
    s1_pos_by_code = pd.Index(source1["entity_id"]).get_indexer(s1_values)
    s1_text = _text_columns(source1)

    target_labels = list(targets.keys())
    target_text = [_text_columns(targets[label]) for label in target_labels]
    offsets = np.cumsum([0] + [len(t[0]) for t in target_text[:-1]])
    all_target_text = tuple(np.concatenate([t[i] for t in target_text]) for i in range(3))
    # Position within the concatenated target arrays, per (source label, candidate code).
    cand_pos_by_label: Dict[int, np.ndarray] = {}
    for label, offset, df in zip(target_labels, offsets, targets.values()):
        matches = np.flatnonzero(src_values == label)
        if matches.size == 0:
            continue
        local = pd.Index(df["entity_id"]).get_indexer(cand_values)
        # Unknown ids point at that source's own trailing "" sentinel.
        cand_pos_by_label[int(matches[0])] = np.where(local >= 0, local, len(df)) + offset

    order, chunks = _entity_chunks(s1_codes, chunk_size)
    if verbose:
        print(f"[scoring] {n_pairs:,} pairs in {len(chunks):,} entity-aligned chunks "
              f"(setup {time.time() - t0:.1f}s)", flush=True)

    def build_payload(start: int, end: int):
        rows = order[start:end]
        p1 = s1_pos_by_code[s1_codes[rows]]
        row_src = src_codes[rows]
        row_cand = cand_codes[rows]
        p2 = np.full(rows.size, -1, dtype=np.int64)
        for src_code, pos_by_code in cand_pos_by_label.items():
            mask = row_src == src_code
            p2[mask] = pos_by_code[row_cand[mask]]
        payload = (
            s1_text[0][p1], s1_text[1][p1], s1_text[2][p1],
            all_target_text[0][p2], all_target_text[1][p2], all_target_text[2][p2],
        )
        ids = (s1_values[s1_codes[rows]], cand_values[row_cand], src_values[row_src])
        return payload, ids

    kept: List[pd.DataFrame] = []
    done_pairs = 0
    last_report = t0

    def finish(ids, pair_features):
        nonlocal done_pairs, last_report
        chunk = pd.DataFrame({
            "source1_entity_id": ids[0],
            "candidate_entity_id": ids[1],
            "source": ids[2],
            **pair_features,
        })
        chunk = _add_relative_rank_features(chunk)
        probs = np.asarray(predict_fn(get_feature_matrix(chunk)), dtype=np.float32)
        out = chunk[ID_COLUMNS].assign(match_probability=probs)
        if keep_fn is not None:
            out = keep_fn(out)
        kept.append(out.reset_index(drop=True))

        done_pairs += len(chunk)
        now = time.time()
        if verbose and (now - last_report > 60 or done_pairs == n_pairs):
            rate = done_pairs / (now - t0)
            eta = (n_pairs - done_pairs) / rate if rate else float("nan")
            print(f"[scoring] {done_pairs:,}/{n_pairs:,} pairs "
                  f"({100 * done_pairs / n_pairs:.1f}%), {rate:,.0f} pairs/s, "
                  f"ETA {eta / 60:.1f} min", flush=True)
            last_report = now

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    if n_workers <= 0:
        for start, end in chunks:
            payload, ids = build_payload(start, end)
            finish(ids, compute_pair_features(*payload, prefix_len=prefix_len))
    else:
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as executor:
            # Bounded window of in-flight chunks: submitting everything up front would
            # pickle every chunk's string payload into the call queue at once.
            pending: deque = deque()
            for start, end in chunks:
                payload, ids = build_payload(start, end)
                pending.append((ids, executor.submit(compute_pair_features, *payload, prefix_len=prefix_len)))
                if len(pending) >= 2 * n_workers:
                    ids_done, future = pending.popleft()
                    finish(ids_done, future.result())
            while pending:
                ids_done, future = pending.popleft()
                finish(ids_done, future.result())

    result = pd.concat(kept, ignore_index=True)
    if verbose:
        print(f"[scoring] kept {len(result):,} of {n_pairs:,} pairs in {time.time() - t0:.1f}s", flush=True)
    return result
