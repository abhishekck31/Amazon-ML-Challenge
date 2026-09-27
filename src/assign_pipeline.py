"""
Target-centric entity matching (v2).

In the ground truth every Source 2 / Source 3 record belongs to at most one Source 1
entity (none of the 7.6M matched records has two owners; ~27% have no owner at all).
So instead of scoring Source1 -> candidate pairs independently (v1: src.blocking +
src.inference), this pipeline asks, for every S2/S3 record, "which single S1 entity
owns it, if any?":

1. Retrieval: TF-IDF over the transliterated name + address tokens of every S1 record
   (tokens appearing in more than MAX_DF S1 records dropped), and for each S2/S3 record
   its top-K S1 records by cosine within the same country. On the training set the true
   owner is the top-1 result ~87% of the time and in the top-10 ~93% (v1 blocking
   reached ~70% recall with ~200 candidates per S1 entity).
2. Features per (S2/S3 record, S1 candidate): retrieval signals (cosine, rank, gap to
   the top-1, margin over the next candidate, how many records retrieved / ranked this
   S1 first) plus src.features.compute_pair_features string similarities. Retrieval
   features are defined relative to the S2/S3 record's own K candidates, identically
   in training and inference.
3. LightGBM and CatBoost classifiers score every candidate; their probabilities are
   blended with the weight that scores best on held-out S1 entities.
4. Assignment: each S2/S3 record goes to its single highest-scoring S1 candidate if
   that probability clears a threshold tuned on held-out S1 entities with the
   leaderboard metric (entity-level macro F0.5), else to nobody.

    python -m src.assign_pipeline train   --data-dir dataset/train --models-dir models_v2
    python -m src.assign_pipeline predict --data-dir dataset/test  --models-dir models_v2 --output-dir output
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_tsv
from src.features import FEATURE_COLUMNS, compute_pair_features
from src.preprocessing import add_clean_columns

K = 10
MAX_DF = 5000
K_NAME = 10
STRING_FEATURES = FEATURE_COLUMNS[:20]
RETRIEVAL_FEATURES = ["cos", "rank", "gap_to_top1", "margin_to_next", "n_cands", "s1_top1_count", "s1_cand_count",
                      "s1_reverse_rank", "s1_best_cos_gap",
                      "cos_name", "name_rank", "name_gap_to_top1",
                      "in_name_retrieval", "in_combined_retrieval", "both_channels_retrieved"]
# Assignment picks the best of a target's K candidates, so how a candidate compares to its
# competitors matters as much as its absolute similarity.
RELATIVE_BASE = ["name_jaro_winkler", "token_sort_ratio", "name_bigram_jaccard",
                 "address_token_similarity", "address_partial_similarity", "address_lcs_ratio"]
RELATIVE_FEATURES = [f"{c}_gap_to_best" for c in RELATIVE_BASE]
FEATURES = RETRIEVAL_FEATURES + STRING_FEATURES + RELATIVE_FEATURES
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_split(data_dir: str, prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(source1, targets) with clean columns; targets = S2 and S3 stacked, with a `source` column."""
    s1 = add_clean_columns(load_tsv(os.path.join(data_dir, f"{prefix}_source1.tsv")))
    targets = pd.concat(
        [load_tsv(os.path.join(data_dir, f"{prefix}_source{i}.tsv")).assign(source=f"S{i}") for i in (2, 3)],
        ignore_index=True,
    )
    return s1.reset_index(drop=True), add_clean_columns(targets)


def owner_index(ground_truth: pd.DataFrame, s1: pd.DataFrame, targets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(owner S1 row index per target row, -1 if none; true match count per S1 row)."""
    pos = ground_truth.assign(cid=ground_truth["matched_entity_ids"].fillna("").str.split(",")).explode("cid")
    pos = pos[pos["cid"].fillna("").str.len() > 0]
    s1_pos = pd.Index(s1["entity_id"]).get_indexer(pos["source1_entity_id"])
    tg_pos = pd.Index(targets["entity_id"]).get_indexer(pos["cid"])
    ok = (s1_pos >= 0) & (tg_pos >= 0)
    owner = np.full(len(targets), -1, dtype=np.int64)
    owner[tg_pos[ok]] = s1_pos[ok]
    true_count = np.bincount(s1_pos[s1_pos >= 0], minlength=len(s1))
    return owner, true_count


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def _pair_cosine(T, S, t_rows: np.ndarray, s_rows: np.ndarray, chunk_size: int = 1_000_000) -> np.ndarray:
    """Cosine of explicit (T row, S row) pairs; both matrices are L2-normalized TF-IDF."""
    out = np.empty(t_rows.size, dtype=np.float32)
    for a in range(0, t_rows.size, chunk_size):
        b = min(a + chunk_size, t_rows.size)
        out[a:b] = np.asarray(T[t_rows[a:b]].multiply(S[s_rows[a:b]]).sum(axis=1)).ravel()
    return out


def retrieve(s1: pd.DataFrame, targets: pd.DataFrame, k: int = K, max_df: int = MAX_DF,
             k_name: int = 0, name_max_df: float = 0.01) -> pd.DataFrame:
    """Candidate S1 rows per target row, same country only, from two TF-IDF channels:
    the top-k by word cosine over name + address ("combined"), and, if k_name > 0, the
    top-k_name by character 3-5-gram cosine over the name alone ("name"). Every candidate
    of the union gets both cosines. Returns rows (tgt, s1, cos, cos_name, ...) sorted by
    target then descending combined cosine, plus retrieval features."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    s1_text = (s1["name_clean"] + " " + s1["address_clean"]).to_numpy()
    tg_text = (targets["name_clean"] + " " + targets["address_clean"]).to_numpy()
    s1_name = s1["name_clean"].to_numpy()
    tg_name = targets["name_clean"].to_numpy()
    s1_country = s1["country"].fillna("").to_numpy()
    tg_country = targets["country"].fillna("").to_numpy()
    n_threads = os.cpu_count()

    def local_keys(R, n1):
        return np.repeat(np.arange(R.shape[0], dtype=np.int64), np.diff(R.indptr)) * n1 + R.indices

    parts = []
    for country in sorted(set(s1_country) - {""}):
        i1 = np.flatnonzero(s1_country == country)
        it = np.flatnonzero(tg_country == country)
        if i1.size == 0 or it.size == 0:
            continue
        t0 = time.time()
        vec = TfidfVectorizer(token_pattern=r"\b\w+\b", max_df=max_df, sublinear_tf=True, dtype=np.float32)
        S = vec.fit_transform(s1_text[i1])
        T = vec.transform(tg_text[it])
        R = sp_matmul_topn(T, S.T.tocsr(), top_n=k, sort=True, n_threads=n_threads)
        keys = local_keys(R, i1.size)
        cos = R.data.astype(np.float32)
        cos_name = np.zeros(keys.size, dtype=np.float32)
        in_comb = np.ones(keys.size, dtype=np.float32)
        in_name = np.zeros(keys.size, dtype=np.float32)

        SN = TN = None
        if k_name > 0:
            vec_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=name_max_df,
                                       sublinear_tf=True, dtype=np.float32)
            try:
                SN = vec_name.fit_transform(s1_name[i1])
            except ValueError:  # too few S1 records for the document-frequency limits
                log(f"retrieval {country}: name channel skipped ({i1.size:,} S1 records)")
        if SN is not None:
            TN = vec_name.transform(tg_name[it])
            RN = sp_matmul_topn(TN, SN.T.tocsr(), top_n=k_name, sort=True, n_threads=n_threads)
            keys_name = local_keys(RN, i1.size)
            union = np.union1d(keys, keys_name)
            pos_comb = np.searchsorted(union, keys)
            pos_name = np.searchsorted(union, keys_name)
            u_cos = np.zeros(union.size, dtype=np.float32)
            u_cos_name = np.zeros(union.size, dtype=np.float32)
            in_comb = np.zeros(union.size, dtype=np.float32)
            in_name = np.zeros(union.size, dtype=np.float32)
            u_cos[pos_comb], in_comb[pos_comb] = cos, 1
            u_cos_name[pos_name], in_name[pos_name] = RN.data, 1
            # Each channel's cosine for the candidates only the other channel retrieved.
            t_loc, s_loc = union // i1.size, union % i1.size
            miss = in_comb == 0
            u_cos[miss] = _pair_cosine(T, S, t_loc[miss], s_loc[miss])
            miss = in_name == 0
            u_cos_name[miss] = _pair_cosine(TN, SN, t_loc[miss], s_loc[miss])
            keys, cos, cos_name = union, u_cos, u_cos_name
        parts.append(pd.DataFrame({
            "tgt": it[keys // i1.size].astype(np.int32),
            "s1": i1[keys % i1.size].astype(np.int32),
            "cos": cos, "cos_name": cos_name,
            "in_combined_retrieval": in_comb, "in_name_retrieval": in_name,
        }))
        log(f"retrieval {country}: {i1.size:,} S1 x {it.size:,} targets -> {keys.size:,} pairs "
            f"({int(in_name.sum()):,} from the name channel, {int((in_name * in_comb).sum()):,} from both; "
            f"{time.time() - t0:.0f}s)")
        del S, T, SN, TN

    cands = pd.concat(parts, ignore_index=True)
    del parts
    order = np.lexsort((-cands["cos"].to_numpy(), cands["tgt"].to_numpy()))
    cands = cands.iloc[order].reset_index(drop=True)

    tgt = cands["tgt"].to_numpy()
    cos = cands["cos"].to_numpy()
    s1_idx = cands["s1"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tgt[1:] != tgt[:-1]])
    lengths = np.diff(np.r_[starts, tgt.size])
    same_next = np.r_[tgt[1:] == tgt[:-1], False]
    cands["rank"] = (np.arange(tgt.size) - np.repeat(starts, lengths)).astype(np.float32)
    cands["gap_to_top1"] = np.repeat(cos[starts], lengths) - cos
    cands["margin_to_next"] = cos - np.where(same_next, np.r_[cos[1:], 0], 0).astype(np.float32)
    cands["n_cands"] = np.repeat(lengths, lengths).astype(np.float32)
    cands["s1_top1_count"] = np.bincount(s1_idx[starts], minlength=len(s1))[s1_idx].astype(np.float32)
    cands["s1_cand_count"] = np.bincount(s1_idx, minlength=len(s1))[s1_idx].astype(np.float32)

    # The S1 side: where this target ranks among every target that retrieved the same S1.
    by_s1 = np.lexsort((-cos, s1_idx))
    s1_sorted = s1_idx[by_s1]
    s1_starts = np.flatnonzero(np.r_[True, s1_sorted[1:] != s1_sorted[:-1]])
    s1_lengths = np.diff(np.r_[s1_starts, s1_sorted.size])
    reverse_rank = np.empty(tgt.size, dtype=np.float32)
    reverse_rank[by_s1] = np.arange(tgt.size) - np.repeat(s1_starts, s1_lengths)
    best_cos = np.empty(tgt.size, dtype=np.float32)
    best_cos[by_s1] = np.repeat(cos[by_s1][s1_starts], s1_lengths)
    cands["s1_reverse_rank"] = reverse_rank
    cands["s1_best_cos_gap"] = best_cos - cos

    # The same target-relative view of the name channel.
    cos_name = cands["cos_name"].to_numpy()
    by_name = np.lexsort((-cos_name, tgt))
    name_rank = np.empty(tgt.size, dtype=np.float32)
    name_rank[by_name] = np.arange(tgt.size) - np.repeat(starts, lengths)
    cands["name_rank"] = name_rank
    cands["name_gap_to_top1"] = np.repeat(np.maximum.reduceat(cos_name, starts), lengths) - cos_name
    cands["both_channels_retrieved"] = cands["in_combined_retrieval"] * cands["in_name_retrieval"]
    return cands


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def feature_matrix(cands: pd.DataFrame, s1: pd.DataFrame, targets: pd.DataFrame,
                   n_workers: int = None, chunk_size: int = 1_000_000) -> np.ndarray:
    """(len(cands), len(FEATURES)) float32 matrix; string features computed in worker processes."""
    n = len(cands)
    X = np.empty((n, len(FEATURES)), dtype=np.float32)
    for j, col in enumerate(RETRIEVAL_FEATURES):
        X[:, j] = cands[col].to_numpy()

    s1_cols = [s1[c].fillna("").to_numpy(dtype=object) for c in ("name_clean", "address_clean", "country")]
    tg_cols = [targets[c].fillna("").to_numpy(dtype=object) for c in ("name_clean", "address_clean", "country")]
    s1_idx = cands["s1"].to_numpy()
    tg_idx = cands["tgt"].to_numpy()
    offset = len(RETRIEVAL_FEATURES)

    def payload(a, b):
        i, t = s1_idx[a:b], tg_idx[a:b]
        return [col[i] for col in s1_cols] + [col[t] for col in tg_cols]

    def store(a, b, result):
        for j, col in enumerate(STRING_FEATURES):
            X[a:b, offset + j] = result[col]

    chunks = [(a, min(a + chunk_size, n)) for a in range(0, n, chunk_size)]
    n_workers = n_workers if n_workers is not None else max(1, (os.cpu_count() or 2) - 1)
    t0, done, last = time.time(), 0, time.time()
    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as ex:
        pending: deque = deque()
        for a, b in chunks:
            pending.append((a, b, ex.submit(compute_pair_features, *payload(a, b))))
            while len(pending) >= 2 * n_workers or (pending and pending[0][2].done()):
                pa, pb, fut = pending.popleft()
                store(pa, pb, fut.result())
                done += pb - pa
                if time.time() - last > 60:
                    rate = done / (time.time() - t0)
                    log(f"features {done:,}/{n:,} ({100 * done / n:.0f}%), {rate:,.0f}/s, "
                        f"ETA {(n - done) / rate / 60:.1f} min")
                    last = time.time()
        while pending:
            pa, pb, fut = pending.popleft()
            store(pa, pb, fut.result())

    # Gap to the best value among the same target's candidates (cands is sorted by target).
    tgt = tg_idx
    starts = np.flatnonzero(np.r_[True, tgt[1:] != tgt[:-1]])
    lengths = np.diff(np.r_[starts, n])
    for j, base in enumerate(RELATIVE_BASE):
        col = X[:, FEATURES.index(base)]
        best = np.repeat(np.maximum.reduceat(col, starts), lengths)
        X[:, FEATURES.index(RELATIVE_FEATURES[j])] = col - best
    log(f"features done: {n:,} rows in {time.time() - t0:.0f}s")
    return X


def score_rows(cands: pd.DataFrame, rows: np.ndarray, s1: pd.DataFrame, targets: pd.DataFrame, predict_fns,
               feature_idx=None, chunk_rows: int = 15_000_000):
    """Each predict_fn's probabilities for the cands rows `rows` (ascending, whole targets
    only). Features are built in target-aligned chunks, so only one chunk's matrix is in
    memory at a time. feature_idx selects the model's columns from FEATURES."""
    tgt = cands["tgt"].to_numpy()[rows]
    starts = np.flatnonzero(np.r_[True, tgt[1:] != tgt[:-1]])
    out = [np.empty(rows.size, dtype=np.float32) for _ in predict_fns]
    a = 0
    while a < rows.size:
        nxt = np.searchsorted(starts, a + chunk_rows)
        b = starts[nxt] if nxt < starts.size else rows.size
        X = feature_matrix(cands.iloc[rows[a:b]], s1, targets)
        if feature_idx is not None:
            X = X[:, feature_idx]
        for o, fn in zip(out, predict_fns):
            o[a:b] = predict_chunked(fn, X)
        del X
        log(f"scored {b:,}/{rows.size:,} pairs")
        a = b
    return out


# --------------------------------------------------------------------------- #
# Assignment + metric
# --------------------------------------------------------------------------- #

def predict_chunked(predict_fn, X: np.ndarray, chunk_size: int = 5_000_000) -> np.ndarray:
    """float32 probabilities, predicted in row chunks to bound the models' temporary copies."""
    return np.concatenate([np.asarray(predict_fn(X[a:a + chunk_size]), dtype=np.float32)
                           for a in range(0, len(X), chunk_size)])


def assign(cands: pd.DataFrame, prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best S1 per target: (target rows, S1 rows, probability)."""
    tgt = cands["tgt"].to_numpy()
    order = np.lexsort((-prob, tgt))
    tgt_sorted = tgt[order]
    first = order[np.r_[True, tgt_sorted[1:] != tgt_sorted[:-1]]]
    return tgt[first], cands["s1"].to_numpy()[first], prob[first]


def entity_macro_f05(best_tgt, best_s1, best_prob, owner, true_count, entity_rows, threshold) -> float:
    """Leaderboard metric over the S1 rows in `entity_rows`, for the assignment at `threshold`."""
    keep = best_prob >= threshold
    s1a = best_s1[keep]
    correct = owner[best_tgt[keep]] == s1a
    n = true_count.size
    pred = np.bincount(s1a, minlength=n)[entity_rows].astype(np.float64)
    tp = np.bincount(s1a[correct], minlength=n)[entity_rows].astype(np.float64)
    truth = true_count[entity_rows].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred > 0, tp / pred, 0.0)
        recall = np.where(truth > 0, tp / truth, 0.0)
        f = np.where(tp > 0, 1.25 * precision * recall / (0.25 * precision + recall), 0.0)
    f = np.where(truth == 0, (pred == 0).astype(np.float64), f)
    return float(f.mean())


def grouped_ids(s1_rows: np.ndarray, tgt_rows: np.ndarray, s1: pd.DataFrame, targets: pd.DataFrame,
                out_col: str) -> pd.DataFrame:
    """One row per S1 entity (all of them, source order) with its targets comma-joined in
    sorted (string) order, as validate_submission.py requires."""
    ids = targets["entity_id"].to_numpy()
    string_rank = np.empty(ids.size, dtype=np.int64)
    string_rank[np.argsort(ids, kind="stable")] = np.arange(ids.size)
    order = np.lexsort((string_rank[tgt_rows], s1_rows))
    df = pd.DataFrame({"s": s1_rows[order], "t": ids[tgt_rows[order]]})
    joined = df.groupby("s", sort=False)["t"].agg(",".join)
    col = np.full(len(s1), "", dtype=object)
    col[joined.index.to_numpy()] = joined.to_numpy()
    return pd.DataFrame({"source1_entity_id": s1["entity_id"].to_numpy(), out_col: col})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def run_train(args) -> None:
    import lightgbm as lgb

    t_start = time.time()
    s1, targets = load_split(args.data_dir, "train")
    ground_truth = load_tsv(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    owner, true_count = owner_index(ground_truth, s1, targets)
    log(f"loaded {len(s1):,} S1 / {len(targets):,} targets")

    cands = retrieve(s1, targets, k=args.k, max_df=args.max_df, k_name=args.k_name, name_max_df=args.name_max_df)
    tgt, s1_idx = cands["tgt"].to_numpy(), cands["s1"].to_numpy()
    label = (owner[tgt] == s1_idx).astype(np.int8)
    matched = owner >= 0
    retrieved = np.zeros(len(targets), dtype=bool)
    retrieved[tgt[label == 1]] = True
    log(f"{len(cands):,} candidate pairs ({len(cands) / len(targets):.1f} per target); "
        f"owner retrieved for {retrieved[matched].mean():.4f} of matched targets")
    if args.k_name > 0:
        pos = label == 1
        for col in ("in_combined_retrieval", "in_name_retrieval", "both_channels_retrieved"):
            hit = np.zeros(len(targets), dtype=bool)
            hit[tgt[pos & (cands[col].to_numpy() == 1)]] = True
            log(f"  {col}: owner found for {hit[matched].mean():.4f} of matched targets")

    rng = np.random.default_rng(args.seed)
    val_entity = rng.random(len(s1)) < args.val_frac
    target_sample = rng.random(len(targets)) < args.train_target_frac
    # Features for the sampled targets only (with all their candidates, so target-relative
    # features are exact); everything else is scored in chunks after training.
    fit_rows = np.flatnonzero(target_sample[tgt])
    X = feature_matrix(cands.iloc[fit_rows], s1, targets)
    fit_label = label[fit_rows]
    fit_val = val_entity[s1_idx[fit_rows]]
    train_rows = np.flatnonzero(~fit_val)
    es_rows = np.flatnonzero(fit_val & (rng.random(fit_rows.size) < 0.2))
    log(f"training on {train_rows.size:,} pairs ({fit_label[train_rows].mean():.3f} positive), "
        f"early stopping on {es_rows.size:,}")

    params = {
        "objective": "binary", "learning_rate": args.learning_rate, "num_leaves": 255,
        "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
        "verbosity": -1, "num_threads": os.cpu_count(), "seed": args.seed,
    }
    dtrain = lgb.Dataset(X[train_rows], label=fit_label[train_rows], feature_name=FEATURES, free_raw_data=True)
    des = lgb.Dataset(X[es_rows], label=fit_label[es_rows], reference=dtrain)
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=args.num_boost_round, valid_sets=[des],
                        valid_names=["early_stop"],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])
    log(f"trained {booster.best_iteration} trees in {time.time() - t0:.0f}s")
    del dtrain, des
    predict_fns = [lambda x: booster.predict(x, num_iteration=booster.best_iteration)]

    cat = None
    if args.cat_iterations > 0:
        from catboost import CatBoostClassifier

        cat = CatBoostClassifier(iterations=args.cat_iterations, learning_rate=args.cat_learning_rate, depth=8,
                                 thread_count=os.cpu_count(), random_seed=args.seed, od_type="Iter", od_wait=30,
                                 verbose=100)
        t0 = time.time()
        cat.fit(X[train_rows], fit_label[train_rows], eval_set=(X[es_rows], fit_label[es_rows]),
                use_best_model=True)
        log(f"trained CatBoost ({cat.get_best_iteration()} trees) in {time.time() - t0:.0f}s")
        predict_fns.append(lambda x: cat.predict_proba(x)[:, 1])

    t0 = time.time()
    probs = [np.empty(len(cands), dtype=np.float32) for _ in predict_fns]
    for p, fn in zip(probs, predict_fns):
        p[fit_rows] = predict_chunked(fn, X)
    del X
    rest = np.flatnonzero(~target_sample[tgt])
    for p, q in zip(probs, score_rows(cands, rest, s1, targets, predict_fns)):
        p[rest] = q
    prob_lgbm = probs[0]
    prob_cat = probs[1] if cat is not None else None
    log(f"predicted {len(cands):,} pairs in {time.time() - t0:.0f}s")

    val_rows = np.flatnonzero(val_entity)

    def tune(prob):
        best_tgt, best_s1, best_prob = assign(cands, prob)
        curve = [(t, entity_macro_f05(best_tgt, best_s1, best_prob, owner, true_count, val_rows, t))
                 for t in THRESHOLD_GRID]
        return max(curve, key=lambda ts: ts[1]) + (curve,)

    # Weight on LightGBM; 1.0 = LightGBM alone, 0.0 = CatBoost alone.
    weights = [1.0] if cat is None else [1.0, 0.75, 0.5, 0.25, 0.0]
    results = {}
    for w in weights:
        prob = prob_lgbm if cat is None else (w * prob_lgbm + (1 - w) * prob_cat).astype(np.float32)
        results[w] = tune(prob)
        log(f"lgbm weight {w:.2f}: validation entity macro F0.5 = {results[w][1]:.4f} at threshold {results[w][0]:.2f}")
    lgbm_weight = max(results, key=lambda w: results[w][1])
    best_t, best_score, scores = results[lgbm_weight]
    # A perfect classifier assigns exactly the targets whose owner was retrieved.
    hit = np.flatnonzero(label == 1)
    ceiling = entity_macro_f05(cands["tgt"].to_numpy()[hit], cands["s1"].to_numpy()[hit],
                               np.ones(hit.size, dtype=np.float32), owner, true_count, val_rows, 0.5)
    log(f"chose lgbm weight {lgbm_weight:.2f}: validation entity macro F0.5 = {best_score:.4f} at threshold "
        f"{best_t:.2f} (perfect-classifier ceiling with this retrieval: {ceiling:.4f})")

    # Retrieval never crosses countries, so each country's S1 entities depend only on that
    # country's assignments and a per-country threshold can be tuned independently.
    prob = prob_lgbm if cat is None else (lgbm_weight * prob_lgbm + (1 - lgbm_weight) * prob_cat).astype(np.float32)
    best_tgt, best_s1, best_prob = assign(cands, prob)
    s1_country = s1["country"].fillna("").to_numpy()
    country_thresholds, country_scores = {}, {}
    for country in sorted(set(s1_country[val_rows]) - {""}):
        rows = val_rows[s1_country[val_rows] == country]
        curve = [(t, entity_macro_f05(best_tgt, best_s1, best_prob, owner, true_count, rows, t))
                 for t in THRESHOLD_GRID]
        t_c, s_c = max(curve, key=lambda ts: ts[1])
        at_global = entity_macro_f05(best_tgt, best_s1, best_prob, owner, true_count, rows, best_t)
        country_thresholds[country], country_scores[country] = float(t_c), s_c
        log(f"{country}: {rows.size:,} validation entities, F0.5 {at_global:.4f} at the global threshold, "
            f"{s_c:.4f} at its own threshold {t_c:.2f}")

    os.makedirs(args.models_dir, exist_ok=True)
    booster.save_model(str(Path(args.models_dir) / "assign_lgbm.txt"), num_iteration=booster.best_iteration)
    if cat is not None:
        cat.save_model(str(Path(args.models_dir) / "assign_catboost.cbm"))
    info = {
        "threshold": float(best_t), "validation_entity_macro_f0.5": best_score,
        "retrieval_ceiling_f0.5": ceiling, "k": args.k, "max_df": args.max_df, "k_name": args.k_name,
        "name_max_df": args.name_max_df,
        "features": FEATURES, "n_trees": booster.best_iteration,
        "lgbm_weight": float(lgbm_weight),
        "country_thresholds": country_thresholds,
        "country_validation_f0.5": {c: round(s, 5) for c, s in country_scores.items()},
        "blend_scores": {f"{w:.2f}": round(r[1], 5) for w, r in results.items()},
        "threshold_curve": {f"{t:.2f}": round(s, 5) for t, s in scores},
    }
    (Path(args.models_dir) / "assign_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    importance = pd.Series(booster.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    log("feature importance (gain):\n" + (importance / importance.sum()).round(4).to_string())
    log(f"saved model to {args.models_dir} ({time.time() - t_start:.0f}s total)")


def run_predict(args) -> None:
    import lightgbm as lgb

    t_start = time.time()
    info = json.loads((Path(args.models_dir) / "assign_info.json").read_text(encoding="utf-8"))
    threshold = args.threshold if args.threshold is not None else info["threshold"]
    booster = lgb.Booster(model_file=str(Path(args.models_dir) / "assign_lgbm.txt"))
    log(f"model with {booster.num_trees()} trees, threshold {threshold:.2f}")

    s1, targets = load_split(args.data_dir, args.split)
    log(f"loaded {len(s1):,} S1 / {len(targets):,} targets")
    cands = retrieve(s1, targets, k=info["k"], max_df=info["max_df"], k_name=info.get("k_name", 0),
                     name_max_df=info.get("name_max_df", 0.01))
    # A model trained with an earlier feature set uses its own columns.
    feature_idx = None if info["features"] == FEATURES else [FEATURES.index(f) for f in info["features"]]
    t0 = time.time()
    lgbm_weight = info.get("lgbm_weight", 1.0)
    predict_fns, weights = [], []
    if lgbm_weight > 0:
        predict_fns.append(booster.predict)
        weights.append(lgbm_weight)
    if lgbm_weight < 1:
        from catboost import CatBoostClassifier

        cat = CatBoostClassifier().load_model(str(Path(args.models_dir) / "assign_catboost.cbm"))
        predict_fns.append(lambda x: cat.predict_proba(x)[:, 1])
        weights.append(1 - lgbm_weight)
    probs = score_rows(cands, np.arange(len(cands)), s1, targets, predict_fns, feature_idx=feature_idx)
    prob = sum(w * p for w, p in zip(weights, probs)).astype(np.float32)
    del probs
    log(f"predicted {len(prob):,} pairs (lgbm weight {lgbm_weight:.2f}) in {time.time() - t0:.0f}s")

    best_tgt, best_s1, best_prob = assign(cands, prob)
    # Per-country thresholds unless overridden; countries unseen in training use the global one.
    country_thresholds = {} if args.threshold is not None else info.get("country_thresholds", {})
    s1_country = s1["country"].fillna("").to_numpy()
    pair_threshold = np.full(best_s1.size, threshold, dtype=np.float32)
    for country, t in country_thresholds.items():
        pair_threshold[s1_country[best_s1] == country] = t
    log("thresholds: " + ", ".join(f"{c} {t:.2f}" for c, t in country_thresholds.items())
        + f", other {threshold:.2f}")
    keep = best_prob >= pair_threshold
    matching = grouped_ids(best_s1[keep], best_tgt[keep], s1, targets, "matched_entity_ids")
    candidates = grouped_ids(cands["s1"].to_numpy(), cands["tgt"].to_numpy(), s1, targets, "candidate_entity_ids")

    os.makedirs(args.output_dir, exist_ok=True)
    candidates.to_csv(Path(args.output_dir) / "candidate_pairs.tsv", sep="\t", index=False)
    matching.to_csv(Path(args.output_dir) / "matching_results.tsv", sep="\t", index=False)
    n_matched = int((matching["matched_entity_ids"] != "").sum())
    log(f"assigned {int(keep.sum()):,} of {len(targets):,} targets; {n_matched:,}/{len(s1):,} S1 entities "
        f"have >=1 match; wrote {args.output_dir}/matching_results.tsv and candidate_pairs.tsv "
        f"({time.time() - t_start:.0f}s total)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-centric entity matching (v2).")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-dir", default="dataset/train")
    train.add_argument("--models-dir", default="models_v2")
    train.add_argument("--k", type=int, default=K)
    train.add_argument("--max-df", type=int, default=MAX_DF)
    train.add_argument("--k-name", type=int, default=K_NAME, help="Name-channel candidates; 0 disables it.")
    train.add_argument("--name-max-df", type=float, default=0.01)
    train.add_argument("--val-frac", type=float, default=0.2)
    train.add_argument("--train-target-frac", type=float, default=0.4)
    train.add_argument("--learning-rate", type=float, default=0.1)
    train.add_argument("--num-boost-round", type=int, default=600)
    train.add_argument("--cat-iterations", type=int, default=1500, help="0 disables the CatBoost blend.")
    train.add_argument("--cat-learning-rate", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=42)

    predict = sub.add_parser("predict")
    predict.add_argument("--data-dir", default="dataset/test")
    predict.add_argument("--split", default="test", choices=["train", "test"])
    predict.add_argument("--models-dir", default="models_v2")
    predict.add_argument("--output-dir", default="output")
    predict.add_argument("--threshold", type=float, default=None, help="Overrides assign_info.json.")

    args = parser.parse_args()
    run_train(args) if args.command == "train" else run_predict(args)


if __name__ == "__main__":
    main()
