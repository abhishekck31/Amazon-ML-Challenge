"""
Phase 7 - Decision threshold optimization for Amazon ML Challenge 2026: Business Entity
Resolution.

A trained classifier (Phase 6) outputs a match probability; turning that into a
match/no-match decision requires picking a threshold.

Two searches live here, and they are NOT interchangeable:

* search_thresholds() / find_best_threshold() sweep a SINGLE global threshold and
  score it with sklearn's row-level macro F0.5 (fbeta_score(average="macro") over
  flattened pairs). This is a fast, cheap diagnostic useful for comparing candidate
  models during Phase 6 - it is NOT the competition's scoring function.

* search_dual_thresholds() sweeps the (T_singleton, T_match) pair used in
  src.inference's actual decision rule and scores each combination with
  src.metrics.entity_level_macro_f05 - the REAL leaderboard metric, computed the way
  the competition computes it (per Source1 entity, then averaged). Use this one to
  pick the thresholds that actually ship in a submission.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.features import get_feature_matrix
from src.metrics import entity_f0_5

DEFAULT_LOW = 0.30
DEFAULT_HIGH = 0.99
DEFAULT_N_STEPS = 140


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #

def search_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    n_steps: int = DEFAULT_N_STEPS,
) -> pd.DataFrame:
    """
    Evaluate macro F0.5 / precision / recall at `n_steps` thresholds evenly spaced in
    [low, high]. Returns one row per threshold - the full sweep, not just the winner,
    so it can be plotted or audited.
    """
    thresholds = np.linspace(low, high, n_steps)
    rows = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "macro_f0.5": fbeta_score(y_true, preds, beta=0.5, average="macro", zero_division=0),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
        })
    return pd.DataFrame(rows)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    n_steps: int = DEFAULT_N_STEPS,
) -> Tuple[float, float, pd.DataFrame]:
    """Return (best_threshold, best_macro_f0.5, full_sweep_table) maximizing macro F0.5
    over [low, high]."""
    table = search_thresholds(y_true, y_prob, low, high, n_steps)
    best_idx = table["macro_f0.5"].idxmax()
    best_row = table.loc[best_idx]
    return float(best_row["threshold"]), float(best_row["macro_f0.5"]), table


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path) -> None:
    """Standard precision-recall curve traced over every threshold sklearn finds in the
    data (not just the [low, high] search window) - the full picture the threshold
    search zooms into."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_threshold(table: pd.DataFrame, output_path) -> None:
    """Macro F0.5 / precision / recall as a function of threshold across the searched
    range, with the best threshold marked - makes the optimum (and the precision/recall
    trade-off around it) visible at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_idx = table["macro_f0.5"].idxmax()
    best_threshold = table.loc[best_idx, "threshold"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["threshold"], table["macro_f0.5"], label="macro F0.5")
    ax.plot(table["threshold"], table["precision"], label="precision", linestyle="--")
    ax.plot(table["threshold"], table["recall"], label="recall", linestyle="--")
    ax.axvline(best_threshold, color="black", linestyle=":", label=f"best={best_threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold search (macro F0.5)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Model loading (any of the four Phase 6 formats) - CLI convenience
# --------------------------------------------------------------------------- #

def load_predict_proba_fn(model_path: str, model_name: str) -> Callable[[pd.DataFrame], np.ndarray]:
    """Load a Phase 6 model file (by its saved name/extension) and return a
    `predict_proba(X) -> P(label=1)` callable. Handles src.train.build_ensemble's
    output (model_name == "ensemble"): reads the JSON manifest and recursively loads +
    weight-averages each component model's predict_proba_fn."""
    if model_name == "ensemble":
        manifest = json.loads(Path(model_path).read_text(encoding="utf-8"))
        component_fns = [load_predict_proba_fn(c["path"], c["name"]) for c in manifest["components"]]
        weights = np.array([c.get("weight", 1.0) for c in manifest["components"]], dtype=float)
        weights = weights / weights.sum()

        def predict_fn(X):
            probs = np.stack([fn(X) for fn in component_fns], axis=0)
            return np.average(probs, axis=0, weights=weights)

        return predict_fn

    ext = Path(model_path).suffix
    if model_name == "logistic_regression" or ext == ".joblib":
        import joblib
        model = joblib.load(model_path)
        return lambda X: model.predict_proba(X)[:, 1]
    if model_name == "lightgbm" or ext == ".txt":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=model_path)
        return lambda X: booster.predict(X)
    if model_name == "xgboost" or ext == ".json":
        import xgboost as xgb
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        return lambda X: clf.predict_proba(X)[:, 1]
    if model_name == "catboost" or ext == ".cbm":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(model_path)
        return lambda X: model.predict_proba(X)[:, 1]
    raise ValueError(f"Unrecognized model type for path={model_path!r}, name={model_name!r}")


# --------------------------------------------------------------------------- #
# Dual-threshold search - optimizes the REAL entity-level leaderboard metric
#
# Decision rule (matches src.inference.apply_dual_threshold exactly):
#   * if an entity's best candidate probability < T_singleton: predict no matches
#     at all (protects the 1.0 score a true singleton gets for an empty prediction).
#   * otherwise: accept every candidate scoring >= T_match.
# --------------------------------------------------------------------------- #

DEFAULT_SINGLETON_GRID = np.round(np.linspace(0.30, 0.95, 14), 3)
DEFAULT_MATCH_GRID = np.round(np.linspace(0.30, 0.95, 14), 3)


def _precompute_sorted_probs(scored_pairs: pd.DataFrame):
    """{entity_id: (probs_desc, candidate_ids_same_order, neg_probs_ascending)}, built
    once so search_dual_thresholds can binary-search each entity's candidates at every
    grid point instead of re-scanning/re-grouping the full pairs table per threshold
    combination (which would be O(n_combinations * n_pairs) instead of O(n_pairs log
    n_pairs) + O(n_combinations * n_entities * log k))."""
    per_entity: Dict[str, List[Tuple[float, str]]] = {}
    for sid, cid, prob in zip(
        scored_pairs["source1_entity_id"], scored_pairs["candidate_entity_id"], scored_pairs["match_probability"]
    ):
        per_entity.setdefault(sid, []).append((prob, cid))

    result = {}
    for sid, items in per_entity.items():
        items.sort(key=lambda pc: -pc[0])
        probs = [p for p, _ in items]
        cids = [c for _, c in items]
        neg_probs_asc = [-p for p in probs]  # ascending, since probs is descending
        result[sid] = (probs, cids, neg_probs_asc)
    return result


def apply_dual_threshold(
    scored_pairs: pd.DataFrame, singleton_threshold: float, match_threshold: float
) -> pd.DataFrame:
    """Apply the dual-threshold decision rule to a scored candidate-pairs DataFrame
    (must have a `match_probability` column) and return only the accepted rows."""
    max_prob_per_entity = scored_pairs.groupby("source1_entity_id")["match_probability"].transform("max")
    clears_singleton_gate = max_prob_per_entity >= singleton_threshold
    return scored_pairs.loc[clears_singleton_gate & (scored_pairs["match_probability"] >= match_threshold)]


def search_dual_thresholds(
    scored_pairs: pd.DataFrame,
    ground_truth_map: Dict[str, Set[str]],
    entity_ids: Iterable[str],
    singleton_grid: Optional[np.ndarray] = None,
    match_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, pd.DataFrame]:
    """
    Grid search (T_singleton, T_match) maximizing entity_level_macro_f05 (src.metrics)
    directly - the actual leaderboard scoring, not a proxy. `scored_pairs` needs
    columns source1_entity_id, candidate_entity_id, match_probability, and should
    cover the FULL candidate pool for `entity_ids` (i.e. output of
    src.blocking.generate_candidate_pairs scored by the model), not a class-balanced
    training sample - the real precision/recall trade-off only shows up against the
    true, heavily-imbalanced candidate distribution.

    Returns (best_singleton_threshold, best_match_threshold, best_score, full_grid_table).
    """
    singleton_grid = DEFAULT_SINGLETON_GRID if singleton_grid is None else singleton_grid
    match_grid = DEFAULT_MATCH_GRID if match_grid is None else match_grid

    entity_ids = list(entity_ids)
    per_entity = _precompute_sorted_probs(scored_pairs)
    true_sets = [ground_truth_map.get(eid, set()) for eid in entity_ids]
    entity_lists = [per_entity.get(eid, ([], [], [])) for eid in entity_ids]

    rows = []
    best_score, best_ts, best_tm = -1.0, float(singleton_grid[0]), float(match_grid[0])
    for t_s in singleton_grid:
        for t_m in match_grid:
            if t_m < t_s:
                continue  # match_threshold must be at least as strict as the singleton gate
            scores = []
            for true_set, (probs, cids, neg_probs_asc) in zip(true_sets, entity_lists):
                if not probs or probs[0] < t_s:
                    pred_set: Set[str] = set()
                else:
                    cutoff = bisect.bisect_left(neg_probs_asc, -t_m)
                    pred_set = set(cids[:cutoff])
                scores.append(entity_f0_5(true_set, pred_set))
            score = float(np.mean(scores)) if scores else 0.0
            rows.append({"singleton_threshold": float(t_s), "match_threshold": float(t_m), "entity_macro_f0.5": score})
            if score > best_score:
                best_score, best_ts, best_tm = score, float(t_s), float(t_m)

    return best_ts, best_tm, best_score, pd.DataFrame(rows)


def tune_dual_thresholds_from_full_pool(
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    model_path: str,
    model_name: str,
    max_block_pairs: int = 200_000,
    singleton_grid: Optional[np.ndarray] = None,
    match_grid: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[float, float, float, pd.DataFrame]:
    """
    End-to-end dual-threshold tuning against the REAL candidate pool (not a
    class-balanced training sample): preprocess -> generate candidates -> score with
    the given model -> search_dual_thresholds() against `ground_truth`.

    Run this on entities the model was NOT trained on (e.g. a held-out slice of
    Source1, or `train_pairs.parquet`'s complement) - tuning and evaluating on the
    same rows the model fit would overstate the score.
    """
    from src.blocking import generate_candidate_pairs
    from src.features import generate_features
    from src.metrics import ground_truth_to_map
    from src.preprocessing import add_clean_columns

    source1 = add_clean_columns(source1)
    targets = {label: add_clean_columns(df) for label, df in targets.items()}
    candidate_pairs = generate_candidate_pairs(source1, targets, max_block_pairs=max_block_pairs, verbose=verbose)
    features = generate_features(candidate_pairs, source1, targets, verbose=verbose)

    predict_fn = load_predict_proba_fn(model_path, model_name)
    features["match_probability"] = predict_fn(get_feature_matrix(features))

    gt_map = ground_truth_to_map(ground_truth)
    entity_ids = source1["entity_id"].unique()
    return search_dual_thresholds(features, gt_map, entity_ids, singleton_grid, match_grid)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _run_single_mode(args) -> None:
    """Diagnostic single-threshold search on a class-balanced Phase 5 val set,
    scored with sklearn's row-level macro F0.5 (NOT the leaderboard metric)."""
    if args.model_path is None:
        info_path = Path(args.models_dir) / "best_model_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        model_path, model_name = info["path"], info["name"]
    else:
        model_path = args.model_path
        model_name = args.model_name or Path(model_path).stem

    val_pairs = pd.read_parquet(args.val_pairs)
    X_val = get_feature_matrix(val_pairs)
    y_val = val_pairs["label"].to_numpy()

    predict_fn = load_predict_proba_fn(model_path, model_name)
    y_prob = predict_fn(X_val)

    best_threshold, best_score, table = find_best_threshold(
        y_val, y_prob, low=args.low, high=args.high, n_steps=args.n_steps
    )
    best_row = table.loc[table["threshold"] == best_threshold].iloc[0]

    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(Path(args.output_dir) / "threshold_search.csv", index=False)
    plot_precision_recall_curve(y_val, y_prob, Path(args.output_dir) / "pr_curve.png")
    plot_metric_vs_threshold(table, Path(args.output_dir) / "threshold_vs_metrics.png")

    result = {
        "model_name": model_name,
        "model_path": str(model_path),
        "search_range": [args.low, args.high],
        "best_threshold": best_threshold,
        "best_macro_f0.5": best_score,
        "precision_at_best": float(best_row["precision"]),
        "recall_at_best": float(best_row["recall"]),
    }
    (Path(args.output_dir) / "best_threshold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _run_dual_mode(args) -> None:
    """Real leaderboard-metric dual-threshold search against the full candidate pool
    for a data split, writing models/dual_threshold_info.json for src.inference to use."""
    from src.data_loader import load_tsv

    if args.model_path is None:
        info_path = Path(args.models_dir) / "best_model_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        model_path, model_name = info["path"], info["name"]
    else:
        model_path = args.model_path
        model_name = args.model_name or Path(model_path).stem

    prefix = args.split
    source1 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source1.tsv"))
    s2 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source2.tsv"))
    s3 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source3.tsv"))
    ground_truth = load_tsv(os.path.join(args.data_dir, f"{prefix}_ground_truth.tsv"))

    best_ts, best_tm, best_score, table = tune_dual_thresholds_from_full_pool(
        source1, {"S2": s2, "S3": s3}, ground_truth, model_path, model_name,
        max_block_pairs=args.max_block_pairs,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(Path(args.output_dir) / "dual_threshold_search.csv", index=False)

    result = {
        "model_name": model_name,
        "model_path": str(model_path),
        "singleton_threshold": best_ts,
        "match_threshold": best_tm,
        "entity_macro_f0.5": best_score,
    }
    (Path(args.output_dir) / "dual_threshold_info.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the decision threshold(s) for a trained model.")
    parser.add_argument(
        "--mode", choices=["single", "dual"], default="dual",
        help="'dual' (default): (T_singleton, T_match) via the REAL entity-level macro "
             "F0.5 against the full candidate pool - use this for an actual submission. "
             "'single': one threshold via a class-balanced val set + row-level macro "
             "F0.5, a cheap diagnostic only.",
    )
    parser.add_argument("--model-path", default=None,
                         help="Defaults to the path recorded in <models-dir>/best_model_info.json")
    parser.add_argument("--model-name", default=None,
                         help="One of logistic_regression/lightgbm/xgboost/catboost; inferred from --model-path if omitted")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="models")
    # --mode single
    parser.add_argument("--val-pairs", default="output/validation_pairs.parquet")
    parser.add_argument("--low", type=float, default=DEFAULT_LOW)
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    # --mode dual
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max-block-pairs", type=int, default=200_000)
    args = parser.parse_args()

    if args.mode == "single":
        _run_single_mode(args)
    else:
        _run_dual_mode(args)


if __name__ == "__main__":
    main()
