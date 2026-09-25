"""
Phase 7 - Decision threshold optimization for Amazon ML Challenge 2026: Business Entity
Resolution.

A trained classifier (Phase 6) outputs a match probability; turning that into a
match/no-match decision requires picking a threshold. This module sweeps thresholds in
[0.30, 0.99] (below 0.30 is never a sane "this is a match" cutoff for this problem),
scores each with macro F0.5, and reports the threshold that maximizes it, alongside
precision-recall curves for the underlying model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

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
    `predict_proba(X) -> P(label=1)` callable."""
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
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the decision threshold for a trained model.")
    parser.add_argument("--model-path", default=None,
                         help="Defaults to the path recorded in <models-dir>/best_model_info.json")
    parser.add_argument("--model-name", default=None,
                         help="One of logistic_regression/lightgbm/xgboost/catboost; inferred from --model-path if omitted")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--val-pairs", default="output/validation_pairs.parquet")
    parser.add_argument("--low", type=float, default=DEFAULT_LOW)
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
