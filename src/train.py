"""
Phase 6 - Model training for Amazon ML Challenge 2026: Business Entity Resolution.

Trains four candidate classifiers on the Phase 5 train_pairs.parquet /
validation_pairs.parquet (Logistic Regression baseline, LightGBM, XGBoost, and
CatBoost if installed), evaluates each on validation with a tuned decision threshold,
and saves every trained model plus the best one under models/.

Metric: macro F0.5 (sklearn.metrics.fbeta_score(beta=0.5, average="macro")) - F0.5
weights precision over recall, which fits entity resolution: a wrong match (false
positive) pollutes downstream data more than a missed match (false negative) that can
still be caught by another blocking rule or a manual review pass. "Macro" averages the
score computed independently on the match / non-match classes, so performance on the
minority "match" class isn't swamped by the much larger "non-match" class.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

from src.features import get_feature_matrix
from src.threshold_search import find_best_threshold


@dataclass
class TrainedModel:
    name: str
    extension: str
    val_probs: np.ndarray
    threshold: float
    metrics: Dict[str, float]
    save_fn: Callable[[Path], None] = field(repr=False)
    model: Any = field(default=None, repr=False)
    """The fitted estimator/booster itself (kept in memory, not just save_fn), so
    callers - e.g. the leaderboard-optimization notebook's feature-importance and SHAP
    sections - can reuse the exact trained model without retraining it a second time."""


# --------------------------------------------------------------------------- #
# Shared evaluation (threshold tuning itself lives in src.threshold_search,
# Phase 7's dedicated module, so both phases use one implementation)
# --------------------------------------------------------------------------- #

def _finalize(
    name: str,
    y_val: np.ndarray,
    val_probs: np.ndarray,
    save_fn: Callable[[Path], None],
    extension: str,
    model: Any = None,
) -> TrainedModel:
    threshold, macro_f05, _ = find_best_threshold(y_val, val_probs)
    preds = (val_probs >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "macro_f0.5": macro_f05,
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "recall": float(recall_score(y_val, preds, zero_division=0)),
        "average_precision": float(average_precision_score(y_val, val_probs)),
    }
    return TrainedModel(name=name, extension=extension, val_probs=val_probs,
                         threshold=threshold, metrics=metrics, save_fn=save_fn, model=model)


# --------------------------------------------------------------------------- #
# Model trainers - one function per algorithm, uniform TrainedModel return
# --------------------------------------------------------------------------- #

def train_logistic_regression(X_train, y_train, X_val, y_val, random_state: int = 42) -> TrainedModel:
    """Baseline: standardized features + L2 logistic regression. No early stopping
    concept applies here (it isn't an iterative boosting method)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    )
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]

    def save_fn(path: Path):
        import joblib
        joblib.dump(model, path)

    return _finalize("logistic_regression", y_val, val_probs, save_fn, ".joblib", model=model)


def train_lightgbm(
    X_train, y_train, X_val, y_val,
    random_state: int = 42, num_boost_round: int = 500, early_stopping_rounds: int = 30,
) -> TrainedModel:
    import lightgbm as lgb

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": random_state,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)

    def save_fn(path: Path):
        model.save_model(str(path))

    return _finalize("lightgbm", y_val, val_probs, save_fn, ".txt", model=model)


def train_xgboost(
    X_train, y_train, X_val, y_val,
    random_state: int = 42, num_boost_round: int = 500, early_stopping_rounds: int = 30,
) -> TrainedModel:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=num_boost_round,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_probs = model.predict_proba(X_val)[:, 1]

    def save_fn(path: Path):
        model.save_model(str(path))

    return _finalize("xgboost", y_val, val_probs, save_fn, ".json", model=model)


def train_catboost(
    X_train, y_train, X_val, y_val,
    random_state: int = 42, num_boost_round: int = 500, early_stopping_rounds: int = 30,
) -> Optional[TrainedModel]:
    """Optional per the spec: returns None (instead of raising) if catboost isn't installed."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None

    model = CatBoostClassifier(
        iterations=num_boost_round,
        random_seed=random_state,
        early_stopping_rounds=early_stopping_rounds,
        loss_function="Logloss",
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    val_probs = model.predict_proba(X_val)[:, 1]

    def save_fn(path: Path):
        model.save_model(str(path))

    return _finalize("catboost", y_val, val_probs, save_fn, ".cbm", model=model)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

MODEL_TRAINERS = {
    "logistic_regression": train_logistic_regression,
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "catboost": train_catboost,
}


def train_all_models(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    random_state: int = 42,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    verbose: bool = True,
) -> Dict[str, TrainedModel]:
    """Train every model in MODEL_TRAINERS on train_pairs, evaluate on val_pairs.
    Returns {name: TrainedModel}, skipping any trainer that returns None (catboost
    when not installed)."""
    X_train = get_feature_matrix(train_pairs)
    y_train = train_pairs["label"].to_numpy()
    X_val = get_feature_matrix(val_pairs)
    y_val = val_pairs["label"].to_numpy()

    results: Dict[str, TrainedModel] = {}
    for name, trainer in MODEL_TRAINERS.items():
        t0 = time.time()
        kwargs = {"random_state": random_state}
        if name in ("lightgbm", "xgboost", "catboost"):
            kwargs.update(num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds)
        result = trainer(X_train, y_train, X_val, y_val, **kwargs)
        elapsed = time.time() - t0
        if result is None:
            if verbose:
                print(f"{name}: skipped (not installed)")
            continue
        results[name] = result
        if verbose:
            m = result.metrics
            print(
                f"{name}: macro_f0.5={m['macro_f0.5']:.4f} threshold={m['threshold']:.3f} "
                f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
                f"avg_precision={m['average_precision']:.4f} ({elapsed:.1f}s)"
            )
    return results


def select_best_model(results: Dict[str, TrainedModel]) -> str:
    """Best = highest validation macro F0.5 at its own tuned threshold."""
    return max(results, key=lambda name: results[name].metrics["macro_f0.5"])


def plot_precision_recall_curves(
    results: Dict[str, TrainedModel], y_val: np.ndarray, output_path: str
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_val, result.val_probs)
        ax.plot(recall, precision, label=f"{name} (AP={result.metrics['average_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (validation)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_models(results: Dict[str, TrainedModel], models_dir: str) -> Dict[str, str]:
    """Save every trained model under models_dir, one native-format file per model.
    Returns {name: saved_path}."""
    os.makedirs(models_dir, exist_ok=True)
    paths = {}
    for name, result in results.items():
        path = Path(models_dir) / f"{name}{result.extension}"
        result.save_fn(path)
        paths[name] = str(path)
    return paths


def save_best_model(
    results: Dict[str, TrainedModel], saved_paths: Dict[str, str], models_dir: str
) -> str:
    """Copy the best model to models/best_model<ext> and write models/best_model_info.json
    with its name, threshold, and validation metrics, so inference code knows which
    file to load and what decision threshold to apply."""
    best_name = select_best_model(results)
    best = results[best_name]
    best_path = Path(saved_paths[best_name])
    best_model_path = Path(models_dir) / f"best_model{best.extension}"
    shutil.copyfile(best_path, best_model_path)

    info = {
        "name": best_name,
        "path": str(best_model_path),
        "extension": best.extension,
        "threshold": best.threshold,
        "metrics": best.metrics,
    }
    info_path = Path(models_dir) / "best_model_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return best_name


def save_comparison_table(results: Dict[str, TrainedModel], models_dir: str) -> pd.DataFrame:
    os.makedirs(models_dir, exist_ok=True)
    rows = [{"model": name, **result.metrics} for name, result in results.items()]
    table = pd.DataFrame(rows).sort_values("macro_f0.5", ascending=False).reset_index(drop=True)
    table.to_csv(Path(models_dir) / "model_comparison.csv", index=False)
    return table


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 6 candidate classifiers.")
    parser.add_argument("--train-pairs", default="output/train_pairs.parquet")
    parser.add_argument("--val-pairs", default="output/validation_pairs.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    train_pairs = pd.read_parquet(args.train_pairs)
    val_pairs = pd.read_parquet(args.val_pairs)
    print(f"Loaded {len(train_pairs):,} train / {len(val_pairs):,} validation rows "
          f"in {time.time() - t0:.1f}s")

    results = train_all_models(
        train_pairs, val_pairs,
        random_state=args.random_state,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    table = save_comparison_table(results, args.models_dir)
    print("\nModel comparison (sorted by macro F0.5):")
    print(table.to_string(index=False))

    plot_precision_recall_curves(
        results, val_pairs["label"].to_numpy(),
        os.path.join(args.models_dir, "precision_recall_curves.png"),
    )

    saved_paths = save_models(results, args.models_dir)
    best_name = save_best_model(results, saved_paths, args.models_dir)
    print(f"\nBest model: {best_name} "
          f"(macro F0.5={results[best_name].metrics['macro_f0.5']:.4f}, "
          f"threshold={results[best_name].metrics['threshold']:.3f})")
    print(f"Saved all models + best_model_info.json under {args.models_dir}/ "
          f"in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
