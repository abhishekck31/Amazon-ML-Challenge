"""
Phase 8 - End-to-end inference pipeline for Amazon ML Challenge 2026: Business Entity
Resolution.

Wires together every earlier phase into one call: preprocess (Phase 2) -> generate
candidates (Phase 3) -> compute features (Phase 4) -> load the trained model (Phase 6)
-> score candidates -> apply the tuned threshold (Phase 7) -> group per source1 entity
-> write a submission-format matching_results.tsv, guaranteed to have exactly one row
per Source1 entity and no duplicate candidate IDs within a row.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.blocking import generate_candidate_pairs
from src.data_loader import load_tsv
from src.features import generate_features, get_feature_matrix
from src.preprocessing import add_clean_columns
from src.threshold_search import load_predict_proba_fn


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #

def preprocess_sources(
    source1: pd.DataFrame, targets: Dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Phase 2: add name_clean / address_clean to source1 and every target source."""
    return add_clean_columns(source1), {label: add_clean_columns(df) for label, df in targets.items()}


def score_candidates(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    model_path: str,
    model_name: str,
) -> pd.DataFrame:
    """Phase 4 features + Phase 6 model -> a match_probability column on every candidate pair."""
    features = generate_features(candidate_pairs, source1, targets, verbose=False)
    predict_fn = load_predict_proba_fn(model_path, model_name)
    features["match_probability"] = predict_fn(get_feature_matrix(features))
    return features


def group_predictions(
    scored_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Apply `threshold`, then collapse to one row per Source1 entity in the
    ground-truth-compatible (source1_entity_id, matched_entity_ids) format:

    * every entity_id in `source1` gets exactly one output row, including entities
      with zero matches above threshold (matched_entity_ids = "") - blocking recall
      misses or low-confidence scores must not silently drop an entity from the file.
    * candidate IDs are de-duplicated and sorted before joining, so a candidate that
      matched via more than one blocking rule (already deduped by src.blocking, but
      kept here as a second guarantee) never appears twice in one row.
    """
    matches = scored_pairs.loc[scored_pairs["match_probability"] >= threshold]

    grouped = (
        matches.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .rename("matched_entity_ids")
        .reset_index()
    )

    all_entities = pd.DataFrame({"source1_entity_id": source1["entity_id"].unique()})
    result = all_entities.merge(grouped, on="source1_entity_id", how="left")
    result["matched_entity_ids"] = result["matched_entity_ids"].fillna("")

    # Safety net: entity_id should already be unique in source1, but a duplicated
    # source row must never turn into a duplicated output row.
    result = result.drop_duplicates(subset="source1_entity_id").reset_index(drop=True)
    return result


def validate_matching_results(result: pd.DataFrame, source1: pd.DataFrame) -> None:
    """Hard checks on the two explicit output guarantees: one row per Source1 entity,
    no duplicate candidate IDs within a row."""
    expected_ids = set(source1["entity_id"])
    actual_ids = set(result["source1_entity_id"])
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        raise ValueError(f"{len(missing):,} Source1 entities missing from matching_results output")
    if extra:
        raise ValueError(f"{len(extra):,} unexpected entity_ids in matching_results output")
    if result["source1_entity_id"].duplicated().any():
        raise ValueError("Duplicate source1_entity_id rows in matching_results output")

    for ids_str in result["matched_entity_ids"]:
        if not ids_str:
            continue
        ids = ids_str.split(",")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate candidate IDs within one row: {ids_str}")


def save_matching_results(result: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #

def run_inference(
    source1: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    model_path: str,
    model_name: str,
    threshold: float,
    max_block_pairs: int = 200_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: preprocess -> generate candidates -> compute features -> score with
    the trained model -> apply threshold -> group into one row per Source1 entity.
    Returns the matching_results DataFrame (not yet written to disk).
    """
    t0 = time.time()
    source1, targets = preprocess_sources(source1, targets)
    if verbose:
        print(f"Preprocessed sources in {time.time() - t0:.1f}s")

    t1 = time.time()
    candidate_pairs = generate_candidate_pairs(source1, targets, max_block_pairs=max_block_pairs, verbose=verbose)
    if verbose:
        print(f"Generated {len(candidate_pairs):,} candidate pairs in {time.time() - t1:.1f}s")

    t2 = time.time()
    scored_pairs = score_candidates(candidate_pairs, source1, targets, model_path, model_name)
    if verbose:
        print(f"Scored candidates in {time.time() - t2:.1f}s")

    result = group_predictions(scored_pairs, source1, threshold)
    validate_matching_results(result, source1)

    if verbose:
        n_with_match = (result["matched_entity_ids"] != "").sum()
        print(
            f"{n_with_match:,}/{len(result):,} Source1 entities have >=1 match "
            f"at threshold={threshold:.3f} (total time {time.time() - t0:.1f}s)"
        )

    return result


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _resolve_model(models_dir: str, model_path: Optional[str], model_name: Optional[str], threshold: Optional[float]):
    if model_path is None:
        info = json.loads((Path(models_dir) / "best_model_info.json").read_text(encoding="utf-8"))
        resolved_path, resolved_name = info["path"], info["name"]
        resolved_threshold = threshold if threshold is not None else info["threshold"]
    else:
        resolved_path, resolved_name = model_path, model_name or Path(model_path).stem
        resolved_threshold = threshold if threshold is not None else 0.5
    return resolved_path, resolved_name, resolved_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 8 end-to-end inference pipeline.")
    parser.add_argument("--data-dir", default="dataset/test")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                         help="File prefix to load (train_*.tsv or test_*.tsv).")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                         help="Overrides the threshold recorded in best_model_info.json.")
    parser.add_argument("--max-block-pairs", type=int, default=200_000)
    parser.add_argument("--output", default="output/matching_results.tsv")
    args = parser.parse_args()

    prefix = args.split
    source1 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source1.tsv"))
    s2 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source2.tsv"))
    s3 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source3.tsv"))

    model_path, model_name, threshold = _resolve_model(
        args.models_dir, args.model_path, args.model_name, args.threshold
    )
    print(f"Using model={model_name} path={model_path} threshold={threshold:.3f}")

    result = run_inference(
        source1, {"S2": s2, "S3": s3}, model_path, model_name, threshold,
        max_block_pairs=args.max_block_pairs,
    )

    save_matching_results(result, args.output)
    print(f"Wrote {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
