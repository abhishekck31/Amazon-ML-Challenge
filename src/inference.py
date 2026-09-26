"""
Phase 8 - End-to-end inference pipeline for Amazon ML Challenge 2026: Business Entity
Resolution.

Wires together every earlier phase into one call: preprocess (Phase 2) -> generate
candidates (Phase 3) -> compute features (Phase 4) -> load the trained model (Phase 6)
-> score candidates -> apply the dual decision threshold (Phase 7) -> group per
Source1 entity -> write BOTH required submission files:

* candidate_pairs.tsv  - every candidate that survived blocking, one row per Source1
  entity (NOT filtered by the model threshold - this is "the exact candidate set fed
  into the model before final thresholding" per the problem statement).
* matching_results.tsv - only the candidates the dual threshold accepted; the only
  file actually scored on the leaderboard.

Decision rule (the "Trap 2/Trap 5" fixes from the competition rules review): for each
Source1 entity, if its best candidate probability is below T_singleton, predict no
matches at all (a true singleton scores 1.0 for an empty prediction, 0.0 for even one
wrong guess) - otherwise accept every candidate scoring >= T_match. Every ID in
matching_results.tsv is therefore guaranteed to be a subset of the same entity's
candidate_pairs.tsv row, since it can only ever be drawn from that same candidate set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.blocking import generate_candidate_pairs, group_entity_ids, to_submission_format
from src.data_loader import load_tsv
from src.features import generate_features, get_feature_matrix
from src.preprocessing import add_clean_columns
from src.scoring import score_pairs_chunked
from src.threshold_search import apply_dual_threshold, load_predict_proba_fn


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
    singleton_threshold: float,
    match_threshold: float,
) -> pd.DataFrame:
    """
    Apply the dual decision threshold, then collapse to one row per Source1 entity in
    the ground-truth-compatible (source1_entity_id, matched_entity_ids) format.

    * every entity_id in `source1` gets exactly one output row, including entities
      with zero accepted matches (matched_entity_ids = "").
    * candidate IDs are de-duplicated and sorted before joining, so a candidate that
      matched via more than one blocking rule (already deduped by src.blocking, but
      kept here as a second guarantee) never appears twice in one row.
    """
    accepted = apply_dual_threshold(scored_pairs, singleton_threshold, match_threshold)
    return group_entity_ids(
        accepted, source1["entity_id"].unique(),
        id_col="source1_entity_id", cand_col="candidate_entity_id", out_col="matched_entity_ids",
    )


def validate_matching_results(
    matching_results: pd.DataFrame, candidate_pairs_submission: pd.DataFrame, source1: pd.DataFrame
) -> None:
    """
    Hard checks on every explicit output guarantee from the problem statement:
    * one row per Source1 entity in both files, no duplicates.
    * no duplicate candidate IDs within a row.
    * every ID in a matching_results row is a subset of that same entity's
      candidate_pairs row ("Trap 5: Candidate Mismatch Error").
    """
    expected_ids = set(source1["entity_id"])
    for name, df, col in [
        ("matching_results", matching_results, "matched_entity_ids"),
        ("candidate_pairs", candidate_pairs_submission, "candidate_entity_ids"),
    ]:
        actual_ids = set(df["source1_entity_id"])
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        if missing:
            raise ValueError(f"{len(missing):,} Source1 entities missing from {name} output")
        if extra:
            raise ValueError(f"{len(extra):,} unexpected entity_ids in {name} output")
        if df["source1_entity_id"].duplicated().any():
            raise ValueError(f"Duplicate source1_entity_id rows in {name} output")
        for ids_str in df[col]:
            if not ids_str:
                continue
            ids = ids_str.split(",")
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate candidate IDs within one {name} row: {ids_str}")

    candidate_map = dict(zip(candidate_pairs_submission["source1_entity_id"],
                              candidate_pairs_submission["candidate_entity_ids"]))
    for sid, matched_str in zip(matching_results["source1_entity_id"], matching_results["matched_entity_ids"]):
        if not matched_str:
            continue
        matched_ids = set(matched_str.split(","))
        candidate_ids = set(candidate_map.get(sid, "").split(",")) if candidate_map.get(sid) else set()
        if not matched_ids.issubset(candidate_ids):
            raise ValueError(
                f"matching_results has ID(s) for {sid} not present in candidate_pairs: "
                f"{matched_ids - candidate_ids}"
            )


def save_tsv(result: pd.DataFrame, output_path: str) -> None:
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
    singleton_threshold: float,
    match_threshold: float,
    max_block_pairs: int = 50_000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: preprocess -> generate candidates -> compute features -> score with
    the trained model -> apply the dual threshold -> group into one row per Source1
    entity. Returns (matching_results, candidate_pairs_submission), neither written to
    disk yet.
    """
    t0 = time.time()
    source1, targets = preprocess_sources(source1, targets)
    if verbose:
        print(f"Preprocessed sources in {time.time() - t0:.1f}s")

    t1 = time.time()
    candidate_pairs = generate_candidate_pairs(source1, targets, max_block_pairs=max_block_pairs, verbose=verbose)
    if verbose:
        print(f"Generated {len(candidate_pairs):,} candidate pairs in {time.time() - t1:.1f}s")

    t_sub = time.time()
    candidate_pairs_submission = to_submission_format(candidate_pairs, source1)
    if verbose:
        print(f"Built candidate_pairs submission in {time.time() - t_sub:.1f}s", flush=True)

    # Chunked + multi-process: the full test candidate pool is hundreds of millions of
    # pairs, too many to featurize in one single-threaded, all-in-memory pass. Each
    # chunk holds complete source1 entities, so applying the dual threshold per chunk
    # is identical to applying it to the whole table; only accepted pairs are kept.
    t2 = time.time()
    accepted = score_pairs_chunked(
        candidate_pairs, source1, targets, load_predict_proba_fn(model_path, model_name),
        keep_fn=lambda df: apply_dual_threshold(df, singleton_threshold, match_threshold),
        verbose=verbose,
    )
    del candidate_pairs
    if verbose:
        print(f"Scored candidates in {time.time() - t2:.1f}s ({len(accepted):,} accepted pairs)", flush=True)

    matching_results = group_entity_ids(
        accepted, source1["entity_id"].unique(),
        id_col="source1_entity_id", cand_col="candidate_entity_id", out_col="matched_entity_ids",
    )
    validate_matching_results(matching_results, candidate_pairs_submission, source1)

    if verbose:
        n_with_match = (matching_results["matched_entity_ids"] != "").sum()
        print(
            f"{n_with_match:,}/{len(matching_results):,} Source1 entities have >=1 match "
            f"at T_singleton={singleton_threshold:.3f} T_match={match_threshold:.3f} "
            f"(total time {time.time() - t0:.1f}s)"
        )

    return matching_results, candidate_pairs_submission


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _resolve_model(
    models_dir: str,
    model_path: Optional[str],
    model_name: Optional[str],
    singleton_threshold: Optional[float],
    match_threshold: Optional[float],
):
    """
    Reads models/best_model_info.json for the model path/name. Thresholds come from
    models/dual_threshold_info.json (written by threshold_search.py's dual-threshold
    CLI) if present; otherwise both default to best_model_info.json's single
    pair-level threshold (a reasonable but NOT leaderboard-tuned fallback - run
    threshold_search.py's dual-threshold search first for a real submission).
    """
    info = json.loads((Path(models_dir) / "best_model_info.json").read_text(encoding="utf-8"))
    resolved_path = model_path if model_path is not None else info["path"]
    resolved_name = model_name if model_name is not None else info["name"]

    dual_info_path = Path(models_dir) / "dual_threshold_info.json"
    if singleton_threshold is not None and match_threshold is not None:
        resolved_ts, resolved_tm = singleton_threshold, match_threshold
    elif dual_info_path.exists():
        dual_info = json.loads(dual_info_path.read_text(encoding="utf-8"))
        resolved_ts = singleton_threshold if singleton_threshold is not None else dual_info["singleton_threshold"]
        resolved_tm = match_threshold if match_threshold is not None else dual_info["match_threshold"]
    else:
        fallback = info["threshold"]
        resolved_ts = singleton_threshold if singleton_threshold is not None else fallback
        resolved_tm = match_threshold if match_threshold is not None else fallback

    return resolved_path, resolved_name, resolved_ts, resolved_tm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 8 end-to-end inference pipeline.")
    parser.add_argument("--data-dir", default="dataset/test")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                         help="File prefix to load (train_*.tsv or test_*.tsv).")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--singleton-threshold", type=float, default=None,
                         help="Overrides models/dual_threshold_info.json.")
    parser.add_argument("--match-threshold", type=float, default=None,
                         help="Overrides models/dual_threshold_info.json.")
    parser.add_argument("--max-block-pairs", type=int, default=50_000,
                        help="Must match the value the training candidate pool was blocked with.")
    parser.add_argument("--matching-results-output", default="output/matching_results.tsv")
    parser.add_argument("--candidate-pairs-output", default="output/candidate_pairs.tsv")
    args = parser.parse_args()

    prefix = args.split
    source1 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source1.tsv"))
    s2 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source2.tsv"))
    s3 = load_tsv(os.path.join(args.data_dir, f"{prefix}_source3.tsv"))

    model_path, model_name, singleton_threshold, match_threshold = _resolve_model(
        args.models_dir, args.model_path, args.model_name, args.singleton_threshold, args.match_threshold
    )
    print(f"Using model={model_name} path={model_path} "
          f"T_singleton={singleton_threshold:.3f} T_match={match_threshold:.3f}")

    matching_results, candidate_pairs_submission = run_inference(
        source1, {"S2": s2, "S3": s3}, model_path, model_name, singleton_threshold, match_threshold,
        max_block_pairs=args.max_block_pairs,
    )

    save_tsv(candidate_pairs_submission, args.candidate_pairs_output)
    save_tsv(matching_results, args.matching_results_output)
    print(f"Wrote {len(candidate_pairs_submission):,} rows to {args.candidate_pairs_output}")
    print(f"Wrote {len(matching_results):,} rows to {args.matching_results_output}")


if __name__ == "__main__":
    main()
