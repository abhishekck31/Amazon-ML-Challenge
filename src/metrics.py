"""
Entity-level Macro F0.5 - the ACTUAL Amazon ML Challenge 2026 leaderboard metric.

Per the official problem statement: the score is F0.5 computed independently for each
Source1 entity (comparing its predicted match set against its true match set), then
averaged across all Source1 entities:

    Score = (1 / |S1|) * sum_{i in S1} F0.5(pred_i, true_i)

This is NOT sklearn.metrics.fbeta_score(y_true, y_pred, average="macro") run on
flattened (source1, candidate) pairs - that "macro" averages the score over the two
CLASSES (match / non-match), which is a different number entirely. The competition's
"macro" means "average the per-entity score over entities". Confusing the two silently
optimizes the wrong objective, which is exactly what earlier phases of this project did
(src/train.py, src/threshold_search.py) before this fix.

Singleton rule, straight from the problem statement: if a Source1 entity has NO true
matches, predicting an empty set scores 1.0; predicting even one wrong candidate scores
0.0 for that entity. This makes precision far more valuable than recall on singletons,
which is why the pipeline needs a dedicated singleton threshold (see
src.threshold_search.search_dual_thresholds) rather than one global cutoff.
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd


def entity_f0_5(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    F0.5 for one Source1 entity's predicted vs. true match set.

    * true empty, pred empty  -> 1.0 (correctly predicted a singleton)
    * true empty, pred non-empty -> 0.0 (any false positive on a singleton is fatal)
    * true non-empty, pred empty -> 0.0 (recall = 0)
    * otherwise standard F-beta with beta=0.5 (precision weighted 2x over recall)
    """
    if not true_ids and not pred_ids:
        return 1.0
    if not pred_ids or not true_ids:
        return 0.0

    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    beta2 = 0.25  # beta ** 2, beta = 0.5
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def entity_level_macro_f05(
    true_map: Dict[str, Set[str]],
    pred_map: Dict[str, Set[str]],
    entity_ids: Iterable[str],
) -> float:
    """
    Average entity_f0_5 over every id in `entity_ids` (the full Source1 population -
    NOT just the ids that happen to be keys in true_map/pred_map). An entity missing
    from either map is treated as predicting/truly having an empty set, exactly like
    a row with matched_entity_ids="" in the TSV format.
    """
    entity_ids = list(entity_ids)
    if not entity_ids:
        return 0.0
    scores = [
        entity_f0_5(true_map.get(eid, set()), pred_map.get(eid, set()))
        for eid in entity_ids
    ]
    return float(np.mean(scores))


def ground_truth_to_map(ground_truth: pd.DataFrame) -> Dict[str, Set[str]]:
    """train_ground_truth.tsv (source1_entity_id, matched_entity_ids) -> {id: set(matches)}."""
    result: Dict[str, Set[str]] = {}
    for sid, matched in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        result[sid] = set(matched.split(",")) if isinstance(matched, str) and matched else set()
    return result


def matching_results_to_map(matching_results: pd.DataFrame) -> Dict[str, Set[str]]:
    """matching_results.tsv (source1_entity_id, matched_entity_ids) -> {id: set(matches)}, same shape
    as ground_truth_to_map so the two can be compared directly."""
    return ground_truth_to_map(matching_results)


def pairs_to_map(
    pairs: pd.DataFrame,
    id_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
) -> Dict[str, Set[str]]:
    """Long-format (one row per pair) DataFrame -> {source1_entity_id: set(candidate_entity_id)}."""
    result: Dict[str, Set[str]] = {}
    for sid, cid in zip(pairs[id_col], pairs[cand_col]):
        result.setdefault(sid, set()).add(cid)
    return result


def score_matching_results(
    matching_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source1_entity_ids: Iterable[str],
) -> float:
    """Convenience wrapper: entity-level macro F0.5 straight from two TSV-shaped DataFrames."""
    pred_map = matching_results_to_map(matching_results)
    true_map = ground_truth_to_map(ground_truth)
    return entity_level_macro_f05(true_map, pred_map, source1_entity_ids)
