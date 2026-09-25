"""
Unit tests for src/metrics.py - the entity-level macro F0.5 leaderboard metric.

Run with:  python -m unittest tests.test_metrics -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.metrics import (
    entity_f0_5,
    entity_level_macro_f05,
    ground_truth_to_map,
    pairs_to_map,
    score_matching_results,
)


class TestEntityF05(unittest.TestCase):
    def test_singleton_correct_prediction_scores_one(self):
        self.assertEqual(entity_f0_5(set(), set()), 1.0)

    def test_singleton_false_positive_scores_zero(self):
        # This is the exact "singleton trap" from the problem statement: one wrong
        # candidate on a true singleton must be catastrophic, not merely penalized.
        self.assertEqual(entity_f0_5(set(), {"S2-1"}), 0.0)

    def test_missed_all_matches_scores_zero(self):
        self.assertEqual(entity_f0_5({"S2-1", "S3-2"}, set()), 0.0)

    def test_perfect_match_scores_one(self):
        self.assertEqual(entity_f0_5({"S2-1", "S3-2"}, {"S2-1", "S3-2"}), 1.0)

    def test_precision_weighted_over_recall(self):
        # 1 true positive + 1 false positive (precision=0.5, recall=1.0) should score
        # LOWER than 1 true positive + 1 false negative (precision=1.0, recall=0.5),
        # because F0.5 weights precision 2x over recall.
        high_recall_low_precision = entity_f0_5({"S2-1"}, {"S2-1", "S2-2"})
        high_precision_low_recall = entity_f0_5({"S2-1", "S2-2"}, {"S2-1"})
        self.assertLess(high_recall_low_precision, high_precision_low_recall)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(entity_f0_5({"S2-1"}, {"S2-2"}), 0.0)

    def test_known_f05_value(self):
        # true={A,B,C}, pred={A,B,D}: tp=2, precision=2/3, recall=2/3
        # F0.5 = 1.25 * p * r / (0.25*p + r) = 1.25*(2/3)*(2/3) / (0.25*(2/3)+(2/3))
        score = entity_f0_5({"A", "B", "C"}, {"A", "B", "D"})
        p = r = 2 / 3
        expected = 1.25 * p * r / (0.25 * p + r)
        self.assertAlmostEqual(score, expected, places=9)


class TestEntityLevelMacroF05(unittest.TestCase):
    def test_averages_over_all_entities_not_just_predicted_ones(self):
        # entity C is in entity_ids but missing from both maps entirely - must count
        # as an empty/empty singleton match (score 1.0), not be silently skipped.
        true_map = {"A": {"S2-1"}, "B": set()}
        pred_map = {"A": {"S2-1"}}
        score = entity_level_macro_f05(true_map, pred_map, entity_ids=["A", "B", "C"])
        # A: 1.0 (perfect), B: 1.0 (empty/empty, absent from pred_map), C: 1.0 (absent from both)
        self.assertEqual(score, 1.0)

    def test_mixed_scores_average_correctly(self):
        true_map = {"A": {"S2-1"}, "B": set()}
        pred_map = {"A": set(), "B": {"S2-9"}}  # A: missed match (0.0), B: false positive singleton (0.0)
        score = entity_level_macro_f05(true_map, pred_map, entity_ids=["A", "B"])
        self.assertEqual(score, 0.0)

    def test_this_is_not_sklearn_macro_fbeta_on_flattened_pairs(self):
        # Two entities: one perfect multi-match, one perfect singleton. Entity-level
        # macro F0.5 must be a clean 1.0 regardless of pair-count imbalance between them
        # (this would NOT hold for a naive flattened-pairs row-level metric).
        true_map = {"A": {"S2-1", "S2-2", "S3-3"}, "B": set()}
        pred_map = {"A": {"S2-1", "S2-2", "S3-3"}, "B": set()}
        score = entity_level_macro_f05(true_map, pred_map, entity_ids=["A", "B"])
        self.assertEqual(score, 1.0)

    def test_empty_entity_ids_returns_zero(self):
        self.assertEqual(entity_level_macro_f05({}, {}, entity_ids=[]), 0.0)


class TestMapBuilders(unittest.TestCase):
    def test_ground_truth_to_map_handles_missing_and_multi(self):
        gt = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-2", None, ""],
        })
        result = ground_truth_to_map(gt)
        self.assertEqual(result["S1-1"], {"S2-1", "S3-2"})
        self.assertEqual(result["S1-2"], set())
        self.assertEqual(result["S1-3"], set())

    def test_pairs_to_map_groups_by_source1_entity(self):
        pairs = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-2", "S2-9"],
        })
        result = pairs_to_map(pairs)
        self.assertEqual(result["S1-1"], {"S2-1", "S3-2"})
        self.assertEqual(result["S1-2"], {"S2-9"})


class TestScoreMatchingResults(unittest.TestCase):
    def test_end_to_end_scoring(self):
        ground_truth = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-2", "", "S2-9"],
        })
        matching_results = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-2", "", "S2-9,S2-99"],  # S1-3 has a false positive
        })
        score = score_matching_results(matching_results, ground_truth, ["S1-1", "S1-2", "S1-3"])
        # S1-1: 1.0, S1-2: 1.0, S1-3: precision=1/2, recall=1 -> F0.5
        p, r = 0.5, 1.0
        expected_s1_3 = 1.25 * p * r / (0.25 * p + r)
        self.assertAlmostEqual(score, (1.0 + 1.0 + expected_s1_3) / 3, places=9)

    def test_missing_source1_row_counts_as_empty_prediction(self):
        # matching_results MUST have one row per Source1 entity; if a caller forgot
        # one, scoring should still work by treating it as an empty prediction rather
        # than crashing - useful as a defensive sanity check on malformed output.
        ground_truth = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-1", ""],
        })
        matching_results = pd.DataFrame({
            "source1_entity_id": ["S1-1"],
            "matched_entity_ids": ["S2-1"],
        })
        score = score_matching_results(matching_results, ground_truth, ["S1-1", "S1-2"])
        self.assertEqual(score, 1.0)  # S1-1 perfect, S1-2 missing row -> treated as empty/empty


if __name__ == "__main__":
    unittest.main()
