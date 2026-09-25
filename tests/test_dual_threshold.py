"""
Unit tests for the dual-threshold / submission-format fixes:
- src.blocking.group_entity_ids / to_submission_format
- src.threshold_search.apply_dual_threshold / search_dual_thresholds

Run with:  python -m unittest tests.test_dual_threshold -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.blocking import group_entity_ids, to_submission_format
from src.threshold_search import apply_dual_threshold, search_dual_thresholds


class TestGroupEntityIds(unittest.TestCase):
    def test_every_entity_present_even_with_no_rows(self):
        pairs = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "candidate_entity_id": ["S2-1", "S3-2"],
        })
        result = group_entity_ids(pairs, ["S1-1", "S1-2"])
        self.assertEqual(set(result["source1_entity_id"]), {"S1-1", "S1-2"})
        row2 = result.loc[result["source1_entity_id"] == "S1-2", "candidate_entity_ids"].iloc[0]
        self.assertEqual(row2, "")

    def test_deduplicates_and_sorts(self):
        pairs = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-1"],
            "candidate_entity_id": ["S3-2", "S2-1", "S3-2"],  # duplicate + out of order
        })
        result = group_entity_ids(pairs, ["S1-1"])
        self.assertEqual(result["candidate_entity_ids"].iloc[0], "S2-1,S3-2")

    def test_empty_pairs_still_covers_all_entities(self):
        pairs = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
        result = group_entity_ids(pairs, ["S1-1", "S1-2"])
        self.assertEqual(len(result), 2)
        self.assertTrue((result["candidate_entity_ids"] == "").all())


class TestToSubmissionFormat(unittest.TestCase):
    def test_one_row_per_source1_entity(self):
        candidate_pairs = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-2", "S2-9"],
            "source": ["S2", "S3", "S2"],
            "rules_matched": ["r1", "r1", "r1"],
        })
        source1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"]})  # S1-3 has zero candidates
        result = to_submission_format(candidate_pairs, source1)
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result.columns), {"source1_entity_id", "candidate_entity_ids"})
        row3 = result.loc[result["source1_entity_id"] == "S1-3", "candidate_entity_ids"].iloc[0]
        self.assertEqual(row3, "")


class TestApplyDualThreshold(unittest.TestCase):
    def _make_scored_pairs(self):
        return pd.DataFrame({
            "source1_entity_id": ["A", "A", "B", "B", "C"],
            "candidate_entity_id": ["c1", "c2", "c3", "c4", "c5"],
            "match_probability": [0.9, 0.4, 0.2, 0.1, 0.6],
        })

    def test_singleton_gate_suppresses_low_confidence_entity(self):
        scored = self._make_scored_pairs()
        # entity B's best prob is 0.2, below singleton_threshold=0.3 -> gets nothing
        accepted = apply_dual_threshold(scored, singleton_threshold=0.3, match_threshold=0.3)
        self.assertNotIn("B", set(accepted["source1_entity_id"]))

    def test_match_threshold_filters_within_admitted_entity(self):
        scored = self._make_scored_pairs()
        # entity A clears singleton gate (max=0.9 >= 0.3); only c1 (0.9) clears match_threshold=0.5
        accepted = apply_dual_threshold(scored, singleton_threshold=0.3, match_threshold=0.5)
        a_rows = accepted.loc[accepted["source1_entity_id"] == "A"]
        self.assertEqual(set(a_rows["candidate_entity_id"]), {"c1"})


class TestSearchDualThresholds(unittest.TestCase):
    def test_recovers_perfect_thresholds_on_separable_data(self):
        # A: true match c1 (prob 0.9), a decoy c2 (prob 0.2) - correct answer needs
        # match_threshold in (0.2, 0.9]. B: true singleton, its only candidate c3
        # scores low (0.1) - correct answer needs singleton_threshold > 0.1.
        scored_pairs = pd.DataFrame({
            "source1_entity_id": ["A", "A", "B"],
            "candidate_entity_id": ["c1", "c2", "c3"],
            "match_probability": [0.9, 0.2, 0.1],
        })
        ground_truth_map = {"A": {"c1"}, "B": set()}
        singleton_grid = np.array([0.05, 0.15, 0.5])
        match_grid = np.array([0.05, 0.15, 0.5])

        best_ts, best_tm, best_score, table = search_dual_thresholds(
            scored_pairs, ground_truth_map, entity_ids=["A", "B"],
            singleton_grid=singleton_grid, match_grid=match_grid,
        )
        # Multiple (T_s, T_m) combinations can tie for a perfect score here (e.g.
        # match_threshold=0.5 alone also filters out B's low-confidence candidate,
        # regardless of T_s) - what matters is the optimizer actually finds one.
        self.assertEqual(best_score, 1.0)
        self.assertTrue((table["entity_macro_f0.5"] == 1.0).any())

    def test_result_matches_manual_entity_f05_computation(self):
        from src.metrics import entity_f0_5

        scored_pairs = pd.DataFrame({
            "source1_entity_id": ["A", "A"],
            "candidate_entity_id": ["c1", "c2"],
            "match_probability": [0.8, 0.6],
        })
        ground_truth_map = {"A": {"c1", "c2"}}
        best_ts, best_tm, best_score, table = search_dual_thresholds(
            scored_pairs, ground_truth_map, entity_ids=["A"],
            singleton_grid=np.array([0.5]), match_grid=np.array([0.5]),
        )
        expected = entity_f0_5({"c1", "c2"}, {"c1", "c2"})
        self.assertAlmostEqual(best_score, expected, places=9)
        self.assertEqual(best_score, 1.0)


if __name__ == "__main__":
    unittest.main()
