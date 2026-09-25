"""
Unit tests for the entity-stratified split added to src/training_data.py.

Run with:  python -m unittest tests.test_training_data_stratified -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.training_data import stratified_split_train_validation


class TestStratifiedSplit(unittest.TestCase):
    def _make_labeled_pairs(self):
        rows = []
        # 20 singleton entities (only a hard negative, no positive), split across US/India
        for i in range(20):
            rows.append({"source1_entity_id": f"S1-single-{i}", "candidate_entity_id": f"S2-neg-{i}",
                         "source": "S2", "label": 0})
        # 20 single-match entities
        for i in range(20):
            rows.append({"source1_entity_id": f"S1-single-match-{i}", "candidate_entity_id": f"S2-pos-{i}",
                         "source": "S2", "label": 1})
            rows.append({"source1_entity_id": f"S1-single-match-{i}", "candidate_entity_id": f"S2-negb-{i}",
                         "source": "S2", "label": 0})
        # 20 multi-match entities
        for i in range(20):
            rows.append({"source1_entity_id": f"S1-multi-{i}", "candidate_entity_id": f"S2-posA-{i}",
                         "source": "S2", "label": 1})
            rows.append({"source1_entity_id": f"S1-multi-{i}", "candidate_entity_id": f"S3-posB-{i}",
                         "source": "S3", "label": 1})
        return pd.DataFrame(rows)

    def _make_source1(self, labeled_pairs):
        entities = labeled_pairs["source1_entity_id"].unique()
        countries = ["US" if i % 2 == 0 else "India" for i in range(len(entities))]
        return pd.DataFrame({
            "entity_id": entities,
            "business_name": ["x"] * len(entities),
            "business_address": ["y"] * len(entities),
            "country": countries,
        })

    def test_no_entity_leakage(self):
        labeled_pairs = self._make_labeled_pairs()
        source1 = self._make_source1(labeled_pairs)
        train, val = stratified_split_train_validation(labeled_pairs, source1, val_size=0.2, random_state=1)
        overlap = set(train["source1_entity_id"]) & set(val["source1_entity_id"])
        self.assertEqual(overlap, set())

    def test_covers_every_pair(self):
        labeled_pairs = self._make_labeled_pairs()
        source1 = self._make_source1(labeled_pairs)
        train, val = stratified_split_train_validation(labeled_pairs, source1, val_size=0.2, random_state=1)
        self.assertEqual(len(train) + len(val), len(labeled_pairs))

    def test_validation_contains_all_three_match_buckets(self):
        # With 20 entities in each of singleton/single-match/multi-match and a 20%
        # split, validation should get some of each bucket, not accidentally all-one-type.
        labeled_pairs = self._make_labeled_pairs()
        source1 = self._make_source1(labeled_pairs)
        _, val = stratified_split_train_validation(labeled_pairs, source1, val_size=0.2, random_state=1)

        val_entities = set(val["source1_entity_id"])
        has_singleton = any(e.startswith("S1-single-") and "match" not in e for e in val_entities)
        has_single_match = any("single-match" in e for e in val_entities)
        has_multi = any(e.startswith("S1-multi-") for e in val_entities)
        self.assertTrue(has_singleton)
        self.assertTrue(has_single_match)
        self.assertTrue(has_multi)


if __name__ == "__main__":
    unittest.main()
