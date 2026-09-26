"""
Unit tests for the phonetic blocking rule added to src/blocking.py.

Run with:  python -m unittest tests.test_blocking_phonetic -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.blocking import DEFAULT_RULES, _key_country_name_phonetic, build_block_indices


class TestPhoneticRule(unittest.TestCase):
    def test_transliteration_variants_share_a_key(self):
        # Same underlying name, two different Latin-alphabet spellings - exact
        # prefix/token rules would miss this; the phonetic rule should not.
        df = pd.DataFrame({
            "name_clean": ["kumar traders", "coomar traders"],
            "country": ["India", "India"],
        })
        keys = _key_country_name_phonetic(df)
        self.assertEqual(keys.iloc[0], keys.iloc[1])

    def test_different_names_get_different_keys(self):
        df = pd.DataFrame({
            "name_clean": ["kumar traders", "apex logistics"],
            "country": ["India", "India"],
        })
        keys = _key_country_name_phonetic(df)
        self.assertNotEqual(keys.iloc[0], keys.iloc[1])

    def test_not_registered_in_default_rules(self):
        # Dropped from DEFAULT_RULES: at full dataset scale it was responsible for
        # ~56M raw candidate pairs per target source for the narrow transliteration
        # edge case it targets - see src/blocking.py's DEFAULT_RULES comment. The key
        # function itself is kept (and tested above) for callers who want it
        # explicitly, e.g. a smaller/non-full-scale blocking run.
        self.assertNotIn("country_name_phonetic", [r.name for r in DEFAULT_RULES])

    def test_builds_a_valid_block_index(self):
        df = pd.DataFrame({
            "name_clean": ["kumar traders", "coomar traders", "apex logistics"],
            "country": ["India", "India", "India"],
        })
        index = build_block_indices(_key_country_name_phonetic(df))
        # rows 0 and 1 (Kumar/Coomar) should land in the same block
        shared_block = [v for v in index.values() if set(v) == {0, 1}]
        self.assertEqual(len(shared_block), 1)


if __name__ == "__main__":
    unittest.main()
