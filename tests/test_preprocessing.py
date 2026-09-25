"""
Unit tests for src/preprocessing.py (Phase 2).

Run with:  python -m unittest tests.test_preprocessing -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.preprocessing import (
    add_clean_columns,
    normalize_address,
    normalize_address_series,
    normalize_name,
    normalize_name_series,
    normalize_text,
    strip_accents,
    strip_accents_series,
)


class TestNormalizeTextBasics(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalize_text("ACME"), "acme")

    def test_removes_punctuation(self):
        self.assertEqual(normalize_text("Acme, Inc.!"), "acme inc")

    def test_ampersand_becomes_and(self):
        self.assertEqual(normalize_text("Smith & Sons"), "smith and sons")

    def test_collapses_multiple_spaces(self):
        self.assertEqual(normalize_text("Acme    Retail   Store"), "acme retail store")

    def test_strips_leading_trailing_whitespace(self):
        self.assertEqual(normalize_text("   Acme   "), "acme")


class TestMissingValueHandling(unittest.TestCase):
    def test_none(self):
        self.assertEqual(normalize_name(None), "")
        self.assertEqual(normalize_address(None), "")

    def test_float_nan(self):
        self.assertEqual(normalize_name(float("nan")), "")

    def test_numpy_nan(self):
        self.assertEqual(normalize_name(np.nan), "")

    def test_pandas_na(self):
        self.assertEqual(normalize_name(pd.NA), "")

    def test_non_string_input(self):
        self.assertEqual(normalize_name(12345), "")

    def test_series_with_missing_values(self):
        s = pd.Series(["Acme Inc", None, np.nan, "  "])
        result = normalize_name_series(s)
        self.assertEqual(list(result), ["acme", "", "", ""])


class TestLegalSuffixRemoval(unittest.TestCase):
    """The core Phase 2 requirement: different real-world variants of the same
    business must normalize to the identical name_clean value."""

    def test_pvt_ltd_variants_collapse_to_same_name(self):
        variants = [
            "XYZ Pvt Ltd",
            "xyz private limited",
            "XYZ PVT. LTD.",
            "XYZ (Pvt.) (Ltd.)",
            "  XYZ   Pvt.   Ltd  ",
        ]
        normalized = {normalize_name(v) for v in variants}
        self.assertEqual(normalized, {"xyz"})

    def test_corp_corporation_inc_llc_variants(self):
        variants = ["Acme Corp", "Acme Corporation", "Acme Inc", "Acme, Inc.", "Acme LLC"]
        normalized = {normalize_name(v) for v in variants}
        self.assertEqual(normalized, {"acme"})

    def test_co_suffix_removed_from_name(self):
        self.assertEqual(normalize_name("Standard Oil Co"), "standard oil")
        self.assertEqual(normalize_name("Standard Oil Co."), "standard oil")

    def test_ampersand_and_suffix_together(self):
        variants = ["Smith & Sons Ltd", "Smith and Sons Limited", "SMITH & SONS, LTD."]
        normalized = {normalize_name(v) for v in variants}
        self.assertEqual(normalized, {"smith and sons"})

    def test_suffix_only_removed_as_whole_word(self):
        # "co" must not match inside "company" / "coworking" etc.
        self.assertEqual(normalize_name("Coworking Spaces Inc"), "coworking spaces")
        self.assertNotIn("worage", normalize_name("Storage Co"))


class TestAddressNormalization(unittest.TestCase):
    def test_address_variants_collapse_to_same_value(self):
        variants = [
            "123 Main St., Apt #4",
            "123   MAIN ST APT 4",
            "123 main st. apt. 4",
        ]
        normalized = {normalize_address(v) for v in variants}
        self.assertEqual(normalized, {"123 main st apt 4"})

    def test_address_does_not_strip_co_as_legal_suffix(self):
        # "Co" in an address is usually a county abbreviation ("Co Rd 42"), not a
        # legal-entity suffix, and must survive normalize_address unlike normalize_name.
        self.assertIn("co", normalize_address("Co Rd 42").split())

    def test_street_designator_synonyms_collapse_to_same_value(self):
        self.assertEqual(normalize_address("123 Main Rd"), normalize_address("123 Main Road"))
        self.assertEqual(normalize_address("1 Elm Ave"), normalize_address("1 Elm Avenue"))
        self.assertEqual(normalize_address("5 Victory Blvd"), normalize_address("5 Victory Boulevard"))

    def test_st_is_not_expanded_due_to_saint_ambiguity(self):
        # "St" is genuinely ambiguous (Street vs. Saint) - guessing wrong would corrupt
        # real address content, so normalize_address must leave it untouched rather
        # than silently assume "Street".
        self.assertEqual(normalize_address("St Louis"), "st louis")
        self.assertNotIn("street", normalize_address("St Louis"))


class TestFrenchAndUnicodeNormalization(unittest.TestCase):
    """France appears only in the test set (never in training data), and country is
    never hardcoded anywhere in this pipeline - these guard the zero-shot path."""

    def test_strip_accents_scalar(self):
        self.assertEqual(strip_accents("Société"), "Societe")
        self.assertEqual(strip_accents("Café"), "Cafe")
        self.assertEqual(strip_accents("Château"), "Chateau")

    def test_strip_accents_series(self):
        result = strip_accents_series(pd.Series(["Société", "Café", None]))
        self.assertEqual(list(result), ["Societe", "Cafe", ""])

    def test_accents_stripped_before_punctuation_removal(self):
        # Without accent-stripping first, the old ASCII-only punctuation regex would
        # delete "é" outright ("soci t ") instead of folding it to "e" ("societe").
        self.assertEqual(normalize_name("Société Générale"), "societe generale")

    def test_french_legal_suffixes_removed(self):
        variants = ["Boulangerie Martin SARL", "Boulangerie Martin SAS", "Boulangerie Martin SA"]
        normalized = {normalize_name(v) for v in variants}
        self.assertEqual(normalized, {"boulangerie martin"})

    def test_french_and_english_suffix_variants_of_same_business_collapse(self):
        # A French-registered version and a US-registered version of a conceptually
        # equivalent legal form should both reduce to the bare business name.
        self.assertEqual(normalize_name("Dupont Consulting EURL"), "dupont consulting")
        self.assertEqual(normalize_name("Dupont Consulting LLC"), "dupont consulting")

    def test_french_address_with_accents_normalizes(self):
        result = normalize_address("12 Rue de l'Église, Boulevard Saint-Michel")
        self.assertNotIn("é", result)
        self.assertIn("eglise", result)
        self.assertIn("boulevard", result)


class TestSeriesVsScalarConsistency(unittest.TestCase):
    """The vectorized *_series functions must match the scalar functions element-wise,
    since blocking.py relies on the vectorized path for performance."""

    def test_name_series_matches_scalar(self):
        values = ["XYZ Pvt Ltd", "Smith & Sons", None, "  Acme  Retail  ", "Société Générale SARL"]
        s = pd.Series(values)
        vectorized = list(normalize_name_series(s))
        scalar = [normalize_name(v) for v in values]
        self.assertEqual(vectorized, scalar)

    def test_address_series_matches_scalar(self):
        values = ["123 Main St., Apt #4", "Co Rd 42", None, "  ", "12 Rue de l'Église, Bd Saint-Michel"]
        s = pd.Series(values)
        vectorized = list(normalize_address_series(s))
        scalar = [normalize_address(v) for v in values]
        self.assertEqual(vectorized, scalar)


class TestAddCleanColumns(unittest.TestCase):
    def test_adds_expected_columns(self):
        df = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["XYZ Pvt Ltd", "Smith & Sons Inc"],
            "business_address": ["123 Main St., Apt #4", None],
            "country": ["US", "US"],
        })
        result = add_clean_columns(df)
        self.assertEqual(list(result["name_clean"]), ["xyz", "smith and sons"])
        self.assertEqual(list(result["address_clean"]), ["123 main st apt 4", ""])
        # original columns/data must be untouched
        self.assertEqual(list(result["business_name"]), list(df["business_name"]))

    def test_does_not_mutate_input_df(self):
        df = pd.DataFrame({
            "business_name": ["Acme Inc"],
            "business_address": ["1 Main St"],
        })
        add_clean_columns(df)
        self.assertNotIn("name_clean", df.columns)


if __name__ == "__main__":
    unittest.main()
