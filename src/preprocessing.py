"""
Phase 2 - Text normalization for Amazon ML Challenge 2026: Business Entity Resolution.

Turns raw business_name / business_address strings into normalized name_clean /
address_clean values, so that superficial differences (casing, punctuation, "&" vs
"and", legal-entity suffixes, extra whitespace) stop being treated as real
differences by blocking (src/blocking.py) and downstream matching.
"""

from __future__ import annotations

import re

import pandas as pd

# Legal-entity suffixes to strip from business names. Matched as whole words after
# punctuation has already been stripped, so "Pvt." / "Pvt," / "(Pvt)" all collapse to
# the same "pvt" token before this pattern runs.
LEGAL_SUFFIXES = [
    "private", "pvt", "limited", "ltd", "corporation", "corp", "inc", "llc", "co",
]

LEGAL_SUFFIX_PATTERN = r"\b(" + "|".join(LEGAL_SUFFIXES) + r")\b"
PUNCTUATION_PATTERN = r"[^a-z0-9\s]"
MULTI_SPACE_PATTERN = r"\s+"

_LEGAL_SUFFIX_RE = re.compile(LEGAL_SUFFIX_PATTERN)
_PUNCTUATION_RE = re.compile(PUNCTUATION_PATTERN)
_MULTI_SPACE_RE = re.compile(MULTI_SPACE_PATTERN)


def normalize_text(value, remove_legal_suffixes: bool = False) -> str:
    """
    Normalize a single string: lowercase, "&" -> "and", drop punctuation, optionally
    strip legal-entity suffixes, collapse whitespace. Missing values (None, NaN, or
    any other non-string) safely become "" instead of raising.
    """
    if not isinstance(value, str):
        # Covers None, float NaN, pd.NA and any other non-string input.
        return ""

    text = value.lower()
    text = text.replace("&", " and ")
    text = _PUNCTUATION_RE.sub(" ", text)
    if remove_legal_suffixes:
        text = _LEGAL_SUFFIX_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(value) -> str:
    """Normalize a business name: full cleaning including legal-suffix removal."""
    return normalize_text(value, remove_legal_suffixes=True)


def normalize_address(value) -> str:
    """
    Normalize a business address: same cleaning as normalize_name, but does NOT strip
    legal-entity suffixes - "co" is a common county-road abbreviation in addresses
    ("Co Rd 42"), so stripping it there would destroy real address content rather
    than remove noise.
    """
    return normalize_text(value, remove_legal_suffixes=False)


def normalize_name_series(s: pd.Series) -> pd.Series:
    """Vectorized normalize_name over a whole column - use this instead of .apply() at scale."""
    text = s.fillna("").astype(str).str.lower()
    text = text.str.replace("&", " and ", regex=False)
    text = text.str.replace(PUNCTUATION_PATTERN, " ", regex=True)
    text = text.str.replace(LEGAL_SUFFIX_PATTERN, " ", regex=True)
    text = text.str.replace(MULTI_SPACE_PATTERN, " ", regex=True).str.strip()
    return text


def normalize_address_series(s: pd.Series) -> pd.Series:
    """Vectorized normalize_address over a whole column."""
    text = s.fillna("").astype(str).str.lower()
    text = text.str.replace("&", " and ", regex=False)
    text = text.str.replace(PUNCTUATION_PATTERN, " ", regex=True)
    text = text.str.replace(MULTI_SPACE_PATTERN, " ", regex=True).str.strip()
    return text


def add_clean_columns(
    df: pd.DataFrame,
    name_col: str = "business_name",
    address_col: str = "business_address",
) -> pd.DataFrame:
    """Return a copy of df with name_clean / address_clean columns added (vectorized)."""
    return df.assign(
        name_clean=normalize_name_series(df[name_col]),
        address_clean=normalize_address_series(df[address_col]),
    )
