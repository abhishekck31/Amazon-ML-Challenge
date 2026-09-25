"""
Phase 2 - Text normalization for Amazon ML Challenge 2026: Business Entity Resolution.

Turns raw business_name / business_address strings into normalized name_clean /
address_clean values, so that superficial differences (casing, punctuation, "&" vs
"and", legal-entity suffixes, accents, extra whitespace) stop being treated as real
differences by blocking (src/blocking.py) and downstream matching.

France zero-shot note: training data (Source1/2/3) only contains US and India: the
test set adds France. Two things in this module exist specifically so the pipeline
generalizes to that unseen country instead of silently breaking on it:

1. strip_accents() / strip_accents_series() run BEFORE the ASCII-only punctuation
   regex. Without this, "Société" or "Café" would have their accented letters treated
   as punctuation and deleted outright ("soci t ", "caf ") instead of being folded to
   their unaccented form ("societe", "cafe") - silently destroying the very characters
   that identify a French business name, precisely because country is never hardcoded
   or branched on anywhere in this pipeline.
2. LEGAL_SUFFIXES includes French (and a few other common) legal-entity forms
   (sarl, sas, sasu, sa, eurl, sci, snc, gie, cie, gmbh) alongside the original
   US/India-oriented ones, so normalize_name() strips them the same way it already
   strips "Pvt Ltd" / "Inc" / "LLC".
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Legal-entity suffixes to strip from business names. Matched as whole words after
# punctuation has already been stripped, so "Pvt." / "Pvt," / "(Pvt)" all collapse to
# the same "pvt" token before this pattern runs.
LEGAL_SUFFIXES = [
    # English / generic / India
    "private", "pvt", "limited", "ltd", "corporation", "corp", "inc", "llc", "co",
    # French (test set introduces France; never seen in training data)
    "sarl", "sasu", "sas", "sa", "eurl", "sci", "snc", "gie", "cie", "association",
    # Other common international forms, low-risk to include alongside the above
    "gmbh",
]

# "sa" and "co" are short enough to theoretically collide with real word fragments,
# but \b...\b requires a whole-word match, so e.g. "sand" or "coast" are unaffected.
LEGAL_SUFFIX_PATTERN = r"\b(" + "|".join(LEGAL_SUFFIXES) + r")\b"
PUNCTUATION_PATTERN = r"[^a-z0-9\s]"
MULTI_SPACE_PATTERN = r"\s+"

_LEGAL_SUFFIX_RE = re.compile(LEGAL_SUFFIX_PATTERN)
_PUNCTUATION_RE = re.compile(PUNCTUATION_PATTERN)
_MULTI_SPACE_RE = re.compile(MULTI_SPACE_PATTERN)

# Street-designator synonyms for address_clean, so "123 Main Rd" and "123 Main Road"
# block/match together. Deliberately does NOT include "st" -> "street": "St" is
# genuinely ambiguous in English addresses ("St" for "Street" vs. "St" for "Saint", as
# in "St Louis"), and guessing wrong would corrupt real address content instead of
# normalizing noise - so it is left untouched rather than risk that.
STREET_SYNONYMS = {
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",  # French abbreviation for boulevard
    "rte": "route",
    "chem": "chemin",   # French for "road/way" (e.g. "Chemin des Vignes")
}
STREET_SYNONYM_PATTERN = r"\b(" + "|".join(STREET_SYNONYMS.keys()) + r")\b"
_STREET_SYNONYM_RE = re.compile(STREET_SYNONYM_PATTERN)


def strip_accents(text: str) -> str:
    """Unicode NFKD-decompose and drop combining diacritical marks: 'Société' -> 'Societe',
    'Café' -> 'Cafe'. Case-preserving; run before lowercasing (order doesn't matter, but
    this module always does it first for clarity)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def strip_accents_series(s: pd.Series) -> pd.Series:
    """Vectorized strip_accents over a whole column, via pandas' str.normalize (wraps
    unicodedata.normalize) + an ASCII round-trip to drop the now-isolated combining marks."""
    return (
        s.fillna("").astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
    )


def normalize_text(value, remove_legal_suffixes: bool = False, expand_street_synonyms: bool = False) -> str:
    """
    Normalize a single string: strip accents, lowercase, "&" -> "and", drop
    punctuation, optionally strip legal-entity suffixes and/or expand street-designator
    abbreviations, collapse whitespace. Missing values (None, NaN, or any other
    non-string) safely become "" instead of raising.
    """
    if not isinstance(value, str):
        # Covers None, float NaN, pd.NA and any other non-string input.
        return ""

    text = strip_accents(value)
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCTUATION_RE.sub(" ", text)
    if remove_legal_suffixes:
        text = _LEGAL_SUFFIX_RE.sub(" ", text)
    if expand_street_synonyms:
        text = _STREET_SYNONYM_RE.sub(lambda m: STREET_SYNONYMS[m.group(0)], text)
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
    than remove noise. Does expand street-designator abbreviations (rd/ave/blvd/...).
    """
    return normalize_text(value, remove_legal_suffixes=False, expand_street_synonyms=True)


def normalize_name_series(s: pd.Series) -> pd.Series:
    """Vectorized normalize_name over a whole column - use this instead of .apply() at scale."""
    text = strip_accents_series(s).str.lower()
    text = text.str.replace("&", " and ", regex=False)
    text = text.str.replace(PUNCTUATION_PATTERN, " ", regex=True)
    text = text.str.replace(LEGAL_SUFFIX_PATTERN, " ", regex=True)
    text = text.str.replace(MULTI_SPACE_PATTERN, " ", regex=True).str.strip()
    return text


def normalize_address_series(s: pd.Series) -> pd.Series:
    """Vectorized normalize_address over a whole column."""
    text = strip_accents_series(s).str.lower()
    text = text.str.replace("&", " and ", regex=False)
    text = text.str.replace(PUNCTUATION_PATTERN, " ", regex=True)
    text = text.str.replace(STREET_SYNONYM_PATTERN, lambda m: STREET_SYNONYMS[m.group(0)], regex=True)
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
