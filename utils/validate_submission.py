"""
Local pre-flight validator for the Amazon ML Challenge 2026 submission files.

Checks every explicit format rule from the problem statement BEFORE you burn one of
the 5-per-day submission slots on a formatting mistake:

* Both files are tab-delimited TSV with the exact expected header.
* matching_results.tsv covers every Source1 entity in the target source1 file exactly
  once (no missing rows, no duplicates, no unexpected extras).
* candidate_pairs.tsv has the same one-row-per-entity coverage.
* Every candidate id list is deduplicated and sorted (as the spec requires), with no
  duplicate ids inside a single row.
* "Trap 5": every id in a matching_results.tsv row is a subset of that same entity's
  candidate_pairs.tsv row.

Usage:
    python utils/validate_submission.py \\
        --matching-results output/matching_results.tsv \\
        --candidate-pairs output/candidate_pairs.tsv \\
        --source1 dataset/test/test_source1.tsv

Prints PASS and exits 0 if every check succeeds. Otherwise prints every failure found
(not just the first) and exits 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

MATCHING_RESULTS_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_PAIRS_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def _check_raw_tab_delimited(path: Path, expected_fields: int, errors: List[str]) -> None:
    """Confirm every non-header line has exactly `expected_fields - 1` tabs, at the
    raw-text level - a belt-and-suspenders check independent of how pandas parsed it."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        errors.append(f"{path}: file is empty")
        return
    for i, line in enumerate(lines[1:], start=2):  # skip header, 1-indexed line numbers
        if not line:
            continue
        if line.count("\t") != expected_fields - 1:
            errors.append(f"{path}: line {i} does not have exactly {expected_fields} tab-delimited fields")


def _check_header(path: Path, df: pd.DataFrame, expected_columns: List[str], errors: List[str]) -> None:
    if list(df.columns) != expected_columns:
        errors.append(f"{path}: header is {list(df.columns)}, expected {expected_columns}")


def _check_coverage(
    path: Path, df: pd.DataFrame, id_col: str, expected_ids: set, errors: List[str]
) -> None:
    actual_ids = set(df[id_col].dropna())
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        errors.append(f"{path}: {len(missing):,} Source1 entities missing (e.g. {sample})")
    if extra:
        sample = ", ".join(sorted(extra)[:5])
        errors.append(f"{path}: {len(extra):,} unexpected entity_ids present (e.g. {sample})")
    if df[id_col].duplicated().any():
        n_dupes = int(df[id_col].duplicated().sum())
        errors.append(f"{path}: {n_dupes:,} duplicate {id_col} rows")


def _check_ids_deduplicated_and_sorted(path: Path, df: pd.DataFrame, id_col: str, list_col: str, errors: List[str]) -> None:
    bad_dupes = 0
    bad_order = 0
    for sid, ids_str in zip(df[id_col], df[list_col]):
        if not isinstance(ids_str, str) or not ids_str:
            continue
        ids = ids_str.split(",")
        if len(ids) != len(set(ids)):
            bad_dupes += 1
        if ids != sorted(ids):
            bad_order += 1
    if bad_dupes:
        errors.append(f"{path}: {bad_dupes:,} rows have duplicate ids within their {list_col} list")
    if bad_order:
        errors.append(f"{path}: {bad_order:,} rows have an unsorted {list_col} list")


def _check_subset(
    matching_results: pd.DataFrame, candidate_pairs: pd.DataFrame, errors: List[str]
) -> None:
    candidate_map = dict(zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_ids"]))
    violations = 0
    for sid, matched_str in zip(matching_results["source1_entity_id"], matching_results["matched_entity_ids"]):
        if not isinstance(matched_str, str) or not matched_str:
            continue
        matched_ids = set(matched_str.split(","))
        candidate_str = candidate_map.get(sid, "")
        candidate_ids = set(candidate_str.split(",")) if isinstance(candidate_str, str) and candidate_str else set()
        if not matched_ids.issubset(candidate_ids):
            violations += 1
    if violations:
        errors.append(
            f"matching_results.tsv: {violations:,} rows have id(s) NOT present in the "
            f"same entity's candidate_pairs.tsv row (Trap 5: Candidate Mismatch Error)"
        )


def validate_submission(
    matching_results_path: str,
    candidate_pairs_path: str,
    source1_path: str,
) -> List[str]:
    """Run every check and return the list of failures (empty list = PASS)."""
    errors: List[str] = []

    mr_path, cp_path, s1_path = Path(matching_results_path), Path(candidate_pairs_path), Path(source1_path)
    for p in (mr_path, cp_path, s1_path):
        if not p.exists():
            errors.append(f"{p}: file does not exist")
    if errors:
        return errors

    matching_results = pd.read_csv(mr_path, sep="\t", dtype=str)
    candidate_pairs = pd.read_csv(cp_path, sep="\t", dtype=str)
    source1 = pd.read_csv(s1_path, sep="\t", dtype=str)

    _check_raw_tab_delimited(mr_path, 2, errors)
    _check_raw_tab_delimited(cp_path, 2, errors)

    _check_header(mr_path, matching_results, MATCHING_RESULTS_COLUMNS, errors)
    _check_header(cp_path, candidate_pairs, CANDIDATE_PAIRS_COLUMNS, errors)

    expected_ids = set(source1["entity_id"])
    _check_coverage(mr_path, matching_results, "source1_entity_id", expected_ids, errors)
    _check_coverage(cp_path, candidate_pairs, "source1_entity_id", expected_ids, errors)

    _check_ids_deduplicated_and_sorted(mr_path, matching_results, "source1_entity_id", "matched_entity_ids", errors)
    _check_ids_deduplicated_and_sorted(cp_path, candidate_pairs, "source1_entity_id", "candidate_entity_ids", errors)

    # Subset check only makes sense once both files individually parse and cover the
    # expected entities - skip it if earlier checks already found structural problems.
    if not errors:
        _check_subset(matching_results, candidate_pairs, errors)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate matching_results.tsv / candidate_pairs.tsv before submission.")
    parser.add_argument("--matching-results", default="output/matching_results.tsv")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv")
    parser.add_argument("--source1", default="dataset/test/test_source1.tsv")
    args = parser.parse_args()

    errors = validate_submission(args.matching_results, args.candidate_pairs, args.source1)

    if not errors:
        print("PASS - all submission format checks passed.")
        sys.exit(0)

    print(f"FAIL - {len(errors)} problem(s) found:\n")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
