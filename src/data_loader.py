"""
Phase 1 - Data loading utilities for Amazon ML Challenge 2026: Business Entity Resolution.

Centralizes TSV file paths and read options so notebooks and every later pipeline
stage (blocking, feature engineering, modeling) share one source of truth instead of
each hardcoding "../dataset/train/..." paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_FILES: Dict[str, Path] = {
    "train_s1": TRAIN_DIR / "train_source1.tsv",
    "train_s2": TRAIN_DIR / "train_source2.tsv",
    "train_s3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}

TEST_FILES: Dict[str, Path] = {
    "test_s1": TEST_DIR / "test_source1.tsv",
    "test_s2": TEST_DIR / "test_source2.tsv",
    "test_s3": TEST_DIR / "test_source3.tsv",
}


def load_tsv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read one competition TSV file with the shared convention (tab-separated)."""
    return pd.read_csv(path, sep="\t", nrows=nrows)


def load_train(nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Load source1/2/3 + ground truth for the train split into a name -> DataFrame dict."""
    return {name: load_tsv(path, nrows=nrows) for name, path in TRAIN_FILES.items()}


def load_test(nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Load source1/2/3 for the test split into a name -> DataFrame dict."""
    return {name: load_tsv(path, nrows=nrows) for name, path in TEST_FILES.items()}


def load_all(nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Load every train and test file into one combined name -> DataFrame dict."""
    return {**load_train(nrows=nrows), **load_test(nrows=nrows)}
