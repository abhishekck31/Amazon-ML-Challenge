"""
Phase 1 - Reusable EDA utilities for Amazon ML Challenge 2026: Business Entity Resolution.

Every function takes a {name: DataFrame} dict (as returned by src.data_loader) and
returns a plain DataFrame, so results can be inspected directly in a notebook or
assembled into a written report via generate_eda_report(). No modeling here - Phase 1
is exploration only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def dataset_summary(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per dataset: row/column count, column names, dtypes."""
    rows = []
    for name, df in datasets.items():
        rows.append({
            "dataset": name,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "columns": ", ".join(df.columns),
            "dtypes": ", ".join(f"{c}:{t}" for c, t in df.dtypes.items()),
        })
    return pd.DataFrame(rows)


def missing_value_report(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-format missing-value count + percentage, per dataset per column."""
    rows = []
    for name, df in datasets.items():
        n = len(df)
        for col, count in df.isnull().sum().items():
            rows.append({
                "dataset": name,
                "column": col,
                "n_missing": int(count),
                "pct_missing": round(100 * count / n, 3) if n else 0.0,
            })
    return pd.DataFrame(rows)


def duplicate_report(datasets: Dict[str, pd.DataFrame], subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Duplicate row counts per dataset. Pass subset=["entity_id"] to check key-level dupes instead of full rows."""
    rows = []
    for name, df in datasets.items():
        n_dupes = int(df.duplicated(subset=subset).sum())
        rows.append({
            "dataset": name,
            "n_duplicate_rows": n_dupes,
            "pct_duplicate": round(100 * n_dupes / len(df), 3) if len(df) else 0.0,
        })
    return pd.DataFrame(rows)


def country_distribution(datasets: Dict[str, pd.DataFrame], top_n: int = 15) -> pd.DataFrame:
    """Top-N country value counts for every dataset that has a 'country' column."""
    rows = []
    for name, df in datasets.items():
        if "country" not in df.columns:
            continue
        counts = df["country"].value_counts(dropna=False).head(top_n)
        for country, count in counts.items():
            rows.append({
                "dataset": name,
                "country": country if pd.notna(country) else "<missing>",
                "count": int(count),
                "pct": round(100 * count / len(df), 3),
            })
    return pd.DataFrame(rows)


def sample_records(datasets: Dict[str, pd.DataFrame], n: int = 5) -> Dict[str, pd.DataFrame]:
    """First n rows of every dataset, for a quick visual sanity check."""
    return {name: df.head(n) for name, df in datasets.items()}


def generate_eda_report(
    datasets: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    top_n_countries: int = 15,
    sample_n: int = 5,
) -> str:
    """
    Run dataset_summary / missing_value_report / duplicate_report /
    country_distribution / sample_records and assemble them into one Markdown
    report string. Writes it to `output_path` if given.
    """
    summary = dataset_summary(datasets)
    missing = missing_value_report(datasets)
    dupes = duplicate_report(datasets)
    countries = country_distribution(datasets, top_n=top_n_countries)
    samples = sample_records(datasets, n=sample_n)

    lines = ["# EDA Report - Amazon ML Challenge 2026 (Business Entity Resolution)", ""]

    lines += ["## Dataset Summary", "", "```", summary.to_string(index=False), "```", ""]
    lines += ["## Missing Value Report", "", "```", missing.to_string(index=False), "```", ""]
    lines += ["## Duplicate Report", "", "```", dupes.to_string(index=False), "```", ""]
    lines += [f"## Country Distribution (top {top_n_countries})", "", "```", countries.to_string(index=False), "```", ""]

    lines += ["## Sample Records", ""]
    for name, sample_df in samples.items():
        lines += [f"### {name}", "", "```", sample_df.to_string(index=False), "```", ""]

    report = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

    return report
