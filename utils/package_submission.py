"""
Automated submission packager for the Amazon ML Challenge 2026.

Builds <team_name>_submission.zip in the exact required layout:

    <team_name>_submission.zip
    |-- output/
    |   |-- matching_results.tsv       (scored on leaderboard)
    |   `-- candidate_pairs.tsv        (candidate set fed to the model pre-threshold)
    |-- code/
    |   `-- business_entity_resolution/
    |       |-- src/                   (all source code)
    |       |-- README.md              (reproduction instructions)
    |       `-- requirements.txt       (pinned dependencies)
    `-- Documentation_template.md      (methodology write-up)

Runs utils/validate_submission.py first (unless --skip-validation) so a malformed
output file never makes it into the zip - catching a format problem here is free;
catching it after submitting costs one of the 5-per-day slots.

Usage:
    python utils/package_submission.py --team-name my_team
    python utils/package_submission.py --team-name my_team --output-dir dist --skip-validation
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.validate_submission import validate_submission  # noqa: E402

# Defensive filter applied while walking src/ for inclusion - src/ shouldn't normally
# contain any of these, but a fresh checkout with stray __pycache__/.pyc files from a
# local test run, or a misplaced venv/dataset copy, must never end up in the zip.
EXCLUDED_PATTERNS = [
    ".git", "*.git/*",
    "__pycache__", "*__pycache__*",
    "*.pyc",
    "venv", "*venv/*",
    "dataset", "*dataset/*",
    "models/*.pth",
    ".idea", "*.idea/*",
    ".vscode", "*.vscode/*",
]


def _is_excluded(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    parts = normalized.split("/")
    for pattern in EXCLUDED_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern):
            return True
        bare = pattern.lstrip("*").split("/")[0]
        if bare and not any(ch in bare for ch in "*?[]") and bare in parts:
            return True
    return False


def _iter_included_files(root: Path) -> List[Path]:
    """Every file under `root`, excluding anything matching EXCLUDED_PATTERNS."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_excluded(rel):
            continue
        files.append(path)
    return files


def _require(path: Path, label: str) -> None:
    if not path.exists():
        print(f"ERROR: required file/directory missing: {label} ({path})")
        sys.exit(1)


def build_submission_zip(
    team_name: str,
    output_dir: str = ".",
    matching_results_path: str = "output/matching_results.tsv",
    candidate_pairs_path: str = "output/candidate_pairs.tsv",
    source1_path: str = "dataset/test/test_source1.tsv",
    skip_validation: bool = False,
) -> Path:
    matching_results = PROJECT_ROOT / matching_results_path
    candidate_pairs = PROJECT_ROOT / candidate_pairs_path

    if not skip_validation:
        print("Running utils/validate_submission.py checks on the output files...")
        errors = validate_submission(str(matching_results), str(candidate_pairs), str(PROJECT_ROOT / source1_path))
        if errors:
            print(f"\nFAIL - {len(errors)} problem(s) found; refusing to package a broken submission:\n")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        print("PASS - matching_results.tsv / candidate_pairs.tsv validated OK.\n")
    else:
        print("Skipping validation (--skip-validation passed).\n")
        _require(matching_results, matching_results_path)
        _require(candidate_pairs, candidate_pairs_path)

    readme_path = PROJECT_ROOT / "README.md"
    requirements_path = PROJECT_ROOT / "requirements.txt"
    doc_path = PROJECT_ROOT / "Documentation_template.md"
    src_dir = PROJECT_ROOT / "src"

    _require(readme_path, "README.md")
    _require(requirements_path, "requirements.txt")
    _require(doc_path, "Documentation_template.md")
    _require(src_dir, "src/")

    zip_path = Path(output_dir) / f"{team_name}_submission.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    entries: List[Tuple[Path, str]] = [
        (matching_results, "output/matching_results.tsv"),
        (candidate_pairs, "output/candidate_pairs.tsv"),
        (doc_path, "Documentation_template.md"),
        (readme_path, "code/business_entity_resolution/README.md"),
        (requirements_path, "code/business_entity_resolution/requirements.txt"),
    ]
    for src_file in _iter_included_files(src_dir):
        rel = src_file.relative_to(src_dir).as_posix()
        entries.append((src_file, f"code/business_entity_resolution/src/{rel}"))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source_path, arcname in entries:
            zf.write(source_path, arcname)

    _print_audit_report(zip_path, entries)
    return zip_path


def _print_audit_report(zip_path: Path, entries: List[Tuple[Path, str]]) -> None:
    total_uncompressed = sum(p.stat().st_size for p, _ in entries)
    compressed_size = zip_path.stat().st_size

    forbidden_present = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if _is_excluded(name):
                forbidden_present.append(name)

    print("=" * 60)
    print("SUBMISSION PACKAGE AUDIT REPORT")
    print("=" * 60)
    print(f"Zip file:            {zip_path}")
    print(f"Total files:         {len(entries)}")
    print(f"Uncompressed size:   {total_uncompressed / 1e6:.2f} MB")
    print(f"Compressed zip size: {compressed_size / 1e6:.2f} MB")
    if forbidden_present:
        print(f"FORBIDDEN FILES FOUND ({len(forbidden_present)}) - investigate before submitting:")
        for f in forbidden_present:
            print(f"  - {f}")
    else:
        print("Forbidden-file check:  PASS (.git/__pycache__/venv/dataset/etc. all absent)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Amazon ML Challenge 2026 submission zip.")
    parser.add_argument("--team-name", required=True, help="Used as the zip filename: <team_name>_submission.zip")
    parser.add_argument("--output-dir", default=".", help="Directory to write the zip into.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the validate_submission.py pre-check.")
    parser.add_argument("--matching-results", default="output/matching_results.tsv")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv")
    parser.add_argument("--source1", default="dataset/test/test_source1.tsv",
                         help="Used only for validation - the Source1 file the submission should cover.")
    args = parser.parse_args()

    build_submission_zip(
        team_name=args.team_name,
        output_dir=args.output_dir,
        matching_results_path=args.matching_results,
        candidate_pairs_path=args.candidate_pairs,
        source1_path=args.source1,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
