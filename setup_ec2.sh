#!/usr/bin/env bash
#
# setup_ec2.sh - one-click bootstrap for the Amazon ML Challenge 2026 pipeline on a
# fresh Ubuntu 22.04/24.04 LTS EC2 instance.
#
# Usage (from the project root, after cloning/uploading the repo):
#   chmod +x setup_ec2.sh
#   ./setup_ec2.sh
#
# Halts immediately on any failure (including a mid-pipe failure, e.g. a failing test
# run piped through `tee`), so a broken environment is never silently left half-set-up.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "=================================================="
echo "Amazon ML Challenge 2026 - EC2 environment setup"
echo "=================================================="

echo ""
echo "--- Updating apt and installing system packages ---"
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    htop \
    tmux \
    unzip \
    build-essential

echo ""
echo "--- Creating Python virtual environment (venv) ---"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

echo ""
echo "--- Upgrading pip ---"
pip install --upgrade pip

echo ""
echo "--- Installing project dependencies from requirements.txt ---"
pip install -r requirements.txt

echo ""
echo "--- Running unit test suite (python -m unittest discover tests) ---"
UNITTEST_LOG="$(mktemp)"
python -m unittest discover tests -v 2>&1 | tee "$UNITTEST_LOG"

if ! grep -q "^OK$" "$UNITTEST_LOG"; then
    echo ""
    echo "ERROR: unit tests did not all pass - see output above. Aborting setup."
    exit 1
fi

TEST_COUNT="$(grep -Eo 'Ran [0-9]+ test' "$UNITTEST_LOG" | grep -Eo '[0-9]+' || echo unknown)"
echo ""
echo "All $TEST_COUNT unit tests passed."
rm -f "$UNITTEST_LOG"

echo ""
echo "=================================================="
echo "System specifications"
echo "=================================================="
echo "CPU model:  $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
echo "vCPUs:      $(nproc)"
echo "Total RAM:  $(free -h | awk '/^Mem:/ {print $2}')"
echo "=================================================="

echo ""
echo "=================================================="
echo "Setup complete. Next steps:"
echo "=================================================="
echo "  1. In every new shell, activate the environment first:"
echo "       source venv/bin/activate"
echo ""
echo "  2. Place the competition dataset under:"
echo "       dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv"
echo "       dataset/test/{test_source1,test_source2,test_source3}.tsv"
echo ""
echo "  3. Run the pipeline end to end:"
echo "       python -m src.blocking --data-dir dataset/train --split train --output output/candidate_pairs.tsv"
echo "       python -m src.training_data --data-dir dataset/train --candidate-pairs output/candidate_pairs.tsv"
echo "       python -m src.train"
echo "       python -m src.threshold_search --mode dual"
echo "       python -m src.inference --data-dir dataset/test --split test"
echo ""
echo "  4. Validate and package a submission:"
echo "       python utils/validate_submission.py"
echo "       python utils/package_submission.py --team-name <your_team_name>"
echo "=================================================="
