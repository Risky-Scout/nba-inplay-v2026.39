#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "================================================================================"
echo " NBA MARKET-MAKER REVIEW AUDIT"
echo " Settled calibration + counterfactual opportunity audit + gate attribution"
echo "================================================================================"

python3 hiring_manager_audit.py "$@"

echo
echo "Open review_outputs/ for the Excel workbook, Markdown summary, and JSON summary."
