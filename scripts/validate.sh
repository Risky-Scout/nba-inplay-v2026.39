#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 test_v2039.py
python3 tests/test_v2037.py
python3 tests/test_v2038.py

echo
echo "================================================================================"
echo " PORTFOLIO VALIDATION: PASS"
echo " Trader v2026.39 | Engine 2026.29 | Quote policy qp7"
echo "================================================================================"
