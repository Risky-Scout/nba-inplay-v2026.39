#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m py_compile *.py tests/*.py
python3 tests/test_secret_audit.py
python3 tests/test_research_validation.py
python3 test_v2039.py
python3 tests/test_v2037.py
python3 tests/test_v2038.py

echo
echo "================================================================================"
echo " PORTFOLIO VALIDATION: PASS"
echo " Trader v2026.40 | Engine 2026.32 | Quote policy qp8"
echo "================================================================================"
