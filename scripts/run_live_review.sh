#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "================================================================================"
echo " NBA IN-PLAY PRICING & MARKET-MAKING ENGINE — LIVE REVIEW"
echo " Real current NBA state + real current public Kalshi markets + real D&T snapshot"
echo " PAPER EXECUTION ONLY — this script cannot submit real-money orders"
echo "================================================================================"

SNAPSHOT="$ROOT/.review_data/current.json"
bash "$ROOT/scripts/fetch_review_snapshot.sh" "$SNAPSHOT" >/dev/null

exec python3 kalshi_nba_inplay_trader.py   --review-dt-snapshot "$SNAPSHOT"   --sims 8000   --shadow-sims 2000   --model-weight 0.35   --no-books
