#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/.review_data/current.json}"
REPO="${GITHUB_REPOSITORY:-Risky-Scout/nba-inplay-v2026.39}"

mkdir -p "$(dirname "$OUT")"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI is required to retrieve the private live-review snapshot." >&2
  exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI is not authenticated. In Codespaces this should be automatic." >&2
  exit 2
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if ! gh api   -H "Accept: application/vnd.github+json"   "/repos/$REPO/contents/review_data/current.json?ref=review-data"   --jq '.content' | tr -d '\n' | base64 --decode > "$TMP"; then
  echo "ERROR: current real D&T review snapshot is unavailable." >&2
  echo "The repository owner must configure the DT_API_KEY Actions secret and run the Live Review Data Refresh workflow." >&2
  exit 3
fi

python3 - "$TMP" <<'PY'
import json, sys
from datetime import datetime
from zoneinfo import ZoneInfo
p=json.load(open(sys.argv[1]))
required=("schema","date","generated_at","epm","box_predictions","team_epm")
missing=[k for k in required if k not in p]
if missing:
    raise SystemExit("invalid review snapshot; missing: "+", ".join(missing))
today=datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
if p["date"] != today:
    raise SystemExit(f"review snapshot is stale: {p['date']} (today is {today})")
print(f"Live review data: {p['date']} | refreshed {p['generated_at']} | EPM {len(p['epm'])} | box {len(p['box_predictions'])}")
PY

mv "$TMP" "$OUT"
trap - EXIT
echo "$OUT"
