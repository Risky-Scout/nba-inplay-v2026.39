#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/.review_data/current.json}"

mkdir -p "$(dirname "$OUT")"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

cd "$ROOT"

# Codespaces already configures authenticated Git access to the repository.
# Use that instead of requiring the optional GitHub CLI.
if ! git fetch --quiet origin review-data; then
  echo "ERROR: could not fetch the private review-data branch." >&2
  echo "Make sure this Codespace was created from the repository and has repository access." >&2
  exit 2
fi

if ! git show FETCH_HEAD:review_data/current.json > "$TMP"; then
  echo "ERROR: current real D&T review snapshot is unavailable on review-data." >&2
  echo "The repository owner must run the Live Review Data Refresh workflow successfully." >&2
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
