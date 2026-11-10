#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/.review_data/current.json}"

mkdir -p "$(dirname "$OUT")"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

cd "$ROOT"

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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
p=json.load(open(sys.argv[1]))
schema=p.get("schema")
if schema == "dt-review-snapshot-v2":
    required=("schema","generated_at","engine","source","dates")
    missing=[k for k in required if k not in p]
    if missing:
        raise SystemExit("invalid review snapshot; missing: "+", ".join(missing))
    now=datetime.now(ZoneInfo("America/New_York"))
    wanted={(now+timedelta(days=d)).strftime("%Y-%m-%d") for d in (-1,0,1)}
    have=set((p.get("dates") or {}).keys())
    if not wanted.issubset(have):
        raise SystemExit(f"review snapshot date coverage incomplete: have {sorted(have)}, need {sorted(wanted)}")
    print(f"Live review data: dates={','.join(sorted(have))} | refreshed {p['generated_at']}")
elif schema == "dt-review-snapshot-v1":
    required=("schema","date","generated_at","epm","box_predictions","team_epm")
    missing=[k for k in required if k not in p]
    if missing:
        raise SystemExit("invalid review snapshot; missing: "+", ".join(missing))
    print(f"Live review data (legacy single-date): {p['date']} | refreshed {p['generated_at']}")
else:
    raise SystemExit(f"unsupported review snapshot schema: {schema}")
PY

mv "$TMP" "$OUT"
trap - EXIT
echo "$OUT"
