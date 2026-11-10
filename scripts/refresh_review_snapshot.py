#!/usr/bin/env python3
"""Generate real D&T snapshots for late-night/current/upcoming Codespaces LIVE REVIEW.

The snapshot spans previous/current/next America/New_York calendar dates so a
game that began before midnight still has its correct game-date priors while the
reviewer can also inspect games scheduled later today or tomorrow.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not os.getenv("DT_API_KEY"):
    sys.exit("DT_API_KEY is not configured; refusing to generate review data")

import nba_markov_engine as mg


def records(df):
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def main():
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    client = mg.DunksAndThreesClient(os.environ["DT_API_KEY"])
    dates = [(now + timedelta(days=d)).strftime("%Y-%m-%d") for d in (-1, 0, 1)]
    by_date = {}

    for date_str in dates:
        epm, box, team = client.fetch_game_day_data(date_str)
        # Keep the date even if D&T has no slate there; the adapter can report it cleanly.
        by_date[date_str] = {
            "epm": records(epm),
            "box_predictions": records(box),
            "team_epm": records(team),
        }
        print(
            f"{date_str}: epm={len(by_date[date_str]['epm'])}, "
            f"box={len(by_date[date_str]['box_predictions'])}, "
            f"team={len(by_date[date_str]['team_epm'])}"
        )

    if not any(v["epm"] and v["box_predictions"] for v in by_date.values()):
        sys.exit("D&T returned no usable data for previous/current/next review dates")

    payload = {
        "schema": "dt-review-snapshot-v2",
        "generated_at": now.isoformat(timespec="seconds"),
        "engine": mg.ENGINE_VERSION,
        "source": "Dunks & Threes API v2 via private GitHub Actions refresher",
        "dates": by_date,
    }
    out = Path(os.environ.get("REVIEW_SNAPSHOT_OUT", "review_data/current.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}: dates={','.join(dates)}")


if __name__ == "__main__":
    main()
