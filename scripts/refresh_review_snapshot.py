#!/usr/bin/env python3
"""Generate the real current D&T snapshot used by Codespaces LIVE REVIEW mode.

This script is intended to run only inside the repository owner's GitHub Actions
workflow with DT_API_KEY supplied as an Actions secret. It never writes the key.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime
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
    # JSON round-trip normalizes numpy/pandas scalar types cleanly.
    return json.loads(df.to_json(orient="records", date_format="iso"))


def main():
    date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    client = mg.DunksAndThreesClient(os.environ["DT_API_KEY"])
    epm, box, team = client.fetch_game_day_data(date_str)
    if epm.empty or box.empty:
        sys.exit(f"D&T returned no usable current data for {date_str}")

    payload = {
        "schema": "dt-review-snapshot-v1",
        "date": date_str,
        "generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "engine": mg.ENGINE_VERSION,
        "source": "Dunks & Threes API v2 via private GitHub Actions refresher",
        "epm": records(epm),
        "box_predictions": records(box),
        "team_epm": records(team),
    }
    out = Path(os.environ.get("REVIEW_SNAPSHOT_OUT", "review_data/current.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"wrote {out}: date={date_str}, epm={len(payload['epm'])}, "
        f"box={len(payload['box_predictions'])}, team={len(payload['team_epm'])}"
    )


if __name__ == "__main__":
    main()
