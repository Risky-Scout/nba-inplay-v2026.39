"""Read-only D&T snapshot adapter for hiring-manager LIVE REVIEW mode.

The snapshot is generated privately by GitHub Actions from the repository owner's
D&T credential. The reviewer never receives the credential. This adapter does
not fabricate or alter rows; it reconstructs the exact DataFrames captured by
the refresher.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


class ReviewDunksAndThreesClient:
    def __init__(self, snapshot_path: str):
        self.path = Path(snapshot_path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"review D&T snapshot not found: {self.path}")
        self._payload = json.loads(self.path.read_text())
        if self._payload.get("schema") != "dt-review-snapshot-v1":
            raise RuntimeError("unsupported review D&T snapshot schema")

    @property
    def metadata(self) -> dict:
        return {
            "date": self._payload.get("date"),
            "generated_at": self._payload.get("generated_at"),
            "engine": self._payload.get("engine"),
            "source": self._payload.get("source"),
        }

    @staticmethod
    def _frame(records) -> pd.DataFrame:
        return pd.DataFrame(records or [])

    def fetch_game_day_data(self, date_str: str):
        snap_date = str(self._payload.get("date") or "")
        if snap_date != str(date_str):
            raise RuntimeError(
                f"review snapshot is for {snap_date or 'UNKNOWN'}, but the trader requested {date_str}; "
                "refresh review-data before running live review"
            )
        return (
            self._frame(self._payload.get("epm")),
            self._frame(self._payload.get("box_predictions")),
            self._frame(self._payload.get("team_epm")),
        )
