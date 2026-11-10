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
        schema = self._payload.get("schema")
        if schema not in ("dt-review-snapshot-v1", "dt-review-snapshot-v2"):
            raise RuntimeError(f"unsupported review D&T snapshot schema: {schema}")

    @property
    def metadata(self) -> dict:
        if self._payload.get("schema") == "dt-review-snapshot-v2":
            dates = sorted((self._payload.get("dates") or {}).keys())
            return {
                "date": ",".join(dates),
                "generated_at": self._payload.get("generated_at"),
                "engine": self._payload.get("engine"),
                "source": self._payload.get("source"),
            }
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
        if self._payload.get("schema") == "dt-review-snapshot-v2":
            day = (self._payload.get("dates") or {}).get(str(date_str))
            if not day:
                avail = ", ".join(sorted((self._payload.get("dates") or {}).keys())) or "none"
                raise RuntimeError(
                    f"review snapshot has no D&T data for {date_str}; available dates: {avail}"
                )
            return (
                self._frame(day.get("epm")),
                self._frame(day.get("box_predictions")),
                self._frame(day.get("team_epm")),
            )

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
