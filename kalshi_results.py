"""Settlement results for Kalshi markets (public endpoint, no key), cached in settlements.csv.

Used by grade_predictions.py and fit_calibration.py. Tries both Kalshi API hosts and reports the first
failure clearly instead of silently returning nothing.
"""
import os
import glob
import time
import requests
import numpy as np
import pandas as pd

IDENTITY_DTYPES = {
    "engine": "string", "engine_version": "string", "cal_id": "string",
    "policy_id": "string", "policy_family_id": "string",
    "exec_model_id": "string", "exec_prior_id": "string",
    "run_id": "string", "order_id": "string", "client_order_id": "string",
    "quote_policy": "string", "fit_id": "string", "schema_version": "string",
    "ticker": "string",
}

def read_log_csv(path, **kwargs):
    """Read evidence logs with opaque identity fields protected before dtype inference.

    Version identifiers are lexical strings: 2026.3 and 2026.30 are distinct identities.
    This prevents pandas from silently turning an original CSV token 2026.30 into 2026.3.
    """
    supplied = kwargs.pop("dtype", None)
    if supplied is not None and not isinstance(supplied, dict):
        raise TypeError("read_log_csv requires dtype to be a mapping")
    dtype = dict(supplied or {})
    dtype.update(IDENTITY_DTYPES)
    return pd.read_csv(path, dtype=dtype, **kwargs)


HOSTS = ["https://api.elections.kalshi.com/trade-api/v2", "https://external-api.kalshi.com/trade-api/v2"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settlements.csv")


def load_predictions(paths=None) -> pd.DataFrame:
    """All prediction logs (or the given ones), with pre-tip rows removed.

    Before tip-off the scoreboard clock does not move. Rows logged before a game's clock first changes are
    dropped: they were priced on a frozen pre-game clock (the Nov 5 pre-game clock bug lived exactly there).
    """
    paths = paths or sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi_predictions_*.csv")))
    if not paths:
        raise SystemExit("No kalshi_predictions_*.csv files found in this folder.")
    frames = []
    for p in paths:
        d = read_log_csv(p)
        d["source_file"] = os.path.basename(p)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "p_model_raw" not in df.columns:
        df["p_model_raw"] = df["p_model_yes"]
    df["p_model_raw"] = df["p_model_raw"].fillna(df["p_model_yes"])
    if "cal_id" not in df.columns:
        df["cal_id"] = "pre-v19"   # logs from code before v2026.19 (incl. the Nov 5 bugs): never count for the live gate
    df["cal_id"] = df["cal_id"].fillna("none").astype(str)
    if "engine" not in df.columns:
        df["engine"] = "pre-2026.22"
    df["engine"] = df["engine"].fillna("pre-2026.22").astype(str)
    df["game"] = df["ticker"].str.split("-").str[1]
    df = df.sort_values("timestamp", kind="mergesort")
    keep = []
    for _, g in df.groupby("game", sort=False):
        # positional slicing: label slicing (g.loc[idx:]) after a sort can cut the wrong rows when index labels repeat
        # across concatenated files
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        moved = (g["clock"].astype(str) != str(g["clock"].iloc[0])).to_numpy()
        if moved.any():
            keep.append(g.iloc[int(np.flatnonzero(moved)[0]):].copy())     # from the first clock change onward
    out = pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0].copy()
    dropped = len(df) - len(out)
    if dropped:
        print(f"Dropped {dropped:,} pre-tip rows (frozen clock).")
    return out


def _fetch_one(s: requests.Session, ticker: str, errors: list) -> str:
    for host in HOSTS:
        for _ in range(3):
            try:
                r = s.get(f"{host}/markets/{ticker}", timeout=8)
            except Exception as e:
                errors.append(f"{host}: {type(e).__name__}: {e}")
                time.sleep(0.5)
                continue
            if r.status_code == 200:
                m = r.json().get("market", {})
                res = str(m.get("result", "") or "").lower()
                return res if res in ("yes", "no") else "unsettled"
            if r.status_code == 429:
                time.sleep(1.0)
                continue
            errors.append(f"{host}: HTTP {r.status_code} {r.text[:120]}")
            break
    return ""


def settlements(tickers) -> dict:
    """{ticker: 'yes' | 'no'} for every settled ticker; fetched once, then read from settlements.csv."""
    cache = {}
    if os.path.exists(CACHE):
        c = pd.read_csv(CACHE)
        cache = dict(zip(c["ticker"], c["result"]))
    need = [t for t in tickers if cache.get(t) not in ("yes", "no")]
    if need:
        print(f"Fetching {len(need)} settlement results from Kalshi...")
        s = requests.Session()
        s.headers["Accept"] = "application/json"
        errors = []
        for i, t in enumerate(need):
            cache[t] = _fetch_one(s, t, errors) or cache.get(t, "")
            if i % 100 == 99:
                print(f"  {i + 1}/{len(need)}")
            if i == 9 and not any(cache[x] in ("yes", "no", "unsettled") for x in need[:10]):
                print("ERROR: the first 10 lookups all failed. Kalshi is not answering from this machine:")
                for e in errors[:4]:
                    print("   ", e)
                break
        pd.DataFrame({"ticker": list(cache), "result": list(cache.values())}).to_csv(CACHE, index=False)
    got = {t: r for t, r in cache.items() if r in ("yes", "no")}
    print(f"Settled: {sum(t in got for t in tickers)} of {len(set(tickers))} markets.")
    return got



# ── v2026.37 shared regime helpers (fit_calibration.py, grade_predictions.py and the trader use these, never copies) ──
PHASES = ("Q1", "Q2", "HT", "Q3", "Q4")
GAP_BINS = ((0.0, 0.03, "g0-3"), (0.03, 0.06, "g3-6"), (0.06, 0.1000001, "g6-10"))   # > 10c is never an authority cell


def phase_of_period(period, per_clock=None) -> str:
    """Q1, Q2, HT (Q2 with 0:00 on the clock), Q3, Q4 (overtime counts as Q4)."""
    p = int(period or 1)
    if p <= 1:
        return "Q1"
    if p == 2:
        return "HT" if per_clock is not None and float(per_clock) <= 0.0 else "Q2"
    return "Q3" if p == 3 else "Q4"


def phase_of_rows(d: pd.DataFrame) -> np.ndarray:
    """Phase per logged prediction row, from `period` (else the clock text) and the clock string ('Q2 0:00' = HT)."""
    cl = d["clock"].astype(str) if "clock" in d.columns else pd.Series([""] * len(d), index=d.index)
    per = pd.to_numeric(d["period"], errors="coerce") if "period" in d.columns else pd.Series(np.nan, index=d.index)
    from_clock = pd.to_numeric(cl.str.extract(r"^Q(\d)", expand=False), errors="coerce").where(~cl.str.startswith("OT"), 5)
    per = per.fillna(from_clock).fillna(1).astype(int)
    ht = cl.str.match(r"^Q2 0:00")
    return np.where(per <= 1, "Q1", np.where(per == 2, np.where(ht, "HT", "Q2"), np.where(per == 3, "Q3", "Q4")))


def gap_bin_of_value(gap: float):
    """Gap bin name for an absolute model-market gap (probability units), or None above 10c (never authorized)."""
    g = abs(float(gap))
    for lo, hi, name in GAP_BINS:
        if lo <= g < hi:
            return name
    return None


def gap_bin_of_rows(p_model, p_mkt) -> np.ndarray:
    g = np.abs(np.asarray(p_model, float) - np.asarray(p_mkt, float))
    return np.select([g < 0.03, g < 0.06, g <= 0.1000001], ["g0-3", "g3-6", "g6-10"], "gt10")


def game_date_of_rows(d: pd.DataFrame) -> pd.Series:
    """GAME date (YYYY-MM-DD) from the ticker's event token (26NOV07...), so a game that runs past midnight stays on one
    side of a walk-forward split; falls back to the row timestamp."""
    if "ticker" in d.columns:
        tok = d["ticker"].astype(str).str.split("-").str[1].str[:7]
        dt = pd.to_datetime(tok, format="%y%b%d", errors="coerce")
        if dt.notna().all():
            return dt.dt.strftime("%Y-%m-%d")
    for c in ("timestamp", "time"):
        if c in d.columns:
            return d[c].astype(str).str[:10]
    return pd.Series(["0"] * len(d), index=d.index)