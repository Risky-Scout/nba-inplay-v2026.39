#!/usr/bin/env python3
"""Fit the engine's probabilities to real outcomes and write calibration.json for the trader (v2026.37).

1) Per-stat probability transform  p_cal = sigmoid(a + b * logit(p_raw)), ridge toward no change. A stat's transform is
   written only if it beats the raw engine STRICTLY WALK-FORWARD (fit on every game before date D, score the games on D;
   5+ held-out games and a game-bootstrap 97.5% lower bound of the Brier gain above 0). Leave-one-game-out is printed as
   a diagnostic only: it lets later games help judge earlier ones.

2) LOCAL MODEL AUTHORITY (v2026.37). The market-anchored pool p = sigmoid((1-w) logit(market) + w logit(model)) gets a
   weight PER CELL of the hierarchy
        *  ->  STAT  ->  *|PHASE  ->  STAT|PHASE  ->  *|PHASE|GAP  ->  STAT|PHASE|GAP
   (phases Q1, Q2, HT, Q3, Q4; gap bins g0-3, g3-6, g6-10 cents; > 10c is never a cell). Every cell is fitted against
   MARKET-ONLY (w = 0) on its own rows, shrunk toward its parent (ridge, then lambda = n_eff / (n_eff + SHRINK_K)), and
   judged on strictly chronological walk-forward evidence. A cell does NOT need the global weight to be earned: global
   w = 0 with PTS|Q2|g3-6 earning 0.11 is a valid result. Each cell gets a tier:
        PRODUCTION  normal confidence-scaled Kelly
        PILOT       one contract, maker, YES only, separate small risk budget
        RESEARCH    no directional money (logged / paper only)
   and a confidence in [0, 1] derived from games, markets, the walk-forward lower bound and fold stability.

3) Sportsbook weights (global and per stat x phase) with the deployed model weight held fixed, on the simplex, judged
   walk-forward.

The population is the trader's own entry regime, read from the newest run_config_*.json (price band, gap window, spread
cap, dead-ball rule, traded stats), so what is fitted is what is traded.

Usage: python3 fit_calibration.py            (uses every kalshi_predictions_*.csv in this folder)
"""
import os
import glob
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from kalshi_results import (load_predictions, settlements, phase_of_rows, gap_bin_of_rows, game_date_of_rows, PHASES,
                            GAP_BINS)
from nba_markov_engine import ENGINE_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_MARKETS = 100      # per stat, to consider a probability transform
MIN_CAL_GAMES = 5      # fewer games of this engine: diagnostics only, nothing written
RIDGE = 8.0            # penalty strength, in units of markets
EPS = 1e-4

# ── authority thresholds (named, printed, configurable) ──────────────────────────────────────────────────────────────
PILOT_MIN_GAMES, PILOT_MIN_MARKETS, PILOT_MIN_DATES, PILOT_MIN_EDGE_C = 5, 75, 2, 1.0
PROD_MIN_GAMES, PROD_MIN_MARKETS, PROD_MIN_DATES, PROD_MIN_EDGE_C = 10, 150, 3, 1.5
PILOT_LB_PCT, PROD_LB_PCT = 20, 5          # game-bootstrap percentiles: 80% / 90% one-sided lower bounds
SHRINK_K = 150.0                           # lambda = n_eff / (n_eff + SHRINK_K): a 150-market cell keeps half its own fit
EFF_MARKETS_PER_GAME = 15.0                # n_eff = min(markets, games x this): markets inside one game are correlated
MIN_CELL_MARKETS = 20                      # below this a cell is not fitted at all (weight = parent)
W_CAP, W_CAP_FEW_GAMES, FEW_GAMES = 0.35, 0.20, 10
CONF = dict(pilot_lo=0.10, pilot_hi=0.50, prod_lo=0.50, prod_hi=1.00)
# Multiplicity: the hierarchy tests ~50 cells on the same nights. A per-cell 90% bound alone lets several pure-noise
# cells into PRODUCTION every refit (seen on synthetic data with one real cell). PRODUCTION therefore also needs the cell
# to survive Benjamini-Hochberg false-discovery-rate control at FDR_PROD across every cell with enough evidence
# (one-sided game-bootstrap p-values). PILOT keeps the spec's per-cell 80% bar WITHOUT a multiplicity correction: it is
# the evidence tier (one contract, 1% nightly budget). MEASURED on a true null (calibrated market, uninformative model,
# 4 nights x 4 games, 6 seeds): about 20% of tested cells pass PILOT by chance; PRODUCTION at FDR 0.10 admitted a noise
# cell in 1 of 6 refits, hence 0.05.
FDR_PROD = 0.05
# v2026.38 statistical integrity
SIGNFLIP_EXACT_MAX, SIGNFLIP_B = 16, 20000   # exact game-sign-flip enumeration up to 16 games, else Monte Carlo
CONFIRM_MIN_GAMES, CONFIRM_MIN_DATES = 6, 2  # fixed confirmation sample after a candidate is frozen
CONFIRM_ALPHA = 0.10                          # one-sided sign-flip p on the confirmation sample (discovery is already FDR-controlled)
REREGISTER_AFTER_DATES = 2                    # a REJECTED cell may become a candidate again only after this many new dates
REGISTRY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "authority_registry.json")
N_BOOT = 1000


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def apply(a, b, p):
    return 1.0 / (1.0 + np.exp(-(a + b * logit(p))))


def fit(x, y, w):
    def loss(t):
        q = np.clip(apply(t[0], t[1], x), EPS, 1 - EPS)
        nll = -np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q)))
        return nll + RIDGE * (t[0] ** 2 + (t[1] - 1.0) ** 2)
    r = minimize(loss, [0.0, 1.0], method="L-BFGS-B", bounds=[(-1.5, 1.5), (0.5, 3.0)])
    return float(r.x[0]), float(r.x[1])


TRADE_STATS = {"PTS", "REB", "AST"}


def pool_p(d, w):
    return 1.0 / (1.0 + np.exp(-((1 - w) * logit(d["p_mkt_yes"].values) + w * logit(d["p_model_cal"].values))))


def fit_pool(d):
    y, wt = d["y"].values, d["w"].values
    def loss(t):
        q = np.clip(pool_p(d, t[0]), EPS, 1 - EPS)
        return -np.sum(wt * (y * np.log(q) + (1 - y) * np.log(1 - q))) + RIDGE * t[0] ** 2
    return float(minimize(loss, [0.2], method="L-BFGS-B", bounds=[(0.0, 1.0)]).x[0])


def pool3_p(d, wm, wb):
    return 1.0 / (1.0 + np.exp(-((1 - wm - wb) * logit(d["p_mkt_yes"].values) + wm * logit(d["p_model_cal"].values)
                                 + wb * logit(d["p_book"].values))))


def fit_stat_w(d, w_prior):
    """Model weight for a subset, ridge-shrunk toward w_prior (its parent; 0 = the market)."""
    y, wt = d["y"].values, d["w"].values
    def loss(t):
        q = np.clip(pool_p(d, t[0]), EPS, 1 - EPS)
        return -np.sum(wt * (y * np.log(q) + (1 - y) * np.log(1 - q))) + 2 * RIDGE * (t[0] - w_prior) ** 2
    return float(minimize(loss, [w_prior], method="L-BFGS-B", bounds=[(0.0, 1.0)]).x[0])


def fit_book_weight(d, w_model):
    """Sportsbook weight with the DEPLOYED model weight held fixed (Kalshi gets 1 - w_model - w_book >= 0)."""
    y, wt = d["y"].values, d["w"].values
    upper = max(0.0, 1.0 - w_model)
    def loss(t):
        q = np.clip(pool3_p(d, w_model, t[0]), EPS, 1 - EPS)
        return -np.sum(wt * (y * np.log(q) + (1 - y) * np.log(1 - q))) + RIDGE * t[0] ** 2
    return float(minimize(loss, [min(0.2, upper)], method="L-BFGS-B", bounds=[(0.0, min(0.6, upper))]).x[0])


def oos_ok(gains, n_boot=2000, seed=0):
    """5+ held-out games and the game-bootstrap 97.5% lower bound of the total held-out gain above 0."""
    g = np.asarray(gains, dtype=float)
    if len(g) < 5 or g.sum() <= 0:
        return False
    rng = np.random.default_rng(seed)
    boots = g[rng.integers(0, len(g), (n_boot, len(g)))].sum(axis=1)
    return bool(np.percentile(boots, 2.5) > 0)


def wbrier(p, y, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


# ── the trader's entry population ───────────────────────────────────────────────────────────────────────────────────
DEFAULT_POLICY = dict(price_lo=0.20, price_hi=0.80, gap_lo=0.0, gap_hi=0.10, max_spread_c=6, dead_ball_only=True,
                      stats=sorted(TRADE_STATS), phases=list(PHASES))


def load_policy_population(here: str = HERE):
    """The entry regime of the newest run_config_*.json (trader defaults if none). Returns (policy dict, source)."""
    cfgs = sorted(glob.glob(os.path.join(here, "run_config_*.json")))
    for path in reversed(cfgs):
        try:
            c = json.load(open(path))
        except Exception:
            continue
        if "price_min" not in c:
            continue
        pol = dict(price_lo=float(c.get("price_min", 20)) / 100.0, price_hi=float(c.get("price_max", 80)) / 100.0,
                   gap_lo=0.0, gap_hi=min(0.10, float(c.get("max_disagree", 0.10))),
                   max_spread_c=int(c.get("max_spread", 6)), dead_ball_only=not bool(c.get("allow_live_ball", False)),
                   stats=sorted({s.strip().upper() for s in str(c.get("stats", "PTS,REB,AST")).split(",") if s.strip()}),
                   phases=list(PHASES))
        return pol, os.path.basename(path)
    return dict(DEFAULT_POLICY), "trader defaults (no run_config_*.json found)"


def trade_regime(d, r=None):
    """Rows priced in the trader's entry regime: price band, model/market gap window (<= 10c), spread cap, dead ball,
    traded stats. Phases are NOT filtered: each phase earns (or fails to earn) its own authority."""
    r = r or DEFAULT_POLICY
    gap = (d["p_model_cal"] - d["p_mkt_yes"]).abs()
    m = d["p_mkt_yes"].between(r["price_lo"], r["price_hi"]) & gap.between(r["gap_lo"], r["gap_hi"] + 1e-9)
    if "stats" in r:
        m &= d["stat"].isin(r["stats"])
    if "yes_ask" in d.columns:
        spr = pd.to_numeric(d["yes_ask"], errors="coerce") - pd.to_numeric(d["yes_bid"], errors="coerce")
        m &= spr.isna() | (spr <= r["max_spread_c"])
    if r.get("dead_ball_only", True) and "dead_ball" in d.columns:
        db = d["dead_ball"].astype(str).str.lower()
        m &= ~db.isin(["false", "0"])
    return d[m]


def period_of(d):
    if "period" in d.columns and d["period"].notna().any():
        per = pd.to_numeric(d["period"], errors="coerce")
    else:
        per = pd.Series(np.nan, index=d.index)
    cl = d["clock"].astype(str)
    from_clock = pd.to_numeric(cl.str.extract(r"^Q(\d)", expand=False), errors="coerce")
    from_clock = from_clock.where(~cl.str.startswith("OT"), 5)
    return per.fillna(from_clock)


def bucket_of(d):
    return phase_of_rows(d)


def gap_bin_of(d):
    return gap_bin_of_rows(d["p_model_cal"].values, d["p_mkt_yes"].values)


def date_of(d):
    return game_date_of_rows(d)


H1_BUCKETS = ("Q1", "Q2", "HT")
BUCKETS = PHASES


# ── 1) per-stat transform, walk-forward ─────────────────────────────────────────────────────────────────────────────
def fit_stat_transforms(df, dates_upto=None, verbose=False):
    """{stat: (a, b)} fitted on `df` (rows of dates strictly before the held-out date when called inside a fold).
    A stat needs MIN_MARKETS settled markets; otherwise it stays raw."""
    out = {}
    for stat, g in df.groupby("stat"):
        if g["ticker"].nunique() < MIN_MARKETS:
            continue
        out[stat] = fit(g["p_model_raw"].values, g["y"].values, g["w"].values)
    return out


def apply_transforms(df, tr):
    p = df["p_model_raw"].values.copy()
    for stat, (a, b) in tr.items():
        m = (df["stat"] == stat).values
        p[m] = apply(a, b, p[m])
    return p


def transform_walk_forward(df):
    """Per stat: held-out Brier gain of the transform over the raw engine, one value per game, strictly chronological."""
    gains = {}
    dts = date_of(df)
    for D in sorted(dts.unique())[1:]:
        past, now = df[dts < D], df[dts == D]
        tr = fit_stat_transforms(past)
        for stat, (a, b) in tr.items():
            for gm, b_ in now[now["stat"] == stat].groupby("game"):
                y, w = b_["y"].values, b_["w"].values
                gains.setdefault(stat, []).append(
                    (wbrier(b_["p_model_raw"].values, y, w) - wbrier(apply(a, b, b_["p_model_raw"].values), y, w))
                    * b_["ticker"].nunique())
    return gains


# ── 2) local authority ───────────────────────────────────────────────────────────────────────────────────────────────
def cell_keys_for(stats):
    keys = ["*"] + list(stats) + [f"*|{p}" for p in PHASES]
    keys += [f"{s}|{p}" for s in stats for p in PHASES]
    keys += [f"*|{p}|{g}" for p in PHASES for _, _, g in GAP_BINS]
    keys += [f"{s}|{p}|{g}" for s in stats for p in PHASES for _, _, g in GAP_BINS]
    return keys


def cell_mask(d, key):
    parts = key.split("|")
    m = np.ones(len(d), dtype=bool)
    if parts[0] != "*":
        m &= (d["stat"] == parts[0]).values
    if len(parts) >= 2:
        m &= (d["_ph"] == parts[1]).values
    if len(parts) == 3:
        m &= (d["_gb"] == parts[2]).values
    return m


def parents_of(key):
    """Coarser cells a key is shrunk toward. Two parents (stat|phase: stat and *|phase) -> the SMALLER weight is used
    (conservative: if either coarser view says the model adds nothing, the child starts from nothing)."""
    parts = key.split("|")
    if key == "*":
        return []
    if len(parts) == 1:
        return ["*"]
    s, p = parts[0], parts[1]
    if len(parts) == 2:
        return ["*"] if s == "*" else [s, f"*|{p}"]
    return [f"*|{p}"] if s == "*" else [f"{s}|{p}", f"*|{p}|{parts[2]}"]


def n_eff_of(markets, games):
    return float(min(markets, games * EFF_MARKETS_PER_GAME))


def fit_authority_weights(reg, keys):
    """Top-down hierarchical weights on one population. Returns {key: dict(weight, weight_fit, parent_weight, markets,
    games, n_eff, lam)}. A cell with < MIN_CELL_MARKETS markets inherits its parent's weight."""
    out = {}
    for key in keys:                       # keys are ordered coarse -> fine
        par = [out[p]["weight"] for p in parents_of(key) if p in out]
        w_par = min(par) if par else 0.0
        sub = reg[cell_mask(reg, key)]
        mk, gm = int(sub["ticker"].nunique()), int(sub["game"].nunique())
        if mk < MIN_CELL_MARKETS or gm < 1:
            out[key] = dict(weight=w_par, weight_fit=None, parent_weight=w_par, markets=mk, games=gm, n_eff=0.0, lam=0.0)
            continue
        w_fit = fit_stat_w(sub, w_par)
        ne = n_eff_of(mk, gm)
        lam = ne / (ne + SHRINK_K)
        w = lam * w_fit + (1.0 - lam) * w_par
        w = min(w, W_CAP if gm >= FEW_GAMES else W_CAP_FEW_GAMES)
        out[key] = dict(weight=float(w), weight_fit=float(w_fit), parent_weight=float(w_par), markets=mk, games=gm,
                        n_eff=ne, lam=float(lam))
    return out


def _prep(frame, transforms, policy):
    f = frame.copy()
    f["p_model_cal"] = apply_transforms(f, transforms)
    f = trade_regime(f, policy)
    f = f.assign(_ph=bucket_of(f), _gb=gap_bin_of(f))
    return f[f["_gb"] != "gt10"]


def walk_forward_cell_evidence(df, keys, policy):
    """Strict chronological walk-forward for every cell at once. For each date D: transforms and hierarchical weights
    are fitted on games strictly before D (nothing from D or later), the regime and the gap bins are recomputed with the
    past-only transform, and each game on D is scored per cell:
        edge_c       directional edge at the market mid, sign(p_cal - p_mkt) x (y - p_mkt) x 100, market-weighted
        brier_gain   (Brier of the market - Brier of the pool at the PAST weight) x markets
    The first date is scored too (empty past: identity transform, weight 0 -> Brier gain 0, edge measured), since no
    parameter was fitted on it. Returns {key: [per-held-out-game dicts]}."""
    ev = {k: [] for k in keys}
    dts = date_of(df)
    for D in sorted(dts.unique()):
        past_raw, now_raw = df[dts < D], df[dts == D]
        tr = fit_stat_transforms(past_raw) if len(past_raw) else {}
        past = _prep(past_raw, tr, policy) if len(past_raw) else past_raw.iloc[0:0].assign(_ph=[], _gb=[])
        now = _prep(now_raw, tr, policy)
        if now.empty:
            continue
        wts = fit_authority_weights(past, keys) if len(past) else {k: dict(weight=0.0) for k in keys}
        now = now.assign(_edge=np.sign(now["p_model_cal"] - now["p_mkt_yes"]) * (now["y"] - now["p_mkt_yes"]) * 100.0)
        for key in keys:
            sub = now[cell_mask(now, key)]
            if sub.empty:
                continue
            w_past = wts.get(key, {}).get("weight", 0.0)
            for gm, b in sub.groupby("game"):
                per_mk = b.groupby("ticker")["_edge"].mean()
                y, w = b["y"].values, b["w"].values
                ev[key].append(dict(date=D, game=gm, markets=int(len(per_mk)), edge_sum=float(per_mk.sum()),
                                    brier_gain=(wbrier(b["p_mkt_yes"].values, y, w) - wbrier(pool_p(b, w_past), y, w)) * len(per_mk),
                                    w_past=float(w_past)))
    return ev


def signflip_p(edge_sums, seed=0) -> float:
    """One-sided p-value of H0: E[edge] = 0 by a GAME-CLUSTERED sign-flip randomization test. Each held-out game's edge
    contribution (sum over its markets) is a cluster; under H0 (symmetric about zero) its sign is exchangeable. The statistic
    is the total edge; the market count is fixed under flips, so this is also the test of the market-weighted mean.
    Exact enumeration of all 2^n sign vectors up to SIGNFLIP_EXACT_MAX games (the identity is included, so p >= 2^-n)."""
    e = np.asarray(edge_sums, dtype=float)
    n = len(e)
    if n == 0:
        return 1.0
    t = e.sum()
    if n <= SIGNFLIP_EXACT_MAX:
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
        sums = (1.0 - 2.0 * bits) @ e
        return float(np.mean(sums >= t - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(SIGNFLIP_B, n))
    return float((np.sum(signs @ e >= t - 1e-12) + 1.0) / (SIGNFLIP_B + 1.0))


def summarize_evidence(rows, seed=0):
    if not rows:
        return dict(games=0, markets=0, dates=0, edge_c=None, lb80_c=None, lb90_c=None, hi90_c=None, brier_gain=0.0,
                    brier_hi95=None, stability=None, edge_sd_c=None, p_value=1.0, p_boot=1.0)
    r = pd.DataFrame(rows)
    games, markets, dates = int(len(r)), int(r["markets"].sum()), int(r["date"].nunique())
    mean = float(r["edge_sum"].sum() / max(1, markets))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), (N_BOOT, len(r)))
    es, ms, bg = r["edge_sum"].values, r["markets"].values, r["brier_gain"].values
    boots = es[idx].sum(axis=1) / np.maximum(1, ms[idx].sum(axis=1))
    bgb = bg[idx].sum(axis=1)
    by_date = r.groupby("date").apply(lambda x: x["edge_sum"].sum() / max(1, x["markets"].sum()))
    # v2026.38: the FDR p-value is a proper null test (game-clustered sign flip), not the share of bootstrap draws <= 0;
    # the bootstrap is kept for the reported intervals only
    p_value = signflip_p(es, seed=seed)
    p_boot = float((np.sum(boots <= 0.0) + 1.0) / (N_BOOT + 1.0))
    return dict(games=games, markets=markets, dates=dates, edge_c=round(mean, 3),
                lb80_c=round(float(np.percentile(boots, PILOT_LB_PCT)), 3), lb90_c=round(float(np.percentile(boots, PROD_LB_PCT)), 3),
                hi90_c=round(float(np.percentile(boots, 95)), 3), brier_gain=round(float(bg.sum()), 5),
                brier_hi95=round(float(np.percentile(bgb, 95)), 5), stability=round(float((by_date > 0).mean()), 3),
                edge_sd_c=round(float(np.std(boots, ddof=1)) if len(r) > 1 else 0.0, 3), p_value=round(p_value, 5),
                p_boot=round(p_boot, 5))


def bh_survivors(pvals: dict, q: float) -> set:
    """Benjamini-Hochberg: keys whose p-value survives FDR q among all keys given."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, keep = len(items), 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            keep = i
    return {k for k, _ in items[:keep]}


def tier_of(s, weight, key, fdr_ok=True):
    """(tier, confidence, research_reason). research_reason 'insufficient_evidence' lets the trader fall through to a
    coarser cell; 'failed_evidence' blocks (a cell with enough data that failed is evidence AGAINST trading there)."""
    insufficient = s["games"] < PILOT_MIN_GAMES or s["markets"] < PILOT_MIN_MARKETS or s["dates"] < PILOT_MIN_DATES
    if insufficient or s["edge_c"] is None:
        return "RESEARCH", 0.0, "insufficient_evidence"
    clearly_worse = s["brier_hi95"] is not None and s["brier_hi95"] < 0
    if weight <= 0 or s["edge_c"] <= 0 or clearly_worse:
        return "RESEARCH", 0.0, "failed_evidence"
    z = s["edge_c"] / max(1e-6, s["edge_sd_c"] or 1e-6)
    stab = s["stability"] if s["stability"] is not None else 0.0
    prod = (s["games"] >= PROD_MIN_GAMES and s["markets"] >= PROD_MIN_MARKETS and s["dates"] >= PROD_MIN_DATES
            and s["edge_c"] >= PROD_MIN_EDGE_C and s["lb90_c"] > 0 and s["brier_gain"] > 0 and fdr_ok)
    if prod:
        strength = min(1.0, s["games"] / (2.0 * PROD_MIN_GAMES)) * min(1.0, z / 3.0) * stab
        return "PRODUCTION", round(float(np.clip(CONF["prod_lo"] + (CONF["prod_hi"] - CONF["prod_lo"]) * strength,
                                                 CONF["prod_lo"], CONF["prod_hi"])), 3), ""
    if s["edge_c"] >= PILOT_MIN_EDGE_C and s["lb80_c"] >= 0:
        strength = min(1.0, s["games"] / PROD_MIN_GAMES, s["markets"] / PROD_MIN_MARKETS) * min(1.0, z / 1.645) * stab
        return "PILOT", round(float(np.clip(CONF["pilot_lo"] + (CONF["pilot_hi"] - CONF["pilot_lo"]) * strength,
                                            CONF["pilot_lo"], CONF["pilot_hi"])), 3), ""
    return "RESEARCH", 0.0, "failed_evidence"


def cell_weight_sd(reg, key, w_par, n_boot=30, seed=0):
    """Game-bootstrap s.d. of a cell's fitted weight (the trader charges it as calibration uncertainty)."""
    sub = reg[cell_mask(reg, key)]
    games = sub["game"].unique()
    if len(games) < 3:
        return None
    rng, ws = np.random.default_rng(seed), []
    by = {g: sub[sub["game"] == g] for g in games}
    for _ in range(n_boot):
        ws.append(fit_stat_w(pd.concat([by[g] for g in rng.choice(games, len(games), replace=True)], ignore_index=True), w_par))
    return float(np.std(ws, ddof=1))


def load_registry(path=None) -> dict:
    """Frozen PRODUCTION candidates for the current engine (another engine's registry is ignored: evidence resets)."""
    try:
        with open(path or REGISTRY_JSON) as f:
            r = json.load(f)
        if r.get("engine") == ENGINE_VERSION and isinstance(r.get("cells"), dict):
            return r
    except Exception:
        pass
    return {"engine": ENGINE_VERSION, "cells": {}}


def save_registry(reg, path=None):
    reg["updated"] = datetime.now().isoformat(timespec="seconds")
    with open(path or REGISTRY_JSON, "w") as f:
        json.dump(reg, f, indent=2, default=str)


def confirmation_sample(rows, frozen_through):
    """The FIXED confirmation sample: whole dates strictly after the freeze, in order, until it holds at least
    CONFIRM_MIN_GAMES games over CONFIRM_MIN_DATES dates. None while the sample is incomplete (decision pending)."""
    later = sorted([r for r in rows if str(r["date"]) > str(frozen_through)], key=lambda r: str(r["date"]))
    dates = sorted({str(r["date"]) for r in later})
    take = []
    for i, d in enumerate(dates, start=1):
        take += [r for r in later if str(r["date"]) == d]
        if i >= CONFIRM_MIN_DATES and len(take) >= CONFIRM_MIN_GAMES:
            return take
    return None


def update_registry(reg, key, discovered, rows, last_date, all_dates):
    """Discovery -> freeze -> confirmation (v2026.38). A cell that passes every PRODUCTION rule on the data available today
    (including FDR across all tested cells) is FROZEN as a CANDIDATE through today's last date. It is CONFIRMED or REJECTED
    exactly once, on the first CONFIRM_MIN_GAMES games / CONFIRM_MIN_DATES dates AFTER the freeze, which the discovery
    search never saw. Only CONFIRMED cells may be PRODUCTION. Returns the cell's registry entry (or None)."""
    cells = reg.setdefault("cells", {})
    e = cells.get(key)
    if e and e.get("status") == "CANDIDATE":
        samp = confirmation_sample(rows, e["frozen_through"])
        if samp is not None:
            es = [r["edge_sum"] for r in samp]
            mk = sum(r["markets"] for r in samp)
            edge = float(sum(es) / max(1, mk))
            p = signflip_p(es)
            ok = edge > 0 and p <= CONFIRM_ALPHA
            e.update(status="CONFIRMED" if ok else "REJECTED", confirm_games=len(samp), confirm_markets=int(mk),
                     confirm_dates=len({str(r["date"]) for r in samp}), confirm_edge_c=round(edge, 3), confirm_p=round(p, 5),
                     decided_through=max(str(r["date"]) for r in samp))
        return e
    if e and e.get("status") == "REJECTED":
        newer = [d for d in all_dates if str(d) > str(e.get("decided_through", ""))]
        if len(newer) < REREGISTER_AFTER_DATES:
            return e
    if e and e.get("status") == "CONFIRMED":
        return e
    if discovered:
        e = dict(status="CANDIDATE", frozen_through=str(last_date), registered=datetime.now().isoformat(timespec="seconds"),
                 previous=None if not e else {k: e.get(k) for k in ("status", "confirm_edge_c", "confirm_p", "decided_through")})
        cells[key] = e
    return e


def build_authority(df, transforms, policy, verbose=True, registry=None):
    """Deployment weights (all data) + walk-forward evidence -> {key: authority object}."""
    stats = sorted(set(policy.get("stats", TRADE_STATS)) & set(df["stat"].unique()))
    keys = cell_keys_for(stats)
    full = _prep(df, transforms, policy)
    wts = fit_authority_weights(full, keys)
    ev = walk_forward_cell_evidence(df, keys, policy)
    auth = {}
    summ = {key: summarize_evidence(ev[key]) for key in keys}
    tested = {k: v["p_value"] for k, v in summ.items()
              if v["games"] >= PILOT_MIN_GAMES and v["markets"] >= PILOT_MIN_MARKETS and v["dates"] >= PILOT_MIN_DATES}
    fdr_set = bh_survivors(tested, FDR_PROD) if tested else set()
    reg = registry if registry is not None else load_registry()
    all_dates = sorted({str(r["date"]) for k_ in keys for r in ev[k_]})
    last_date = all_dates[-1] if all_dates else ""
    for key in keys:
        s = summ[key]
        if s["markets"] == 0 and wts[key]["markets"] == 0:
            continue
        w = wts[key]
        tier, conf, why = tier_of(s, w["weight"], key, fdr_ok=key in fdr_set)
        entry = update_registry(reg, key, tier == "PRODUCTION", ev[key], last_date, all_dates)
        if tier == "PRODUCTION" and not (entry and entry.get("status") == "CONFIRMED"):
            # discovered (or still awaiting / failed confirmation): not PRODUCTION; it can still be PILOT on its own evidence
            tier, conf, why = tier_of(s, w["weight"], key, fdr_ok=False)
        a = dict(weight=round(w["weight"], 4), weight_fit=None if w["weight_fit"] is None else round(w["weight_fit"], 4),
                 parent_weight=round(w["parent_weight"], 4), lam=round(w["lam"], 3), n_eff=round(w["n_eff"], 1),
                 games=s["games"], markets=s["markets"], dates=s["dates"], wf_edge_c=s["edge_c"], wf_ci80_lo_c=s["lb80_c"],
                 wf_ci90_lo_c=s["lb90_c"], wf_ci90_hi_c=s["hi90_c"], wf_brier_gain=s["brier_gain"], stability=s["stability"],
                 tier=tier, confidence=conf, research_reason=why, fit_markets=w["markets"], fit_games=w["games"],
                 wf_p_value=s["p_value"], wf_p_boot=s["p_boot"], fdr_survivor=key in fdr_set, cells_tested=len(tested),
                 confirmation=None if not entry else dict(entry))
        if tier != "RESEARCH":
            sd = cell_weight_sd(full, key, w["parent_weight"])
            if sd is not None:
                a["w_sd"] = round(sd, 4)
        auth[key] = a
    return auth, full


def print_thresholds():
    print(f"\nAuthority thresholds: PILOT >= {PILOT_MIN_GAMES} games, {PILOT_MIN_MARKETS} markets, {PILOT_MIN_DATES} dates, "
          f"walk-forward edge >= {PILOT_MIN_EDGE_C}c, 80% lower bound >= 0 | PRODUCTION >= {PROD_MIN_GAMES} games, "
          f"{PROD_MIN_MARKETS} markets, {PROD_MIN_DATES} dates, edge >= {PROD_MIN_EDGE_C}c, 90% lower bound > 0, Brier gain > 0 | "
          f"shrinkage lambda = n_eff/(n_eff+{SHRINK_K:g}), n_eff = min(markets, {EFF_MARKETS_PER_GAME:g} x games) | "
          f"PRODUCTION also needs Benjamini-Hochberg FDR <= {FDR_PROD} across all cells tested (game-clustered sign-flip "
          f"p-values) AND confirmation on {CONFIRM_MIN_GAMES}+ later games / {CONFIRM_MIN_DATES}+ later dates after the cell is frozen")


def walk_forward_gains(tr, cell, w_fallback_fn, w_cell_fn, p_fn=None):
    """Chronological, game-blocked: fit on every game BEFORE date d, score each game ON d. Returns one gain per game."""
    p_fn = p_fn or pool_p
    dt_all, dt_c = date_of(tr), date_of(cell)
    gains = []
    for d in sorted(dt_c.unique())[1:]:
        past_all, past_c, now_c = tr[dt_all < d], cell[dt_c < d], cell[dt_c == d]
        if past_c["ticker"].nunique() < 20:
            continue
        w_fb, w_c = w_fallback_fn(past_all), w_cell_fn(past_c, past_all)
        for g, b in now_c.groupby("game"):
            yb, wb = b["y"].values, b["w"].values
            gains.append((wbrier(p_fn(b, w_fb), yb, wb) - wbrier(p_fn(b, w_c), yb, wb)) * b["ticker"].nunique())
    return gains


def book_weights(bk, auth):
    """Global and per (stat, phase) sportsbook weights with the model weight held at what the trader deploys there,
    judged walk-forward against market+model."""
    out = {}
    if bk["ticker"].nunique() < MIN_MARKETS or bk["game"].nunique() < 2:
        return None, out
    wm_g = auth.get("*", {}).get("weight", 0.0) if auth.get("*", {}).get("tier") != "RESEARCH" else 0.0
    gain = walk_forward_gains(bk, bk, lambda past: 0.0, lambda pc, pa: fit_book_weight(pc, wm_g),
                              p_fn=lambda d, wb: pool3_p(d, wm_g, wb))
    wb_global = round(fit_book_weight(bk, wm_g), 4) if oos_ok(gain) else None
    print(f"Sportsbook weight (global, model {wm_g:.2f} fixed): walk-forward {'gain -> ' + str(wb_global) if wb_global is not None else 'NO gain'} ({len(gain)} games)")
    bk = bk.assign(_b=bucket_of(bk))
    for (s_, b_), gc in bk.groupby(["stat", "_b"]):
        if gc["ticker"].nunique() < MIN_MARKETS or gc["game"].nunique() < 2:
            continue
        a = auth.get(f"{s_}|{b_}") or auth.get(f"*|{b_}") or {}
        wm = a.get("weight", 0.0) if a.get("tier", "RESEARCH") != "RESEARCH" else 0.0
        g2 = walk_forward_gains(bk, gc, lambda past: 0.0, lambda pc, pa, wm=wm: fit_book_weight(pc, wm),
                                p_fn=lambda d, wb, wm=wm: pool3_p(d, wm, wb))
        ok = oos_ok(g2)
        print(f"   book weight {s_}|{b_}: walk-forward {'gain' if ok else 'NO gain'} ({len(g2)} later games)")
        if ok:
            out[f"{s_}|{b_}"] = round(fit_book_weight(gc, wm), 4)
    return wb_global, out


def main():
    df = load_predictions()
    n_all = len(df)
    df = df[df["engine"] == ENGINE_VERSION]   # a calibration is only valid for the engine that produced the predictions
    print(f"Using {len(df):,} of {n_all:,} predictions made by engine {ENGINE_VERSION} (older engines are excluded).")
    if df.empty:
        raise SystemExit(f"No predictions from engine {ENGINE_VERSION} yet: paper/live trade a night with it, then refit.")
    res = settlements(df["ticker"].unique())
    df["y"] = df["ticker"].map(res).map({"yes": 1.0, "no": 0.0})
    df = df.dropna(subset=["y"])
    if df.empty:
        raise SystemExit("No settled predictions: nothing to fit (see the Kalshi error above, if any).")
    df["w"] = 1.0 / df.groupby("ticker")["ticker"].transform("size")
    n_games, n_dates = df["game"].nunique(), date_of(df).nunique()
    print(f"{df['ticker'].nunique()} settled markets, {n_games} games, {n_dates} game dates.")
    policy, pol_src = load_policy_population()
    print(f"Entry population (what the trader trades): {policy}  [from {pol_src}]")

    # 1) per-stat transforms: walk-forward authorization, LOGO diagnostic
    out = {}
    wf = transform_walk_forward(df)
    print(f"\n{'stat':<5}{'markets':>8}{'a':>8}{'b':>7}   Brier raw / market   walk-forward games   decision")
    full_tr = fit_stat_transforms(df)
    for stat, g in df.groupby("stat"):
        n_mk = g["ticker"].nunique()
        if stat not in full_tr:
            print(f"{stat:<5}{n_mk:>8}   skipped: fewer than {MIN_MARKETS} markets")
            continue
        a, b = full_tr[stat]
        y, w = g["y"].values, g["w"].values
        br, bm = wbrier(g["p_model_raw"].values, y, w), wbrier(g["p_mkt_yes"].values, y, w)
        keep = oos_ok(wf.get(stat, []))
        # diagnostic only (leave one game out lets later games judge earlier ones)
        logo = []
        for gg in g["game"].unique():
            trm, tem = g["game"].values != gg, g["game"].values == gg
            a_g, b_g = fit(g["p_model_raw"].values[trm], y[trm], w[trm])
            logo.append(wbrier(g["p_model_raw"].values[tem], y[tem], w[tem]) - wbrier(apply(a_g, b_g, g["p_model_raw"].values[tem]), y[tem], w[tem]))
        if keep:
            out[stat] = {"a": round(a, 4), "b": round(b, 4), "markets": int(n_mk), "brier_raw": round(br, 5),
                         "brier_market": round(bm, 5), "wf_games": len(wf.get(stat, []))}
        print(f"{stat:<5}{n_mk:>8}{a:>8.3f}{b:>7.3f}   {br:.4f} / {bm:.4f}   {len(wf.get(stat, [])):>4}   "
              f"{'WRITTEN' if keep else 'not written (no walk-forward gain)'}   [LOGO diagnostic mean gain {np.mean(logo):+.5f}]")
    transforms = {s: (v["a"], v["b"]) for s, v in out.items()}

    # 2) local authority
    print_thresholds()
    registry = load_registry()
    auth, full = build_authority(df, transforms, policy, registry=registry)
    df["p_model_cal"] = apply_transforms(df, transforms)
    tiers = {t: [k for k, v in auth.items() if v["tier"] == t] for t in ("PRODUCTION", "PILOT", "RESEARCH")}
    print(f"\n{'CELL':<16}{'TIER':<12}{'CONF':>6}{'WEIGHT':>8}{'FIT':>7}{'PARENT':>8}{'EDGE':>7}{'LB80':>7}{'LB90':>7}"
          f"{'GAMES':>7}{'MKTS':>6}{'DATES':>6}  reason")
    for k, v in auth.items():
        if v["markets"] == 0:
            continue
        print(f"{k:<16}{v['tier']:<12}{v['confidence']:>6.2f}{v['weight']:>8.3f}{(v['weight_fit'] if v['weight_fit'] is not None else float('nan')):>7.3f}"
              f"{v['parent_weight']:>8.3f}{(v['wf_edge_c'] if v['wf_edge_c'] is not None else float('nan')):>+7.2f}"
              f"{(v['wf_ci80_lo_c'] if v['wf_ci80_lo_c'] is not None else float('nan')):>+7.2f}"
              f"{(v['wf_ci90_lo_c'] if v['wf_ci90_lo_c'] is not None else float('nan')):>+7.2f}{v['games']:>7}{v['markets']:>6}{v['dates']:>6}  {v['research_reason']}")
    g_ = auth.get("*", {})
    w_global = g_.get("weight", 0.0) if g_.get("tier") in ("PRODUCTION", "PILOT") else 0.0
    pool = {"w": round(w_global, 4), "w_fitted": g_.get("weight_fit"), "earned": g_.get("tier") == "PRODUCTION",
            "markets": int(full["ticker"].nunique()), "games": int(full["game"].nunique()), "population": "trade regime",
            "regime": policy, "regime_source": pol_src, "authority": auth,
            # compatibility for older readers: only authorized cells
            "stat_w": {k: v["weight"] for k, v in auth.items() if "|" not in k and k != "*" and v["tier"] != "RESEARCH"},
            "cell_w": {k: v["weight"] for k, v in auth.items() if "|" in k and v["tier"] != "RESEARCH"},
            "thresholds": dict(PILOT_MIN_GAMES=PILOT_MIN_GAMES, PILOT_MIN_MARKETS=PILOT_MIN_MARKETS, PILOT_MIN_DATES=PILOT_MIN_DATES,
                               PILOT_MIN_EDGE_C=PILOT_MIN_EDGE_C, PROD_MIN_GAMES=PROD_MIN_GAMES, PROD_MIN_MARKETS=PROD_MIN_MARKETS,
                               PROD_MIN_DATES=PROD_MIN_DATES, PROD_MIN_EDGE_C=PROD_MIN_EDGE_C, SHRINK_K=SHRINK_K)}
    if g_.get("w_sd") is not None:
        pool["w_sd"] = g_["w_sd"]

    # 3) sportsbook weights
    bk = full[full["p_book"].notna()] if "p_book" in full.columns else full.iloc[0:0]
    wb, bcw = book_weights(bk, auth)
    if wb is not None:
        pool["w_book"] = wb
    if bcw:
        pool["book_cell_w"] = bcw
    if bk["ticker"].nunique() < MIN_MARKETS:
        print(f"Sportsbook weight: not fitted yet ({bk['ticker'].nunique()} markets with book prices; need {MIN_MARKETS} over 2+ games)")

    print(f"\nGLOBAL WEIGHT: {w_global:.2f}  ({g_.get('tier', 'RESEARCH')})")
    print("LOCAL AUTHORITY:")
    for t in ("PRODUCTION", "PILOT", "RESEARCH"):
        print(f"    {t}: {len(tiers[t])} cells" + (f"  {tiers[t]}" if t != "RESEARCH" and tiers[t] else ""))
    if n_games < MIN_CAL_GAMES:
        print(f"\nOnly {n_games} game(s) from engine {ENGINE_VERSION}: diagnostics above, but NOTHING written "
              f"(needs {MIN_CAL_GAMES}+ games so one night cannot steer the engine).")
        return
    out["_pool"] = pool
    blob = json.dumps(out, sort_keys=True, default=str)
    out["_meta"] = {"id": hashlib.sha1(blob.encode()).hexdigest()[:10], "engine": ENGINE_VERSION, "fitted": datetime.now().isoformat(timespec="seconds"),
                    "files": sorted(df["source_file"].unique().tolist()), "form": "p = sigmoid(a + b*logit(p_raw))",
                    "schema": "authority-v1"}
    path = os.path.join(HERE, "calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    save_registry(registry, os.path.join(HERE, "authority_registry.json"))
    _rc = registry.get("cells", {})
    print(f"authority_registry.json: {sum(1 for v in _rc.values() if v.get('status') == 'CANDIDATE')} frozen candidate(s) awaiting "
          f"confirmation, {sum(1 for v in _rc.values() if v.get('status') == 'CONFIRMED')} confirmed, "
          f"{sum(1 for v in _rc.values() if v.get('status') == 'REJECTED')} rejected "
          f"(confirmation: first {CONFIRM_MIN_GAMES}+ games over {CONFIRM_MIN_DATES}+ dates after the freeze, sign-flip p <= {CONFIRM_ALPHA})")
    print(f"\nWrote calibration.json (id {out['_meta']['id']}). The trader loads it at startup and uses _pool.authority.")


if __name__ == "__main__":
    main()