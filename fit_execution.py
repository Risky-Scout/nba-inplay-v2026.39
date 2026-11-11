"""Fit the execution model (v2026.37) from the trader's own LIVE quote logs and decide champion / challenger.

Labels (censoring, not guessing):
  FILLED              the order has a fill event (any quantity)                                   -> 1
  TERMINAL_UNFILLED   a terminal event CANCELLED_CONFIRMED / EXPIRED_CONFIRMED with nothing filled -> 0
  CENSORED            anything else (still resting, cancel requested but never proven terminal)    -> excluded
Models (one feature definition, kalshi_exec_model.py; continuous features standardized with TRAINING-ONLY scalers):
  fill      P(fill before the quote's own TTL | quote, state), logistic, inverse-propensity weighted; TTL and the stoppage
            type are features, so a 5 s free-throw quote and a 45 s period-break quote are not the same object
  drift     E[side mid move over 30 s | quote, state] on TERMINAL-UNFILLED quotes: the counterfactual market move for a
            comparable signal without our fill (it may load on the model-market gap: that is convergence, not execution)
  mark_all  E[M30 - drift_hat | fill] over ALL fills (fills just before an NBA play are the pick-offs that cost money):
            PRODUCTION quote economics
  mark_ef   the same on event-free fills only: microstructure diagnostic
  Both markout models are residualized against drift and may NOT load on the gap (that would count directional alpha
  twice: QuoteEV adds it separately).
Validation: rolling expanding windows (dates 1..k -> k+1, every later date). Fill: log loss, Brier, calibration intercept
and slope, reliability bins. Markout: MSE, MAE, bias. POLICY: doubly-robust value of the model's quote choice vs the
bid+1 and join-the-bid baselines on held-out dates. Promotion needs the POLICY to beat both baselines, not just a lower
MSE. The newest fit is always written as challenger_exec_model.json; it replaces champion_exec_model.json only if it is
validated and beats the champion's policy value on dates the champion never trained on. The trader trades the champion
and scores the challenger in shadow.

  python3 fit_execution.py
"""
import glob
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import kalshi_exec_model as XM
from kalshi_exec_model import EXEC_FEATURES, exec_features
from execution_hazard import action_key

HERE = os.path.dirname(os.path.abspath(__file__))
QUOTE_POLICY = "qp8"
MIN_QUOTES, MIN_FILLS, MIN_MARKS, MIN_VAL_FILLS, MIN_DATES = 300, 40, 30, 10, 2
L2 = 1.0
N_BOOT = 30
CHAMPION = os.path.join(HERE, "champion_exec_model.json")
CHALLENGER = os.path.join(HERE, "challenger_exec_model.json")
TERMINAL_UNFILLED = ("CANCELLED_CONFIRMED", "EXPIRED_CONFIRMED")
GAP_COL = XM.GAP_COL


def _read(pattern, here=HERE):
    fr = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(here, pattern))) if os.path.getsize(f) > 0]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def label_quotes(q: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """qp7 labels.

    Any observed fill is immediately a positive for the any-fill model.
    Terminal-unfilled is a negative.  Terminal fill fraction/filled quantity
    are trained only after terminal proof; unresolved partial fills remain
    censored for the fraction target.
    """
    q = q.copy()
    if "requested_qty" not in q.columns:
        q["requested_qty"] = 1
        print("WARNING: old execution rows missing requested_qty; imputing 1 for training only")
    q["requested_qty"] = pd.to_numeric(q["requested_qty"], errors="coerce").fillna(1).clip(lower=1)
    oid = q["order_id"].astype(str)
    if ev.empty:
        q["label"] = np.nan
        q["status"] = "CENSORED"
        q["filled_qty_terminal"] = np.nan
        q["fill_fraction"] = np.nan
        q["any_fill"] = np.nan
        q["fully_filled"] = np.nan
        return q
    e = ev.copy()
    e["order_id"] = e["order_id"].astype(str)
    fills = e[e["event"].astype(str) == "fill"].copy()
    fill_ids = set(fills["order_id"].astype(str))
    if len(fills):
        if "contracts" in fills.columns:
            fqty = pd.to_numeric(fills["contracts"], errors="coerce").fillna(0)
        elif "filled" in fills.columns:
            fqty = pd.to_numeric(fills["filled"], errors="coerce").fillna(0)
        else:
            fqty = pd.Series(1.0, index=fills.index)
        fill_sum = fills.assign(_fq=fqty).groupby("order_id")["_fq"].sum()
    else:
        fill_sum = pd.Series(dtype=float)
    term = e[e["event"].astype(str) == "terminal"].copy()
    term_ids = set(term["order_id"].astype(str))
    fq = oid.map(fill_sum).fillna(0.0)
    terminal = oid.isin(term_ids)
    observed_fill = oid.isin(fill_ids)
    q["filled_qty_terminal"] = np.where(terminal, fq, np.nan)
    q["fill_fraction"] = np.where(terminal, np.minimum(1.0, fq / q["requested_qty"]), np.nan)
    q["any_fill"] = np.where(observed_fill, 1.0, np.where(terminal, 0.0, np.nan))
    q["fully_filled"] = np.where(terminal, (fq >= q["requested_qty"]).astype(float), np.nan)
    q["label"] = q["any_fill"]
    q["status"] = np.where(observed_fill, "FILLED", np.where(terminal, "TERMINAL_UNFILLED", "CENSORED"))
    return q
def load(here=HERE):
    q, ev, mk, qm = (_read("kalshi_quotes_*.csv", here), _read("kalshi_quote_events_*.csv", here),
                     _read("kalshi_markouts_*.csv", here), _read("kalshi_quote_marks_*.csv", here))
    if q.empty:
        return q
    q = q[~q["order_id"].astype(str).str.startswith("paper")]
    if "quote_policy" in q.columns:
        q = q[q["quote_policy"].astype(str) == QUOTE_POLICY]
    if q.empty:
        return q
    q = label_quotes(q, ev)
    if not mk.empty and "order_id" in mk.columns:
        mk = mk.assign(order_id=mk["order_id"].astype(str), m30=pd.to_numeric(mk["m30"], errors="coerce"))
        q["m30_all"] = q["order_id"].astype(str).map(mk.groupby("order_id")["m30"].mean())
        ef = mk[mk.get("m30_event", pd.Series(dtype=str)).astype(str).str.lower() == "false"]
        q["m30_ef"] = q["order_id"].astype(str).map(ef.groupby("order_id")["m30"].mean())
    else:
        q["m30_all"], q["m30_ef"] = np.nan, np.nan
    if not qm.empty and "order_id" in qm.columns:
        q["q_m30"] = q["order_id"].astype(str).map(qm.assign(order_id=qm["order_id"].astype(str)).groupby("order_id")["q_m30"].mean())
    else:
        q["q_m30"] = np.nan
    q["date"] = q["run_id"].astype(str).str[:8]
    q["game"] = q["ticker"].astype(str).str.split("-").str[1]
    return q.reset_index(drop=True)


def _num(r, k, d=0.0):
    v = getattr(r, k, d)
    try:
        v = float(v)
        return d if not np.isfinite(v) else v
    except (TypeError, ValueError):
        return d


def side_book_of(r):
    yb, ya = int(_num(r, "yes_bid")), int(_num(r, "yes_ask"))
    return (yb, ya) if r.side == "yes" else (100 - ya, 100 - yb)


def features_at(r, price):
    """Raw features of quote row r as if it had been placed at `price` (same book, same state)."""
    bid, ask = side_book_of(r)
    qb, qa = _num(r, "qty_bid"), _num(r, "qty_ask")
    qsb, qsa = (qb, qa) if r.side == "yes" else (qa, qb)
    gap = 100.0 * (_num(r, "p_used") - _num(r, "p_mkt"))
    micro = {k: _num(r, k) for k in XM.MICRO_KEYS}
    micro["has_ws"] = str(getattr(r, "has_ws", "")).lower() in ("true", "1")
    return exec_features(r.side, str(r.stat), int(_num(r, "period")), bid, ask, int(price), gap, ttl_s=_num(r, "ttl", 8.0),
                         stop=str(getattr(r, "stop", "other") or "other"), phase=str(getattr(r, "phase", "") or "") or None,
                         qty_bid=qsb, qty_ask=qsa, queue_ahead=qsb if int(price) == bid else 0.0,
                         book_age_s=_num(r, "book_age_ms") / 1000.0, t_since_action_s=_num(r, "t_since_action_s"), micro=micro,
                         order_qty=_num(r, "requested_qty", 1.0), best_depth=qsb)


def design(q):
    return np.array([features_at(r, int(r.price)) for r in q.itertuples()]) if len(q) else np.zeros((0, len(EXEC_FEATURES)))


def fit_logit(X, y, w):
    def nll(b):
        z = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        return -np.sum(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) + L2 * np.sum(b[1:] ** 2)
    def grad(b):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        g = -(X.T @ (w * (y - p)))
        g[1:] += 2 * L2 * b[1:]
        return g
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def fit_ridge(X, y, w, exclude_gap=True):
    X = X.copy()
    R = L2 * np.eye(X.shape[1]); R[0, 0] = 0.0
    if exclude_gap:
        X[:, GAP_COL] = 0.0
        R[GAP_COL, GAP_COL] = 1e6
    W = np.diag(w)
    return np.linalg.solve(X.T @ W @ X + R, X.T @ W @ y)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_slope(p, y):
    """Logistic recalibration y ~ a + b logit(p): (intercept, slope); slope 1 / intercept 0 = calibrated."""
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.c_[np.ones(len(lp)), lp]
    b = minimize(lambda t: -np.sum(y * (X @ t) - np.log1p(np.exp(X @ t))), [0.0, 1.0], method="BFGS").x
    return float(b[0]), float(b[1])


def fit_all(qtr):
    """Scaler + drift + fill + markout models on training rows only."""
    X = design(qtr)
    sc = XM.fit_scaler(X)
    Xs = XM.standardize(sc, X)
    lab = qtr["label"].values
    if "fill_fraction" in qtr.columns:
        frac = pd.to_numeric(qtr["fill_fraction"], errors="coerce").to_numpy()
    else:
        # Legacy qp6/one-contract research rows: binary fill is the exact fraction.
        frac = pd.to_numeric(qtr["label"], errors="coerce").to_numpy()
    known = ~np.isnan(lab)
    w = 1.0 / np.clip(pd.to_numeric(qtr.get("propensity", 1.0), errors="coerce").fillna(1.0).values, 0.05, 1.0)
    b_fill = fit_logit(Xs[known], lab[known], w[known]) if known.sum() >= 2 else np.zeros(Xs.shape[1])
    kf = np.isfinite(frac)
    b_frac = fit_logit(Xs[kf], np.clip(frac[kf], 1e-5, 1 - 1e-5), w[kf]) if kf.sum() >= 2 else b_fill.copy()
    if "filled_qty_terminal" in qtr.columns:
        yqty = pd.to_numeric(qtr["filled_qty_terminal"], errors="coerce").to_numpy()
    else:
        req = pd.to_numeric(qtr["requested_qty"] if "requested_qty" in qtr else 1.0, errors="coerce")
        if np.isscalar(req):
            req = np.full(len(qtr), float(req))
        else:
            req = np.asarray(req, float)
        yqty = req * frac
    kq = np.isfinite(yqty)
    b_qty = fit_ridge(Xs[kq], yqty[kq], w[kq], exclude_gap=False) if kq.sum() >= 10 else np.zeros(Xs.shape[1])
    unf = (lab == 0) & qtr["q_m30"].notna().values
    b_drift = fit_ridge(Xs[unf], qtr["q_m30"].values[unf], w[unf], exclude_gap=False) if unf.sum() >= 10 else np.zeros(Xs.shape[1])
    drift_hat = Xs @ b_drift
    out = dict(scaler=sc, fill=b_fill, fill_any=b_fill, fill_fraction=b_frac, filled_qty=b_qty, drift=b_drift)
    for name, col in (("mark_all", "m30_all"), ("mark_ef", "m30_ef")):
        has = (lab == 1) & qtr[col].notna().values
        y = qtr[col].values - drift_hat
        out[name] = fit_ridge(Xs[has], y[has], w[has]) if has.sum() >= 10 else np.zeros(Xs.shape[1])
        out[f"n_{name}"] = int(has.sum())
    return out


def _model_dict(m):
    return dict(features=EXEC_FEATURES, scaler=m["scaler"], fill=list(map(float, m["fill"])),
                fill_any=list(map(float, m["fill_any"])), fill_fraction=list(map(float, m["fill_fraction"])),
                filled_qty=list(map(float, m["filled_qty"])),
                mark_all=list(map(float, m["mark_all"])), mark_ef=list(map(float, m["mark_ef"])), drift=list(map(float, m["drift"])))


def ladder_of(r):
    bid, ask = side_book_of(r)
    top = min(ask - 1, int(np.floor((bid + ask) / 2.0)))
    return [bid] if ask - bid < 2 else list(range(bid, max(bid, top) + 1))


MIN_SUPPORT_ACTIONS = 20  # logged actions at a quote offset before PRODUCTION may use it
SUPPORT_EPS = 0.05          # pi_b(target | X) below this = no observed support: extrapolation, not evidence
MIN_OVERLAP = 0.60          # share of held-out decisions where every compared policy is supported
PROMO_MIN_DATES, PROMO_MIN_GAMES, PROMO_MIN_ROWS = 2, 8, 40
PROMO_LCB_PCT = 5           # 90% one-sided lower bound of the clustered bootstrap of V(challenger) - V(champion)
PROMO_LOGLOSS_TOL, PROMO_MSE_TOL = 0.002, 0.05
N_DELTA_BOOT = 2000


def behaviour_probs(r):
    """pi_b(q | X) over the logged quote ladder. v2026.38 quotes log the full distribution (pi_b); older rows only the
    probability of the action actually taken, so only that one level has known support."""
    raw = getattr(r, "pi_b", None)
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            return {int(k): float(v) for k, v in json.loads(raw).items()}
        except (ValueError, TypeError):
            pass
    return {int(r.price): max(1e-6, _num(r, "propensity", 1.0))}


def behaviour_action_probs(r):
    """qp7 behavior probability over exact (price, requested_qty, TTL-bucket) actions."""
    raw = getattr(r, "pi_b_action", None)
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            return {str(k): float(v) for k, v in json.loads(raw).items()}
        except (ValueError, TypeError):
            return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return {}


def policy_rows(model, qte, extra=None):
    """Per held-out decision: the doubly-robust value of each target policy and whether it has observed support.
    Policies: the model's argmax QEV ('model'), bid+1 (join when spread < 2), join-the-bid, plus `extra` models.
    DR term: Qhat(x, pi) + 1{a = pi} / pi_b(a | x) x (r - Qhat(x, a)), counted only where pi_b(pi | x) >= SUPPORT_EPS."""
    md = _model_dict(model) if not isinstance(model.get("fill"), list) else model
    extra = extra or {}
    rows = []
    for r in qte.itertuples():
        if np.isnan(r.label):
            continue
        alpha = 100.0 * (_num(r, "p_used") - _num(r, "p_mkt"))
        fee = _num(r, "fee_c")
        pib = behaviour_probs(r)
        pia = behaviour_action_probs(r)
        lad = sorted(set(ladder_of(r)) | set(pib))
        bid, ask = side_book_of(r)
        a = int(r.price)
        qty = max(1, int(_num(r, "requested_qty", 1.0)))
        ttl = _num(r, "ttl", 8.0)
        pa = pia.get(action_key(a, qty, ttl), 0.0)
        pa_safe = max(1e-6, pa)
        out = dict(game=r.game, date=r.date)
        for name, mdl in [("model", md)] + list(extra.items()):
            q = {px: XM.quote_value(mdl, features_at(r, px), alpha, fee)["qev"] for px in lad}
            tgt = {"model": max(q, key=q.get)} if name == "model" else {name: max(q, key=q.get)}
            if name == "model":
                tgt.update(bid1=(bid if ask - bid < 2 else bid + 1), join=bid)
            resid = (_num(r, "m30_all", np.nan) - float(XM.standardize(mdl["scaler"], features_at(r, a)) @ np.asarray(mdl["drift"]))) \
                if r.label == 1 else 0.0
            rew = float(r.label) * (alpha + (0.0 if not np.isfinite(resid) else resid) - fee)
            qa = q.get(a, XM.quote_value(mdl, features_at(r, a), alpha, fee)["qev"])
            for k, p_ in tgt.items():
                qp = q.get(p_, XM.quote_value(mdl, features_at(r, p_), alpha, fee)["qev"])
                out[k] = qp + (rew - qa) * float(a == p_) / pa_safe
                out[f"{k}_sup"] = bool(pia.get(action_key(int(p_), qty, ttl), 0.0) >= SUPPORT_EPS)
                out[f"{k}_px"] = int(p_)
                out[f"{k}_action"] = action_key(int(p_), qty, ttl)
        rows.append(out)
    return pd.DataFrame(rows)


def policy_values(model, qte):
    """{policy: (mean, s.e., n_supported, overlap share)} on the decisions where that policy has observed support."""
    pr = policy_rows(model, qte)
    res = {}
    for k in ("model", "bid1", "join"):
        if pr.empty or k not in pr:
            res[k] = (None, None, 0, 0.0)
            continue
        v = pr.loc[pr[f"{k}_sup"], k]
        res[k] = (float(v.mean()) if len(v) else None, float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                  int(len(v)), float(pr[f"{k}_sup"].mean()))
    return res


def clustered_delta(d: pd.DataFrame, col_a: str, col_b: str, seed=0):
    """Mean and 90% one-sided lower bound of V(a) - V(b) over decisions where BOTH have support, bootstrapped by game."""
    sub = d[d[f"{col_a}_sup"] & d[f"{col_b}_sup"]]
    out = dict(rows=int(len(sub)), games=int(sub["game"].nunique()) if len(sub) else 0,
               dates=int(sub["date"].nunique()) if len(sub) else 0, overlap=float(len(sub) / max(1, len(d))), mean=None, lcb90=None)
    if len(sub) < 2:
        return out
    diff = (sub[col_a] - sub[col_b]).values
    out["mean"] = float(diff.mean())
    g = sub["game"].values
    games = np.unique(g)
    if len(games) < 2:
        return out
    idx = [np.where(g == x)[0] for x in games]
    sums, cnts = np.array([diff[i].sum() for i in idx]), np.array([len(i) for i in idx])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(games), (N_DELTA_BOOT, len(games)))
    boots = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    out["lcb90"] = float(np.percentile(boots, PROMO_LCB_PCT))
    return out


def rolling_validation(q):
    dates = sorted(q["date"].unique())
    folds = []
    for k in range(1, len(dates)):
        tr, te = q[q["date"].isin(dates[:k])], q[q["date"] == dates[k]]
        if tr["label"].notna().sum() < 20 or te["label"].notna().sum() < 5:
            continue
        m = fit_all(tr)
        Xte = XM.standardize(m["scaler"], design(te))
        lab = te["label"].values
        kn = ~np.isnan(lab)
        pf = 1.0 / (1.0 + np.exp(-np.clip(Xte @ m["fill"], -30, 30)))
        base = np.nanmean(tr["label"].values)
        f = dict(test_date=dates[k], n_test=int(kn.sum()), fills=int(np.nansum(lab)),
                 logloss=logloss(pf[kn], lab[kn]), logloss_const=logloss(np.full(kn.sum(), base), lab[kn]),
                 brier=float(np.mean((pf[kn] - lab[kn]) ** 2)), brier_const=float(np.mean((base - lab[kn]) ** 2)))
        if kn.sum() >= 10 and 0 < lab[kn].sum() < kn.sum():
            f["calib_intercept"], f["calib_slope"] = calib_slope(pf[kn], lab[kn])
        bins = pd.qcut(pf[kn], q=min(5, max(1, int(kn.sum() // 10))), duplicates="drop") if kn.sum() >= 10 else None
        if bins is not None:
            f["reliability"] = [dict(pred=float(pf[kn][bins == b].mean()), obs=float(lab[kn][bins == b].mean()), n=int((bins == b).sum()))
                                for b in bins.categories]
        drift_te = Xte @ m["drift"]
        has = (lab == 1) & te["m30_all"].notna().values
        if has.sum():
            y = te["m30_all"].values[has] - drift_te[has]
            pred = Xte[has] @ m["mark_all"]
            ytr = tr["m30_all"].values - XM.standardize(m["scaler"], design(tr)) @ m["drift"]
            const = np.nanmean(ytr[(tr["label"].values == 1) & tr["m30_all"].notna().values]) if m["n_mark_all"] else 0.0
            f.update(mse=float(np.mean((pred - y) ** 2)), mse_const=float(np.mean((const - y) ** 2)),
                     mae=float(np.mean(np.abs(pred - y))), bias=float(np.mean(pred - y)), n_marks=int(has.sum()))
        f["policy"] = policy_values(m, te)
        f["_rows"] = policy_rows(m, te)
        folds.append(f)
    return folds


def summarize_folds(folds):
    if not folds:
        return None
    n = sum(f["n_test"] for f in folds)
    agg = lambda k: sum(f[k] * f["n_test"] for f in folds) / n
    out = dict(folds=len(folds), n_test=n, fills=sum(f["fills"] for f in folds), logloss=agg("logloss"),
               logloss_const=agg("logloss_const"), brier=agg("brier"), brier_const=agg("brier_const"))
    mk = [f for f in folds if "mse" in f]
    if mk:
        nm = sum(f["n_marks"] for f in mk)
        out.update(mse=sum(f["mse"] * f["n_marks"] for f in mk) / nm, mse_const=sum(f["mse_const"] * f["n_marks"] for f in mk) / nm, n_marks=nm)
    sl = [f["calib_slope"] for f in folds if "calib_slope" in f]
    out["calib_slope"] = float(np.median(sl)) if sl else None
    rows = pd.concat([f["_rows"] for f in folds if len(f.get("_rows", []))], ignore_index=True) if folds else pd.DataFrame()
    if len(rows):
        out["delta_vs_bid1"] = clustered_delta(rows, "model", "bid1")
        out["delta_vs_join"] = clustered_delta(rows, "model", "join")
        out["overlap_model"] = float(rows["model_sup"].mean())
    for k in ("model", "bid1", "join"):
        vs = [(f["policy"][k][0], f["policy"][k][2]) for f in folds if f["policy"][k][0] is not None]
        nn = sum(c for _, c in vs)
        out[f"value_{k}"] = sum(v * c for v, c in vs) / nn if nn else None
        ses = [f["policy"][k][1] for f in folds if f["policy"][k][1] is not None]
        out[f"value_{k}_se"] = float(np.sqrt(np.mean(np.square(ses)) / max(1, len(ses)))) if ses else None
    return out


def bootstrap_coefs(q, m_full, seed=0):
    """Game-bootstrap coefficient draws (scaler and drift held at the deployed fit): quote-value uncertainty."""
    rng = np.random.default_rng(seed)
    games = q["game"].unique()
    if len(games) < 3:
        return [], []
    by = {g: q.index[q["game"] == g].values for g in games}
    X = XM.standardize(m_full["scaler"], design(q))
    lab = q["label"].values
    w = 1.0 / np.clip(pd.to_numeric(q.get("propensity", 1.0), errors="coerce").fillna(1.0).values, 0.05, 1.0)
    y_m = q["m30_all"].values - X @ np.asarray(m_full["drift"])
    bf, bm = [], []
    for _ in range(N_BOOT):
        ix = np.concatenate([by[g] for g in rng.choice(games, len(games), replace=True)])
        kn = ~np.isnan(lab[ix])
        if kn.sum() < 10 or lab[ix][kn].sum() < 2:
            continue
        bf.append(list(map(float, fit_logit(X[ix][kn], lab[ix][kn], w[ix][kn]))))
        has = (lab[ix] == 1) & np.isfinite(y_m[ix])
        bm.append(list(map(float, fit_ridge(X[ix][has], y_m[ix][has], w[ix][has]))) if has.sum() >= 5 else list(map(float, m_full["mark_all"])))
    return bf, bm


def challenger_vs_champion(champ: dict, chall: dict, later: pd.DataFrame) -> dict:
    """Promotion (v2026.38) needs ALL of: enough unseen dates/games/decisions, joint action support, a positive clustered
    90% lower bound on V(challenger) - V(champion), no worse fill calibration, no worse markout prediction."""
    checks, delta = {}, {}
    rows = policy_rows(champ, later, extra={"challenger": chall}) if len(later) else pd.DataFrame()
    if len(rows):
        delta = clustered_delta(rows, "challenger", "model")
    checks[f">= {PROMO_MIN_DATES} unseen dates"] = delta.get("dates", 0) >= PROMO_MIN_DATES
    checks[f">= {PROMO_MIN_GAMES} unseen games"] = delta.get("games", 0) >= PROMO_MIN_GAMES
    checks[f">= {PROMO_MIN_ROWS} jointly supported decisions"] = delta.get("rows", 0) >= PROMO_MIN_ROWS
    checks[f"action overlap >= {MIN_OVERLAP:.0%}"] = delta.get("overlap", 0.0) >= MIN_OVERLAP
    checks["delta mean > 0"] = (delta.get("mean") or 0.0) > 0
    checks["delta 90% lower bound > 0"] = delta.get("lcb90") is not None and delta["lcb90"] > 0
    kn = later[later["label"].notna()] if len(later) else later
    if len(kn) >= 10:
        lab = kn["label"].values
        def ll(mdl):
            pf = 1.0 / (1.0 + np.exp(-np.clip(XM.standardize(mdl["scaler"], design(kn)) @ np.asarray(mdl["fill"]), -30, 30)))
            return logloss(pf, lab)
        def mse(mdl):
            has = (lab == 1) & kn["m30_all"].notna().values
            if has.sum() < 5:
                return None
            X = XM.standardize(mdl["scaler"], design(kn))
            y = kn["m30_all"].values[has] - X[has] @ np.asarray(mdl["drift"])
            return float(np.mean((X[has] @ np.asarray(mdl["mark_all"]) - y) ** 2))
        l_c, l_h = ll(chall), ll(champ)
        checks["fill calibration not worse (log loss)"] = l_c <= l_h + PROMO_LOGLOSS_TOL
        m_c, m_h = mse(chall), mse(champ)
        checks["adverse-selection prediction not worse (markout MSE)"] = m_c is None or m_h is None or m_c <= m_h * (1.0 + PROMO_MSE_TOL)
    else:
        checks["fill calibration not worse (log loss)"] = False
        checks["adverse-selection prediction not worse (markout MSE)"] = False
    return dict(promote=all(checks.values()), checks=checks, delta=delta)


def main(here=HERE):
    q = load(here)
    out = dict(quote_policy=QUOTE_POLICY, features=EXEC_FEATURES, fitted=datetime.now().isoformat(timespec="seconds"),
               validated=False, spec=dict(L2=L2, labels="censored", markout="residual vs drift; all fills = production"))
    if q.empty:
        print(f"No LIVE {QUOTE_POLICY} quotes yet: nothing to fit (run --pilot-live to collect them).")
        json.dump(out, open(os.path.join(here, os.path.basename(CHALLENGER)), "w"), indent=2)
        return out
    n_q, kn = len(q), q["label"].notna()
    n_f, n_cens = int((q["label"] == 1).sum()), int((~kn).sum())
    n_m = int(((q["label"] == 1) & q["m30_all"].notna()).sum())
    dates = sorted(q["date"].unique())
    print(f"LIVE {QUOTE_POLICY} quotes {n_q} | filled {n_f} | terminal unfilled {int((q['label'] == 0).sum())} | "
          f"CENSORED (not terminal: excluded) {n_cens} | 30s marks on fills {n_m} | dates {len(dates)}")
    out.update(n_quotes=n_q, n_fills=n_f, n_censored=n_cens, n_marks=n_m, trained_through=dates[-1])
    folds = rolling_validation(q)
    summ = summarize_folds(folds)
    out["validation"] = dict(folds=[{k: v for k, v in f.items() if k != "_rows"} for f in folds], summary=summ)
    if summ:
        print(f"Rolling walk-forward ({summ['folds']} folds, {summ['n_test']} held-out quotes, {summ['fills']} fills): "
              f"fill log loss {summ['logloss']:.4f} vs constant {summ['logloss_const']:.4f} | Brier {summ['brier']:.4f} vs "
              f"{summ['brier_const']:.4f} | calibration slope {summ['calib_slope']}")
        if "mse" in summ:
            print(f"   markout residual MSE {summ['mse']:.2f} vs constant {summ['mse_const']:.2f} ({summ['n_marks']} marks)")
        print(f"   POLICY VALUE (doubly robust on supported decisions, cents/quote): model {summ['value_model']} | "
              f"bid+1 {summ['value_bid1']} | join {summ['value_join']} | model support {summ.get('overlap_model', 0):.0%}")
        for k_ in ("delta_vs_bid1", "delta_vs_join"):
            d_ = summ.get(k_) or {}
            print(f"   model - {k_[9:]}: mean {d_.get('mean')} | 90% LB {d_.get('lcb90')} (game bootstrap) | "
                  f"{d_.get('rows', 0)} decisions with joint support ({d_.get('overlap', 0):.0%}), {d_.get('games', 0)} games")
    enough = n_q >= MIN_QUOTES and n_f >= MIN_FILLS and n_m >= MIN_MARKS and len(dates) >= MIN_DATES
    ok = bool(enough and summ and summ["fills"] >= MIN_VAL_FILLS and summ["logloss"] <= summ["logloss_const"]
              and ("mse" not in summ or summ["mse"] <= summ["mse_const"])
              and summ["value_model"] is not None and summ["value_model"] > 0
              and summ.get("overlap_model", 0.0) >= MIN_OVERLAP
              and all((summ.get(k_) or {}).get("lcb90") is not None and summ[k_]["lcb90"] > 0
                      and summ[k_]["overlap"] >= MIN_OVERLAP for k_ in ("delta_vs_bid1", "delta_vs_join"))
              and (summ["calib_slope"] is None or 0.5 <= summ["calib_slope"] <= 2.0))
    if not enough:
        print(f"Not enough evidence to validate (need >= {MIN_QUOTES} quotes, {MIN_FILLS} fills, {MIN_MARKS} marks, "
              f"{MIN_DATES}+ dates; have {n_q}, {n_f}, {n_m}, {len(dates)}).")
    m = fit_all(q)
    md = _model_dict(m)
    bf, bm = bootstrap_coefs(q, md)
    md.update(boot_fill=bf, boot_mark=bm)
    pv, pse = (summ or {}).get("value_model"), (summ or {}).get("value_model_se")
    explore_scale = float(np.clip(pse / (abs(pv) + pse), 0.1, 1.0)) if (pv is not None and pse) else 1.0
    # observed action support by quote offset (price - side bid): PRODUCTION quotes are restricted to these in the trader;
    # anything else is extrapolation and may only be tried as PILOT exploration
    off = q.apply(lambda r: int(r["price"]) - side_book_of(r)[0], axis=1) if len(q) else pd.Series(dtype=int)
    cnt = off.value_counts()
    supported = sorted(int(k) for k, v in cnt.items() if v >= MIN_SUPPORT_ACTIONS)
    out.update(md, validated=ok, explore_scale=round(explore_scale, 3), n_mark_all=m["n_mark_all"], n_mark_ef=m["n_mark_ef"],
               supported_offsets=supported, offset_counts={str(int(k)): int(v) for k, v in cnt.items()})
    print(f"Action support by quote offset (logged LIVE actions): {dict(cnt.sort_index())} -> supported for PRODUCTION: {supported}")
    out["exec_model_id"] = XM.model_id(out)
    with open(os.path.join(here, os.path.basename(CHALLENGER)), "w") as f:
        json.dump(out, f, indent=2)
    print(f"challenger_exec_model.json written ({out['exec_model_id']}, {'VALIDATED' if ok else 'not validated'}; "
          f"exploration scale {explore_scale:.2f})")
    champ_path = os.path.join(here, os.path.basename(CHAMPION))
    champ = None
    try:
        champ = json.load(open(champ_path))
    except Exception:
        pass
    promote, promo = False, {}
    if ok:
        if not champ or not champ.get("validated"):
            promote = True          # first champion: validation above already required LB90 > 0 vs both baselines
            promo = dict(reason="no validated champion")
        else:
            promo = challenger_vs_champion(champ, out, q[q["date"] > str(champ.get("trained_through", ""))])
            promote = promo["promote"]
            print(f"Champion {champ.get('exec_model_id')} vs challenger {out['exec_model_id']} on decisions the champion never saw:")
            for k_, v_ in promo["checks"].items():
                print(f"   {'ok ' if v_ else 'no '} {k_}")
            print(f"   V(challenger) - V(champion): mean {promo['delta'].get('mean')} | 90% LB {promo['delta'].get('lcb90')} -> "
                  f"{'PROMOTE' if promote else 'champion kept'}")
    out["promotion"] = promo
    with open(os.path.join(here, os.path.basename(CHALLENGER)), "w") as f:
        json.dump(out, f, indent=2, default=str)
    if promote:
        if champ:
            shutil.copy(champ_path, champ_path.replace(".json", "_prev.json"))
        shutil.copy(os.path.join(here, os.path.basename(CHALLENGER)), champ_path)
        print(f"PROMOTED: challenger {out['exec_model_id']} is the new champion (the trader uses it from its next start).")
    return out


if __name__ == "__main__":
    main()