"""Execution model shared by the trader and fit_execution.py (v2026.40 / qp8): ONE feature definition and ONE scoring path, so
a model is always scored on exactly the features, and the training-only scaler, it was fitted with.

Model objects (champion_exec_model.json / challenger_exec_model.json):
    features      EXEC_FEATURES (must match exactly)
    scaler        {"mean": [...], "sd": [...]} from TRAINING rows only; binary columns are left unscaled (mean 0, sd 1)
    fill          logistic coefficients: P(fill before the quote's own TTL | quote, state)   (censored rows excluded)
    mark_all      E[execution residual over 30 s | fill]  ALL fills  -> production quote economics
    mark_ef       same, event-free fills only                         -> microstructure diagnostic
    drift         E[market move over 30 s | quote, state] for comparable UNFILLED quotes: the counterfactual used to
                  residualize markouts, so directional alpha is not counted twice
    boot_fill / boot_mark   coefficient draws from a game/date bootstrap: quote-value uncertainty
"""
import hashlib
import json
import math
from typing import Dict, Optional, Tuple

import numpy as np

STOP_TYPES = ("timeout", "substitution", "freethrow", "period", "instantreplay")
EXEC_FEATURES = ["const", "side_no", "aggr", "spread", "price", "gap_c", "ttl_s",
                 "stop_timeout", "stop_substitution", "stop_freethrow", "stop_period", "stop_instantreplay",
                 "ph_q1", "ph_q2", "ph_ht", "ph_q4", "stat_reb", "stat_ast",
                 "log_qbid", "log_qask", "imbalance", "log_queue_ahead", "order_qty", "qty_over_best_depth",
                 "qty_over_queue_ahead", "log_order_qty", "book_age_s", "t_since_action_s",
                 "mid_vel_250ms", "mid_vel_1s", "spread_vel", "imb_vel", "bid_depletion", "ask_replenish", "has_ws"]
BINARY = {"const", "side_no", "has_ws"} | {f for f in EXEC_FEATURES if f.startswith(("stop_", "ph_", "stat_"))}
GAP_COL = EXEC_FEATURES.index("gap_c")
MICRO_KEYS = ("mid_vel_250ms", "mid_vel_1s", "spread_vel", "imb_vel", "bid_depletion", "ask_replenish")


def stop_type_of(last_action_type: str) -> str:
    t = (last_action_type or "").lower()
    for k in STOP_TYPES:
        if k in t:
            return k
    return "other"


def exec_features(side: str, stat: str, period: int = 0, bid: int = 0, ask: int = 0, price: int = 0, gap_c: float = 0.0,
                  ttl_s: float = 8.0, stop: str = "other", phase: Optional[str] = None, qty_bid: float = 0.0,
                  qty_ask: float = 0.0, queue_ahead: float = 0.0, book_age_s: float = 0.0, t_since_action_s: float = 0.0,
                  micro: Optional[dict] = None, order_qty: float = 1.0, best_depth: float = 0.0) -> np.ndarray:
    """Raw (unscaled) features of one candidate quote. bid/ask are the SIDE's book (for NO: 100 - yes_ask / 100 - yes_bid)
    and qty_bid/qty_ask the depth at that side's best bid / best ask. aggr = (price - bid) / max(ask - bid, 1) replaces
    the collinear (level, distance-to-mid) pair. gap_c = 100 x (p_used - p_mkt) on that side."""
    micro = micro or {}
    if phase is None:
        phase = "Q1" if (period or 0) <= 1 else ("Q2" if period == 2 else ("Q3" if period == 3 else "Q4"))
    qb, qa = max(0.0, float(qty_bid or 0.0)), max(0.0, float(qty_ask or 0.0))
    spread = float(ask - bid)
    vals = {"const": 1.0, "side_no": float(side == "no"), "aggr": (price - bid) / max(spread, 1.0), "spread": spread,
            "price": price / 100.0, "gap_c": float(gap_c), "ttl_s": float(ttl_s or 0.0),
            **{f"stop_{k}": float(stop == k) for k in STOP_TYPES},
            "ph_q1": float(phase == "Q1"), "ph_q2": float(phase == "Q2"), "ph_ht": float(phase == "HT"), "ph_q4": float(phase == "Q4"),
            "stat_reb": float(stat == "REB"), "stat_ast": float(stat == "AST"),
            "log_qbid": math.log1p(qb), "log_qask": math.log1p(qa), "imbalance": (qb - qa) / max(qb + qa, 1e-9),
            "log_queue_ahead": math.log1p(max(0.0, float(queue_ahead or 0.0))),
            "order_qty": max(1.0, float(order_qty or 1.0)),
            "qty_over_best_depth": max(1.0, float(order_qty or 1.0)) / max(float(best_depth or qb or 0.0), 1.0),
            "qty_over_queue_ahead": max(1.0, float(order_qty or 1.0)) / max(float(queue_ahead or 0.0), 1.0),
            "log_order_qty": math.log1p(max(1.0, float(order_qty or 1.0))),
            "book_age_s": float(book_age_s or 0.0), "t_since_action_s": min(120.0, float(t_since_action_s or 0.0)),
            **{k: float(micro.get(k) or 0.0) for k in MICRO_KEYS}, "has_ws": float(bool(micro.get("has_ws")))}
    return np.array([vals[f] for f in EXEC_FEATURES], dtype=float)


def fit_scaler(X: np.ndarray) -> Dict[str, list]:
    """Mean / s.d. per column from the rows given (TRAINING rows only). Binary columns: (0, 1), i.e. untouched."""
    mean, sd = X.mean(axis=0), X.std(axis=0)
    for j, f in enumerate(EXEC_FEATURES):
        if f in BINARY or sd[j] < 1e-9:
            mean[j], sd[j] = (0.0, 1.0) if f in BINARY else (mean[j], 1.0)
    return {"mean": [float(v) for v in mean], "sd": [float(v) for v in sd]}


def standardize(scaler: dict, X: np.ndarray) -> np.ndarray:
    return (np.asarray(X, float) - np.asarray(scaler["mean"])) / np.asarray(scaler["sd"])


def model_id(model: dict) -> str:
    """Content hash of what the model predicts with (features, scaler, coefficients, training spec)."""
    keep = {k: model.get(k) for k in ("features", "scaler", "fill_any", "fill_fraction", "filled_qty", "fill", "mark_all", "mark_ef", "drift", "spec", "trained_through")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def quote_value(model: dict, x_raw: np.ndarray, alpha_c: float, fee_c: float, z: float = 0.0) -> dict:
    """P(fill), E[execution residual | fill] and QuoteEV = P(fill) x (alpha + residual - fee), each with a bootstrap
    s.d. when the model carries coefficient draws. qev_lcb = mean - z x sd."""
    xs = standardize(model["scaler"], x_raw)
    bf = [np.asarray(model.get("fill_any", model.get("fill")), float)] + [np.asarray(b, float) for b in (model.get("boot_fill") or [])]
    bm = [np.asarray(model["mark_all"], float)] + [np.asarray(b, float) for b in (model.get("boot_mark") or [])]
    k = min(len(bf), len(bm))
    pf = np.array([float(_sig(b @ xs)) for b in bf[:k]])
    em = np.array([float(b @ xs) for b in bm[:k]])
    bfrac = np.asarray(model.get("fill_fraction", model.get("fill_any", model.get("fill"))), float)
    pfrac = float(_sig(bfrac @ xs))
    qty_raw = max(1.0, float(np.asarray(x_raw, float)[EXEC_FEATURES.index("order_qty")]))
    expected_filled_qty = float(np.clip(qty_raw * pfrac, 0.0, qty_raw))
    if model.get("filled_qty") is not None:
        direct_filled_qty = float(np.clip(np.asarray(model["filled_qty"], float) @ xs, 0.0, qty_raw))
    else:
        direct_filled_qty = expected_filled_qty
    # Economic value is per REQUESTED contract, so partial-fill fraction—not probability of any fill—is
    # the correct multiplier. Also expose total expected value for the exact requested quantity.
    edge_draw = alpha_c + em - fee_c
    qev = pfrac * edge_draw
    qev_total = expected_filled_qty * edge_draw
    use = slice(1, None) if k > 1 else slice(0, 1)
    sd = float(np.std(qev[use], ddof=1)) if k > 2 else 0.0
    return dict(pf=float(pf[0]), fill_fraction=pfrac, expected_filled_qty=expected_filled_qty,
                direct_filled_qty=direct_filled_qty, em=float(em[0]), qev=float(qev[0]),
                qev_total=float(qev_total[0]), qev_sd=sd, qev_lcb=float(qev[0]) - z * sd,
                pf_sd=float(np.std(pf[use], ddof=1)) if k > 2 else 0.0,
                em_sd=float(np.std(em[use], ddof=1)) if k > 2 else 0.0)


def quote_ev_c(model: dict, x: np.ndarray, alpha_c: float, fee_c: float) -> Tuple[float, float, float]:
    """(P(fill), E[residual | fill], QuoteEV) in cents (compatibility wrapper of quote_value)."""
    q = quote_value(model, x, alpha_c, fee_c)
    return q["pf"], q["em"], q["qev"]


def load_model(path: str, quote_policy: str) -> Optional[dict]:
    """A model file usable for trading: same quote policy, same features, validated, and a scaler. None otherwise."""
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception:
        return None
    if m.get("quote_policy") != quote_policy or not m.get("validated") or m.get("features") != EXEC_FEATURES \
            or not m.get("scaler") or m.get("fill_any", m.get("fill")) is None or m.get("fill_fraction") is None or m.get("mark_all") is None:
        return None
    m.setdefault("exec_model_id", model_id(m))
    return m