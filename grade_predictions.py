#!/usr/bin/env python3
"""Grade every prediction the model logged (traded or not) against Kalshi settlement, and decide
whether the model has earned live trading.

Edge test: whenever the model disagreed with the market mid, did the outcome land on the model's side
more often than the market price implied?
    edge (cents) = mean of  sign(model - market) * (outcome - market) * 100
Positive = trading toward the model beats the market price (before spread and fees). Each market is
weighted equally, and the 90% interval resamples PLAYERS (a player's 20+/25+/30+ ladder moves together).

LIVE GATE (written to model_status.json, read by the trader before --live). Since v2026.30 money is authorized by
execution evidence: the model edge gate below AND settled profit on fills that actually happened (>= 100 fills, 90%
interval above 0) AND event-free 30s maker markouts (>= 30, mean >= 0). Eligible-candidate P&L is research only.
    only predictions logged by v2026.19+ count (older logs include the Nov 5 bugs). They are out of sample by
    construction: any calibration active when they were made was fitted on earlier nights. PASS needs >= 300 settled markets, a 90% interval entirely above 0, and a mean edge of at
    least 2c (roughly the cost of getting filled).
Usage: python3 grade_predictions.py [kalshi_predictions_YYYY-MM-DD.csv ...]   (default: all of them)
"""
import os
import sys
import json
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from kalshi_results import load_predictions, settlements
from nba_markov_engine import ENGINE_VERSION
from execution_size_authority import write_authority, size_bucket
from nightly_reports import write_all as write_nightly_reports

GATE_MARKETS, GATE_EDGE_C, GATE_GAMES = 300, 2.0, 15
QUOTE_POLICY = "qp8"   # must match kalshi_nba_inplay_trader.QUOTE_POLICY: priors learned under another policy are not reused
HERE = os.path.dirname(os.path.abspath(__file__))


def current_policy(here: str = HERE) -> dict:
    """Policy identity of the newest run_config_*.json (what the next run will most likely trade). Evidence for the fill
    gate is pooled by policy FAMILY (every economic parameter + execution model; the nightly-refit execution prior is
    excluded so evidence can accumulate across nights) and reported by exact policy_id."""
    import glob
    for path in sorted(glob.glob(os.path.join(here, "run_config_*.json")))[::-1]:
        try:
            c = json.load(open(path))
        except Exception:
            continue
        if c.get("policy_family_id"):
            return dict(policy_id=c.get("policy_id"), policy_family_id=c.get("policy_family_id"), source=os.path.basename(path))
    return dict(policy_id=None, policy_family_id=None, source="none")
GATE_ELIGIBLE = 150    # strategy gate: settled trade-eligible (market, side) candidates from this engine
GATE_ORDERS, GATE_ORDER_GAMES, GATE_CONTRACTS = 100, 10, 250   # fill gate: settled LIVE entry ORDERS (partial fills of one order
                                                              # are one observation), across games, with enough contracts
GATE_MARK_ORDERS, GATE_MARK_GAMES = 50, 10    # markout gate: LIVE maker orders with a 30s mark; contract-weighted mean >= 0
GATE_FILLS, GATE_MARKS = GATE_ORDERS, GATE_MARK_ORDERS          # names kept for older callers


def brier(p, y, w):
    return float(np.sum(w * (np.clip(p, 0, 1) - y) ** 2) / np.sum(w))


def edge_report(d, label):
    d = d.copy()
    d["edge"] = np.sign(d["p_model_yes"] - d["p_mkt_yes"]) * (d["y"] - d["p_mkt_yes"]) * 100.0
    per = d.groupby("ticker").agg(edge=("edge", "mean"), player=("player", "first"), game=("game", "first"))
    n_games = per["game"].nunique()
    # Resample whole GAMES once there are enough of them (predictions inside a game are strongly correlated);
    # with few games, resample players, which is still too optimistic, so the gate also requires GATE_GAMES games.
    groups = [g.values for _, g in per.groupby("game" if n_games >= 10 else "player")["edge"]]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [5, 95])
    mean = per["edge"].mean()
    verdict = ("model BEAT the market price" if lo > 0 else "market was SHARPER than the model" if hi < 0
               else "no detectable edge either way (interval spans 0)")
    print(f"   {label}: {mean:+.2f}c per contract (90% CI {lo:+.2f} to {hi:+.2f}; {len(groups)} players, {len(per)} markets)")
    print(f"   -> {verdict}")
    return dict(markets=int(len(per)), players=int(per["player"].nunique()), games=int(n_games), edge_c=round(float(mean), 3),
                ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3)), d


def strategy_report(res: dict) -> dict:
    """The gate that matters for money: settle every trade-ELIGIBLE candidate (all filters passed, at the price we would
    have paid, after fees) and measure net profit per contract. A model can be mediocre overall and profitable on the
    subset it trades, or the reverse."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "kalshi_eligible_*.csv")))
    if not files:
        print("\n9) STRATEGY SUBSET: no kalshi_eligible_*.csv yet")
        return {"passed": False, "candidates": 0}
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    e = e[e["engine"].astype(str) == ENGINE_VERSION] if "engine" in e.columns else e.iloc[0:0]
    if "passes_conservative" in e.columns:
        pc = e["passes_conservative"].astype(str).str.lower().isin(["true", "1"])
        dropped = e[~pc].drop_duplicates(["ticker", "side"], keep="first")
        dropped = dropped.assign(y=dropped["ticker"].map(res).map({"yes": 1.0, "no": 0.0})).dropna(subset=["y"])
        if len(dropped):
            w = np.where(dropped["side"] == "yes", dropped["y"], 1.0 - dropped["y"])
            net = pd.Series(w * 100.0 - dropped["quote_price"].values - dropped["fee_c"].values)
            lo, hi = _game_boot(net, pd.Series(dropped["ticker"].str.split("-").str[1].values))
            print(f"\n9b) Counterfactual: {len(dropped)} candidates the v2026.28 rule would take but the current rule drops "
                  f"(edge mostly spread capture): net {net.mean():+.2f}c (90% CI {lo:+.2f} to {hi:+.2f}) at the quote price, "
                  f"BEFORE any adverse selection on fills. Clearly positive over many games = the give-back prior is too strict.")
        e = e[pc]
    e = e.drop_duplicates(["ticker", "side"], keep="first")          # first time each side became eligible
    e["y"] = e["ticker"].map(res).map({"yes": 1.0, "no": 0.0})
    e = e.dropna(subset=["y"])
    if e.empty:
        print(f"\n9) STRATEGY SUBSET: no settled eligible candidates from engine {ENGINE_VERSION}")
        return {"passed": False, "candidates": 0}
    win = np.where(e["side"] == "yes", e["y"], 1.0 - e["y"])
    e["net_c"] = win * 100.0 - e["quote_price"] - e["fee_c"]
    e["game_tok"] = e["ticker"].str.split("-").str[1]
    groups = [g.values for _, g in e.groupby("game_tok")["net_c"]]
    rng = np.random.default_rng(1)
    boots = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(2000)] if len(groups) > 1 else [e["net_c"].mean()]
    lo, hi = np.percentile(boots, [5, 95])
    ok = bool(len(e) >= GATE_ELIGIBLE and lo > 0)
    print(f"\n9) STRATEGY SUBSET (RESEARCH ONLY: eligible candidates at the quote price, ignoring which would have filled): "
          f"{len(e)} candidates, {len(groups)} games")
    print(f"   net {e['net_c'].mean():+.2f}c per contract (90% CI {lo:+.2f} to {hi:+.2f}); predicted net edge was {e['net_edge_c'].mean():+.2f}c")
    print(e.groupby("stat").agg(n=("net_c", "size"), net_c=("net_c", "mean"), predicted_c=("net_edge_c", "mean")).round(2).to_string())
    return {"passed": ok, "candidates": int(len(e)), "games": int(len(groups)), "net_c": round(float(e["net_c"].mean()), 3),
            "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3)}


def _game_boot(values: pd.Series, games: pd.Series, seed: int = 2):
    groups = [g.values for _, g in values.groupby(games)]
    if len(groups) < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [5, 95])
    return float(lo), float(hi)


def fill_report(res: dict) -> dict:
    """The question that decides money for a maker: E[profit | we were actually filled]. Maker fills are selected (we get
    filled most when our price is stale), so a policy can be profitable over eligible candidates and lose on fills.
    Each entry contract is settled (100 if its side won) minus price and fee paid; 90% interval resamples games."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "kalshi_fills_*.csv")))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    if not frames:
        print("\n10) FILLS: no kalshi_fills_*.csv yet")
        return {"passed": False, "fills": 0}
    e = pd.concat(frames, ignore_index=True)
    e = e[pd.to_numeric(e["contracts"], errors="coerce") > 0].copy()
    if "engine" not in e.columns:
        e["engine"] = None
    if "mode" not in e.columns:
        e["mode"] = ""
    # Only LIVE fills can authorize live trading: paper fills are simulated (the market trading through our price) and
    # say nothing about queue position, partial fills or who chose to trade with us.
    pol = current_policy()
    for c_ in ("quote_policy", "arm", "policy_family_id", "policy_id", "tier"):
        if c_ not in e.columns:
            e[c_] = None
    live_cur = e[(e["engine"].astype(str) == ENGINE_VERSION) & (e["mode"].astype(str).str.upper() == "LIVE")]
    # v2026.37: production evidence is policy-specific: same quote policy, BASE arm only (exploration arms are reported
    # separately and never certify the base policy), same policy family as the run being authorized
    cur = live_cur[(live_cur["quote_policy"].astype(str) == QUOTE_POLICY) & (live_cur["arm"].fillna("base").astype(str) == "base")
                   & (live_cur["policy_family_id"].astype(str) == str(pol["policy_family_id"]))].copy()
    arms = live_cur[(live_cur["quote_policy"].astype(str) == QUOTE_POLICY) & (live_cur["arm"].fillna("base").astype(str) != "base")].copy()
    e["y"] = e["ticker"].map(res).map({"yes": 1.0, "no": 0.0})
    cur["y"] = cur["ticker"].map(res).map({"yes": 1.0, "no": 0.0})
    arms["y"] = arms["ticker"].map(res).map({"yes": 1.0, "no": 0.0})
    def summarise(d, label):
        d = d.dropna(subset=["y"]).copy()
        if d.empty:
            print(f"   {label}: no settled entry fills")
            return None
        win = np.where(d["side"] == "yes", d["y"], 1.0 - d["y"])
        fee_pc = 100.0 * d["fee_usd"].astype(float) / d["contracts"].astype(float)
        d["net_c"] = win * 100.0 - d["price_cents"].astype(float) - fee_pc
        d["exp_c"] = 100.0 * d["p_used"].astype(float) - d["price_cents"].astype(float) - fee_pc
        d["game_tok"] = d["ticker"].str.split("-").str[1]
        w = d["contracts"].astype(float)
        # one observation per ORDER (a resting order filled in 15 pieces is one decision, not 15)
        oid = d["order_id"].astype(str) if "order_id" in d.columns else d.index.astype(str)
        orders = d.assign(_o=oid, _w=w, _nw=d["net_c"] * w, _ew=d["exp_c"] * w).groupby("_o").agg(
            w=("_w", "sum"), nw=("_nw", "sum"), ew=("_ew", "sum"), game=("game_tok", "first"))
        net = float(orders["nw"].sum() / orders["w"].sum())
        groups = [g for _, g in orders.groupby("game")]
        if len(groups) > 1:
            rng = np.random.default_rng(2)
            bs = []
            for _ in range(2000):
                pick = pd.concat([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                bs.append(pick["nw"].sum() / pick["w"].sum())
            lo, hi = (float(x) for x in np.percentile(bs, [5, 95]))
        else:
            lo = hi = net
        print(f"   {label}: {len(orders)} orders ({len(d)} fill messages), {int(w.sum())} contracts, {orders['game'].nunique()} games | "
              f"net {net:+.2f}c per contract (90% CI {lo:+.2f} to {hi:+.2f}) | expected at fill "
              f"{orders['ew'].sum() / orders['w'].sum():+.2f}c | P&L ${float(orders['nw'].sum()) / 100:+.2f}")
        if "order_type" in d.columns:
            print(d.groupby("order_type").apply(lambda x: pd.Series(dict(
                fills=len(x), net_c=np.average(x.net_c, weights=x.contracts), expected_c=np.average(x.exp_c, weights=x.contracts)))
            ).round(2).to_string())
        return dict(fills=int(len(d)), orders=int(len(orders)), contracts=int(w.sum()), games=int(orders["game"].nunique()),
                    net_c=round(net, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3))
    print("\n10) FILLS (what actually got filled, settled, after fees: the number that decides money)")
    summarise(e, "all engines, paper + live (diagnostic)")
    if len(arms):
        summarise(arms, f"exploration arms, LIVE {QUOTE_POLICY} (diagnostic: never certifies the base policy)")
    if len(live_cur) and live_cur["policy_family_id"].notna().any():
        print("   LIVE fills by policy family / exact policy id (engine " + ENGINE_VERSION + "):")
        print(live_cur.groupby([live_cur["policy_family_id"].astype(str), live_cur["policy_id"].astype(str)]).size().to_string())
    s_cur = summarise(cur, f"engine {ENGINE_VERSION}, LIVE, {QUOTE_POLICY}, base arm, policy family {pol['policy_family_id']} (gate)")
    ok = bool(s_cur and s_cur["orders"] >= GATE_ORDERS and s_cur["games"] >= GATE_ORDER_GAMES
              and s_cur["contracts"] >= GATE_CONTRACTS and s_cur["ci_lo"] > 0)
    print(f"   FILL GATE: needs >= {GATE_ORDERS} live orders, >= {GATE_ORDER_GAMES} games, >= {GATE_CONTRACTS} contracts, "
          f"90% CI above 0 -> {'PASS' if ok else 'not passed'}")
    return dict(s_cur or {"fills": 0, "orders": 0}, passed=ok, policy_family_id=pol["policy_family_id"], policy_source=pol["source"])


def markout_report():
    """Event-free markouts (no NBA play between fill and mark): the clean measure of being picked off. Also estimates
    the share of spread capture given back, which is what --as-spread-frac in the trader should be."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "kalshi_markouts_*.csv")))
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    if not frames:
        print("\n11) MARKOUTS: no kalshi_markouts_*.csv yet")
        return None, {"passed": False, "marks": 0}
    m = pd.concat(frames, ignore_index=True)
    # Execution evidence belongs to the QUOTE POLICY (v2026.36), not the prediction engine: the engine changes which
    # markets are quoted, the policy changes which fills arrive. Filtered by policy below.
    print("\n11) MARKOUTS (cents vs fill price; negative = the market moved against the fill)")
    est = None
    for h in (5, 15, 30, 120):
        col, ev = f"m{h}", f"m{h}_event"
        if col not in m.columns:
            continue
        v = pd.to_numeric(m[col], errors="coerce")
        clean = v[(m[ev].astype(str).str.lower() == "false")] if ev in m.columns else v.iloc[0:0]
        print(f"   {h:>4}s: all {v.mean():+.2f}c (n={v.notna().sum()}) | event-free {clean.mean():+.2f}c (n={clean.notna().sum()})")
        if h == 30 and "p_mkt" in m.columns and clean.notna().sum() >= 20:
            cap = 100.0 * pd.to_numeric(m.loc[clean.index, "p_mkt"], errors="coerce") - pd.to_numeric(m.loc[clean.index, "price"], errors="coerce")
            ok = cap.notna() & clean.notna() & (cap > 0)
            if ok.sum() >= 20:
                est = float(np.clip(((cap[ok] - clean[ok]).mean()) / cap[ok].mean(), 0.0, 1.0))
    if est is not None:
        print(f"   -> spread give-back on event-free 30s marks: {est:.2f} (diagnostic). Keep --as-spread-frac 1 while the side "
              f"execution alphas below are used: the raw markout already contains any retained spread (no double count)")
    else:
        print("   -> not enough event-free 30s marks (need 20) to estimate the spread give-back; keep 1.00 (no spread credit)")
    g = {"passed": False, "marks": 0, "policy_family_id": None, "exec_prior_earned": False}
    if "m30" in m.columns:
        mk = m[m["maker"].astype(str).str.lower().isin(["true", "1"])] if "maker" in m.columns else m
        mk = mk.assign(m30=pd.to_numeric(mk["m30"], errors="coerce")).dropna(subset=["m30"])
        mk = mk.assign(_w=pd.to_numeric(mk["contracts"], errors="coerce").fillna(1.0) if "contracts" in mk.columns else 1.0)
        mean = float(np.average(mk["m30"], weights=mk["_w"])) if len(mk) else None
        # Production evidence: LIVE maker ORDERS only (paper fills are simulated), partial fills merged per order,
        # uncertainty clustered by game. Two views: ALL 30s marks (production safety: fills just before a play the feed
        # showed seconds later are the pick-offs) and event-free marks (clean microstructure / spread give-back).
        live = mk[mk["mode"].astype(str).str.upper() == "LIVE"] if "mode" in mk.columns else mk.iloc[0:0]
        if "quote_policy" in live.columns:
            live = live[live["quote_policy"].astype(str) == QUOTE_POLICY]
        if "arm" in live.columns and len(live):
            # exploration arms (+/-1c) are reported by arm; the side prior is learned from the BASE policy's fills only
            print(f"   By quote arm (LIVE {QUOTE_POLICY}, all 30s marks):")
            print(live.groupby(["side", live["arm"].fillna("base")]).agg(fills=("m30", "size"), m30_c=("m30", "mean")).round(2).to_string())
            live = live[live["arm"].fillna("base").astype(str) == "base"]
        # v2026.38: execution evidence is POLICY-FAMILY specific, exactly like the fill gate. Marks made under another
        # family (other champion execution model or other economic settings) are reported, never used for the gate/prior.
        fam = current_policy().get("policy_family_id")
        if len(live):
            _lf = live["policy_family_id"].astype(str) if "policy_family_id" in live.columns else pd.Series("none", index=live.index)
            other = live[_lf != str(fam)]
            if len(other):
                print(f"   Other policy families (reported only; they do not authorize family {fam}):")
                print(other.assign(_f=_lf[other.index]).groupby(["_f", "side"]).agg(fills=("m30", "size"), m30_c=("m30", "mean")).round(2).to_string())
            live = live[_lf == str(fam)] if fam else live.iloc[0:0]
        def side_view(d, event_free=False):
            if event_free and "m30_event" in d.columns:
                d = d[d["m30_event"].astype(str).str.lower() == "false"]
            if d.empty:
                return dict(orders=0, games=0, mean_c=None, lo_c=None, se_c=None)
            okey = d["order_id"].astype(str) if "order_id" in d.columns else d["ticker"].astype(str) + d.get("time", "").astype(str)
            o = d.assign(_o=okey, _mw=d["m30"] * d["_w"], _g=d["ticker"].astype(str).str.split("-").str[1]).groupby("_o").agg(
                mw=("_mw", "sum"), w=("_w", "sum"), game=("_g", "first"))
            mean_ = float(o["mw"].sum() / o["w"].sum())
            groups = [x for _, x in o.groupby("game")]
            if len(groups) > 1:
                rng = np.random.default_rng(3)
                bs = []
                for _ in range(2000):
                    pick = pd.concat([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                    bs.append(pick["mw"].sum() / pick["w"].sum())
                lo, se = float(np.percentile(bs, 5)), float(np.std(bs, ddof=1))
            else:
                lo, se = None, None
            return dict(orders=int(len(o)), games=int(o["game"].nunique()), mean_c=round(mean_, 3),
                        lo_c=None if lo is None else round(lo, 3), se_c=None if se is None else round(se, 3))
        priors = {"engine": ENGINE_VERSION, "quote_policy": QUOTE_POLICY, "policy_family_id": fam,
                  "fitted": datetime.now().isoformat(timespec="seconds")}
        for sd in ("yes", "no"):
            d = live[live["side"] == sd] if "side" in live.columns else live.iloc[0:0]
            v, vf = side_view(d), side_view(d, event_free=True)
            n = v["orders"]
            # staged evidence: <10 diagnostics; 10-29 provisional; 30-49 interval shown, no promotion; 50+ (and 10+ games)
            # production judgment: the game-clustered 90% lower bound of the ALL-fill markout must be >= 0
            if n < 10:
                status = "diagnostic"
            elif n < 30:
                status = "provisional"
            elif n < 50 or v["games"] < GATE_MARK_GAMES:
                status = "interval_only"
            else:
                status = "pass" if (v["lo_c"] is not None and v["lo_c"] >= 0) else "fail"
            # never claim near-zero uncertainty (identical marks give a zero bootstrap s.d.): floor 0.25c; 3c if one game
            se = max(0.25, v["se_c"]) if v["se_c"] is not None else (None if v["mean_c"] is None else 3.0)
            priors[sd] = dict(v, status=status, se_c=se, event_free=vf)
            print(f"   {sd.upper():3s} LIVE maker orders: {n} over {v['games']} games | all-fill 30s {v['mean_c']}c (90% LB {v['lo_c']}) | "
                  f"event-free {vf['mean_c']}c ({vf['orders']} orders) -> {status.upper()}")
        g = {"marks": int(len(mk)), "live_orders": priors["yes"]["orders"] + priors["no"]["orders"],
             "games": int(live["ticker"].astype(str).str.split("-").str[1].nunique()) if len(live) else 0,
             "mean_c": None if mean is None else round(mean, 3), "sides": {k: priors[k]["status"] for k in ("yes", "no")},
             "live_mean_c": priors["yes"]["mean_c"],
             # live execution authority: the YES side must pass (NO stays off until it passes on its own)
             "passed": priors["yes"]["status"] == "pass", "policy_family_id": fam,
             "yes_orders": priors["yes"]["orders"], "exec_prior_earned": bool(priors["yes"]["orders"] >= 30)}
        print(f"   MARKOUT GATE (policy family {fam}, YES side, >= {GATE_MARK_ORDERS} live orders over >= {GATE_MARK_GAMES} games, game-clustered 90% "
              f"lower bound >= 0): {'PASS' if g['passed'] else 'not passed'}   [all marks incl. paper: "
              f"{'n/a' if mean is None else format(mean, '+.2f') + 'c'}]")
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exec_priors.json"), "w") as f:
                json.dump(priors, f, indent=2)
            print("   exec_priors.json written (LIVE base-arm orders only; used by the trader only for this quote policy)")
        except Exception as e:
            print(f"   exec_priors.json not written: {e}")
    return est, g


AUTH_RULE = ("PRODUCTION = engine sanity AND >= 1 PRODUCTION authority cell for this engine AND fill gate (this policy family) "
             "AND markout gate (this policy family) AND execution prior earned (>= 30 LIVE base-arm YES orders, this policy family)")


def production_authority_cells(path=None):
    """PRODUCTION cells in calibration.json, counted only if the file was fitted for the current engine."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    try:
        with open(path) as f:
            c = json.load(f)
    except Exception:
        return []
    if (c.get("_meta", {}) or {}).get("engine") != ENGINE_VERSION:
        return []
    a = (c.get("_pool", {}) or {}).get("authority", {}) or {}
    return sorted(k for k, v in a.items() if isinstance(v, dict) and v.get("tier") == "PRODUCTION")


def production_authorization(sanity, fills, mgate, pol, cal_path=None):
    """The ONE production-authorization definition (v2026.38). model_status['passed'] is exactly this conjunction; the
    trader re-checks every term against its own run (engine, family, calibration) and requires this flag as well."""
    fam = pol.get("policy_family_id")
    cells = production_authority_cells(cal_path)
    checks = {
        "engine sanity": bool((sanity or {}).get("passed")),
        f"PRODUCTION authority cell ({len(cells)})": bool(cells),
        "fill gate, this policy family": bool((fills or {}).get("passed") and fam and (fills or {}).get("policy_family_id") == fam),
        "markout gate, this policy family": bool((mgate or {}).get("passed") and fam and (mgate or {}).get("policy_family_id") == fam),
        "execution prior earned, this policy family": bool((mgate or {}).get("exec_prior_earned") and fam
                                                           and (mgate or {}).get("policy_family_id") == fam),
    }
    return dict(passed=all(checks.values()), checks=checks, production_cells=cells, policy_family_id=fam)


def _phase(d):
    cl = d["clock"].astype(str)
    q = cl.str.extract(r"^Q(\d)", expand=False)
    ph = np.where(cl.str.startswith("OT"), "Q4", "Q" + q.fillna("?"))
    return np.where(cl.str.match(r"^Q2 0:00"), "HT", ph)


def phase_report(df, w_blend: float = 0.35):
    """5c) First-half attribution (v2026.34, per the H1 specialists): Q1, Q2, HT, Q3, Q4 graded separately.
    One observation per (market, phase) = its last prediction in that phase. Brier and log loss for model, market and a
    logit blend at w_blend; then the directional edge sign(model - market) x (y - market) by gap bucket with a 90%
    game-bootstrap interval. Rule for ever trading a phase: model or blend beats the market on held-out games AND
    the edge does not fall as the gap grows (if 6-10c gaps lose, the model is overconfident there)."""
    d = df.copy()
    d["phase"] = _phase(d)
    d = d.sort_values("timestamp").groupby(["ticker", "phase"]).tail(1)
    if "game" not in d.columns:
        d["game"] = d["ticker"].astype(str).str.split("-").str[1]
    eps = 1e-4
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    d["p_blend"] = 1.0 / (1.0 + np.exp(-(w_blend * lg(d["p_model_yes"]) + (1 - w_blend) * lg(d["p_mkt_yes"]))))
    ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))))
    bs = lambda p, y: float(np.mean((np.clip(p, 0, 1) - y) ** 2))
    print(f"\n5c) By game phase (one row per market per phase; blend = {w_blend:.0%} model). Lower Brier/log loss is better.")
    rows = []
    for ph in ("Q1", "Q2", "HT", "Q3", "Q4"):
        g = d[d["phase"] == ph]
        if g.empty:
            continue
        y = g["y"].values
        rows.append(dict(phase=ph, markets=len(g), games=g["game"].nunique(),
                         brier_model=bs(g["p_model_yes"].values, y), brier_blend=bs(g["p_blend"].values, y),
                         brier_market=bs(g["p_mkt_yes"].values, y), logloss_model=ll(g["p_model_yes"].values, y),
                         logloss_market=ll(g["p_mkt_yes"].values, y)))
    if rows:
        print(pd.DataFrame(rows).set_index("phase").round(4).to_string())
    d["gap"] = (d["p_model_yes"] - d["p_mkt_yes"]).abs()
    d["edge"] = np.sign(d["p_model_yes"] - d["p_mkt_yes"]) * (d["y"] - d["p_mkt_yes"]) * 100.0
    d["gap_bin"] = pd.cut(d["gap"], [0, .03, .06, .10], labels=["0-3c", "3-6c", "6-10c"], include_lowest=True)
    out = []
    for (ph, gb), g in d.dropna(subset=["gap_bin"]).groupby(["phase", "gap_bin"], observed=True):
        lo, hi = _game_boot(g["edge"], g["game"])
        out.append(dict(phase=ph, gap=gb, markets=len(g), games=g["game"].nunique(), edge_c=g["edge"].mean(), ci90_lo=lo, ci90_hi=hi))
    if out:
        print("   Directional edge by phase and gap (cents; 90% game-bootstrap interval):")
        print(pd.DataFrame(out).set_index(["phase", "gap"]).round(2).to_string())
    diag = [c for c in ("starter", "stint_min", "pf", "stints_done", "sim_pace", "pace_target") if c in d.columns]
    if "starter" in d.columns and d["starter"].notna().any():
        h1 = d[d["phase"].isin(["Q1", "Q2"])]
        if not h1.empty:
            h1 = h1.assign(abs_err=(h1["p_model_yes"] - h1["y"]).abs(), mkt_err=(h1["p_mkt_yes"] - h1["y"]).abs())
            print("   First-half error by starter status (model vs market absolute error):")
            print(h1.groupby("starter").agg(markets=("abs_err", "size"), model_err=("abs_err", "mean"),
                                           market_err=("mkt_err", "mean")).round(4).to_string())


def matchup_ab(df):
    """A/B of the extra matchup layer: p_model (production strength) vs p_shadow (the other end), Brier per market,
    90% interval resampling games. Negative = production better. The evidence for raising/lowering --matchup-strength."""
    if "p_shadow" not in df.columns or df["p_shadow"].notna().sum() == 0:
        print("\n12) MATCHUP A/B: no p_shadow logged yet")
        return
    d = df[df["p_shadow"].notna()]
    per = d.groupby("ticker").apply(lambda x: pd.Series(dict(
        dd=np.average((x.p_model_yes - x.y) ** 2 - (x.p_shadow - x.y) ** 2), game=x.game.iloc[0])))
    lo, hi = _game_boot(per["dd"].reset_index(drop=True), per["game"].reset_index(drop=True))
    verdict = "production better" if hi < 0 else "shadow (other matchup strength) better" if lo > 0 else "inconclusive"
    print(f"\n12) MATCHUP A/B: production minus shadow Brier {per['dd'].mean():+.4f} (90% CI {lo:+.4f} to {hi:+.4f}; "
          f"{per['game'].nunique()} games, {len(per)} markets) -> {verdict}")


def ci_report(df):
    """Game-cluster bootstrap 90% intervals for the model-minus-market Brier difference, per stat (negative = model better).
    One night gives an interval, not a verdict: 'encouraging but inconclusive' is a legitimate answer."""
    print("\n8) Model minus market Brier, per stat, 90% interval resampling whole games (negative = model better)")
    rng = np.random.default_rng(0)
    for stat, g in df.groupby("stat"):
        per = g.groupby("ticker").apply(lambda x: pd.Series(dict(
            d=np.average((x.p_model_yes - x.y) ** 2 - (x.p_mkt_yes - x.y) ** 2), game=x.game.iloc[0])))
        games = [v.values for _, v in per.groupby("game")["d"]]
        if len(games) < 2:
            print(f"   {stat:4s} {per['d'].mean():+.4f}  (one game: no interval)")
            continue
        boots = [np.concatenate([games[j] for j in rng.integers(0, len(games), len(games))]).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [5, 95])
        tag = "model better" if hi < 0 else "market better" if lo > 0 else "inconclusive"
        print(f"   {stat:4s} {per['d'].mean():+.4f}  (90% CI {lo:+.4f} to {hi:+.4f}; {len(games)} games) -> {tag}")


def ladder_report(df):
    """Whole-distribution check: from each set of lines (one player/stat/time) infer the model's and the market's mean
    and spread (normal fit to P(X > line)), and compare spreads per stat. Ratio < 1 = model too narrow (overconfident)."""
    from scipy.stats import norm
    rows = []
    for (pl, st, ts), g in df[df["p_model_raw"].between(0.03, 0.97)].groupby(["player", "stat", "timestamp"]):
        if g["line"].nunique() < 2:
            continue
        A = np.c_[np.ones(len(g)), -g["line"].values]
        cm = np.linalg.lstsq(A, norm.ppf(g["p_model_raw"].values), rcond=None)[0]
        ck = np.linalg.lstsq(A, norm.ppf(np.clip(g["p_mkt_yes"].values, .03, .97)), rcond=None)[0]
        if cm[1] > 0 and ck[1] > 0:
            rows.append(dict(stat=st, sd_ratio=ck[1] / cm[1], mean_diff=cm[0] / cm[1] - ck[0] / ck[1]))
    if rows:
        r = pd.DataFrame(rows)
        print("\n7) Whole distribution vs market (ladders): spread ratio model/market (1 = same) and mean difference")
        print(r.groupby("stat").agg(snapshots=("sd_ratio", "size"), spread_ratio=("sd_ratio", "median"),
                                    mean_model_minus_market=("mean_diff", "median")).round(3).to_string())


def authority_report(df, res):
    """v2026.37 local-authority report: every settled prediction assigned to its phase, gap bin and stat x phase x gap
    cell; per cell: markets, games, directional edge with a 90% game-bootstrap interval, Brier of model / market / the
    pool at the cell's deployed weight. Then a promotion table from STRICT WALK-FORWARD evidence (the same function
    fit_calibration.py uses), compared with the tier in calibration.json. Nothing is promoted automatically here."""
    from kalshi_results import phase_of_rows, gap_bin_of_rows
    try:
        cal = json.load(open(os.path.join(HERE, "calibration.json")))
    except Exception:
        cal = {}
    auth = (cal.get("_pool", {}) or {}).get("authority", {}) if cal.get("_meta", {}).get("engine") == ENGINE_VERSION else {}
    d = df[df["engine"] == ENGINE_VERSION].copy() if "engine" in df.columns else df.copy()
    if d.empty:
        print("\n13) LOCAL AUTHORITY CELLS: no settled predictions from this engine yet")
        return
    d["phase"] = phase_of_rows(d)
    d["gap_bin"] = gap_bin_of_rows(d["p_model_yes"].values, d["p_mkt_yes"].values)
    d = d[d["gap_bin"] != "gt10"]
    d["cell"] = d["stat"].astype(str) + "|" + d["phase"] + "|" + d["gap_bin"]
    d["edge"] = np.sign(d["p_model_yes"] - d["p_mkt_yes"]) * (d["y"] - d["p_mkt_yes"]) * 100.0
    eps = 1e-4
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    rows = []
    for cell, g in d.groupby("cell"):
        w_c = float((auth.get(cell) or {}).get("weight", 0.0))
        pool = 1.0 / (1.0 + np.exp(-((1 - w_c) * lg(g["p_mkt_yes"].values) + w_c * lg(g["p_model_yes"].values))))
        per = g.groupby("ticker").agg(edge=("edge", "mean"), game=("game", "first"))
        lo, hi = _game_boot(per["edge"].reset_index(drop=True), per["game"].reset_index(drop=True))
        rows.append(dict(cell=cell, tier=(auth.get(cell) or {}).get("tier", "-"), markets=len(per), games=per["game"].nunique(),
                         edge_c=per["edge"].mean(), ci90_lo=lo, ci90_hi=hi,
                         brier_model=brier(g["p_model_yes"].values, g["y"].values, g["w"].values),
                         brier_market=brier(g["p_mkt_yes"].values, g["y"].values, g["w"].values),
                         brier_pool=brier(pool, g["y"].values, g["w"].values)))
    t = pd.DataFrame(rows).set_index("cell").round(4)
    print("\n13) LOCAL AUTHORITY CELLS (stat|phase|gap; in-sample description, 90% game-bootstrap interval)")
    print(t.to_string())
    for tier in ("PILOT", "PRODUCTION"):
        sub = t[t["tier"] == tier]
        print(f"   {tier} cells: " + ("none" if sub.empty else ""))
        if not sub.empty:
            print(sub.to_string())
    # promotion / demotion recommendations from strict walk-forward evidence (same code as fit_calibration.py)
    try:
        import fit_calibration as FC
        dd = df[df["engine"] == ENGINE_VERSION].copy()
        dd["p_model_raw"] = dd["p_model_raw"].fillna(dd["p_model_yes"])
        pol, _src = FC.load_policy_population(HERE)
        tr = {k: (float(v["a"]), float(v["b"])) for k, v in cal.items() if not k.startswith("_") and isinstance(v, dict) and "a" in v} \
            if cal.get("_meta", {}).get("engine") == ENGINE_VERSION else {}
        fresh, _ = FC.build_authority(dd, tr, pol)
    except Exception as e:
        print(f"   (walk-forward promotion table unavailable: {e})")
        return
    print(f"\n14) CELL PROMOTION / DEMOTION RECOMMENDATIONS (strict walk-forward; never automatic: run fit_calibration.py to apply)")
    print(f"   {'CELL':<16}{'TIER NOW':<12}{'WF TIER':<12}{'EDGE':>7}{'LB90':>7}{'GAMES':>6}{'MKTS':>6}{'DATES':>6}   RECOMMENDATION")
    order = {"RESEARCH": 0, "PILOT": 1, "PRODUCTION": 2}
    for k, v in fresh.items():
        if v["markets"] == 0:
            continue
        now = (auth.get(k) or {}).get("tier", "RESEARCH")
        wf = v["tier"]
        if v["dates"] < 2:
            rec = "WAIT (one night never promotes)"
        elif order[wf] > order[now]:
            rec = f"PROMOTE to {wf}"
        elif order[wf] < order[now]:
            rec = "DISABLE" if wf == "RESEARCH" and v["research_reason"] == "failed_evidence" else f"DEMOTE to {wf}"
        else:
            rec = f"KEEP {now}"
        if now == "RESEARCH" and wf == "RESEARCH" and v["research_reason"] == "insufficient_evidence":
            continue
        e_ = v["wf_edge_c"]; lb = v["wf_ci90_lo_c"]
        print(f"   {k:<16}{now:<12}{wf:<12}{(e_ if e_ is not None else float('nan')):>+7.2f}{(lb if lb is not None else float('nan')):>+7.2f}"
              f"{v['games']:>6}{v['markets']:>6}{v['dates']:>6}   {rec}")


def tier_report(res):
    """v2026.37: by authority tier (RESEARCH / PILOT / PRODUCTION): candidates, quotes, fills, settled fills, net cents per
    contract, 30 s markout, fill rate."""
    import glob
    def rd(pat):
        fr = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, pat))) if os.path.getsize(f) > 0]
        return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
    el, q, fl, mk = rd("kalshi_eligible_*.csv"), rd("kalshi_quotes_*.csv"), rd("kalshi_fills_*.csv"), rd("kalshi_markouts_*.csv")
    if all(x.empty or "tier" not in x.columns for x in (el, q, fl)):
        print("\n15) BY AUTHORITY TIER: no tier-tagged logs yet (v2026.37+)")
        return
    print("\n15) BY AUTHORITY TIER (RESEARCH / PILOT / PRODUCTION)")
    print(f"   {'TIER':<12}{'CANDS':>7}{'QUOTES':>8}{'FILLS':>7}{'SETTLED':>8}{'NET c/ct':>10}{'M30 c':>8}{'FILL RATE':>10}")
    for t in ("RESEARCH", "PILOT", "PRODUCTION"):
        ne = int((el["tier"] == t).sum()) if "tier" in el.columns else 0
        nq = int((q["tier"] == t).sum()) if "tier" in q.columns else 0
        f_ = fl[(fl["tier"] == t) & (pd.to_numeric(fl["contracts"], errors="coerce") > 0)] if "tier" in fl.columns else fl.iloc[0:0]
        nf = int(f_["order_id"].nunique()) if len(f_) else 0
        f_ = f_.assign(y=f_["ticker"].map(res).map({"yes": 1.0, "no": 0.0})).dropna(subset=["y"]) if len(f_) else f_
        net = None
        if len(f_):
            win = np.where(f_["side"] == "yes", f_["y"], 1.0 - f_["y"])
            fee_pc = 100.0 * f_["fee_usd"].astype(float) / f_["contracts"].astype(float)
            net = float(np.average(win * 100.0 - f_["price_cents"].astype(float) - fee_pc, weights=f_["contracts"].astype(float)))
        m30 = None
        if "tier" in mk.columns and len(mk):
            v = pd.to_numeric(mk.loc[mk["tier"] == t, "m30"], errors="coerce").dropna()
            m30 = float(v.mean()) if len(v) else None
        fr = (nf / nq) if nq else None
        fmt = lambda v, p="+.2f": "n/a" if v is None else format(v, p)
        print(f"   {t:<12}{ne:>7}{nq:>8}{nf:>7}{len(f_):>8}{fmt(net):>10}{fmt(m30):>8}{fmt(fr, '.1%'):>10}")


def funnel_report():
    """v2026.37 opportunity funnel summed over every kalshi_funnel_*.csv cycle row, with % of markets seen."""
    import glob
    fr = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "kalshi_funnel_*.csv"))) if os.path.getsize(f) > 0]
    if not fr:
        print("\n16) OPPORTUNITY FUNNEL: no kalshi_funnel_*.csv yet")
        return
    d = pd.concat(fr, ignore_index=True)
    stages = [c for c in d.columns if c not in ("time", "run_id", "cycle")]
    tot = d[stages].sum()
    base = max(1, int(tot.get("markets_seen", 0)))
    print(f"\n16) OPPORTUNITY FUNNEL ({len(d)} cycles; % of market-cycles seen)")
    for k in stages:
        print(f"   {k:<26}{int(tot[k]):>10}  {100.0 * tot[k] / base:7.2f}%")


def size_authority_report(pol: dict) -> dict:
    """Build qp7 size authority from LIVE base-arm terminal quote episodes for the current policy family."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    qf = [f for f in sorted(glob.glob(os.path.join(here, "kalshi_quotes_*.csv"))) if os.path.getsize(f) > 0]
    ef = [f for f in sorted(glob.glob(os.path.join(here, "kalshi_quote_events_*.csv"))) if os.path.getsize(f) > 0]
    mf = [f for f in sorted(glob.glob(os.path.join(here, "kalshi_markouts_*.csv"))) if os.path.getsize(f) > 0]
    if not qf or not ef or not pol.get("policy_family_id"):
        a = write_authority(pd.DataFrame(), pol.get("policy_family_id"), os.path.join(here, "exec_size_authority.json"))
        print("\n15b) SIZE AUTHORITY: no qualifying qp7 terminal evidence yet")
        return a
    q = pd.concat([pd.read_csv(f) for f in qf], ignore_index=True)
    e = pd.concat([pd.read_csv(f) for f in ef], ignore_index=True)
    q = q[(q.get("quote_policy", "").astype(str) == QUOTE_POLICY) &
          (q.get("policy_family_id", "").astype(str) == str(pol.get("policy_family_id"))) &
          (q.get("arm", "base").fillna("base").astype(str) == "base")].copy()
    if "mode" in q:
        q = q[q["mode"].astype(str).str.upper() == "LIVE"]
    if q.empty:
        a = write_authority(pd.DataFrame(), pol.get("policy_family_id"), os.path.join(here, "exec_size_authority.json"))
        print("\n15b) SIZE AUTHORITY: no LIVE qp7 base-arm quotes for this family")
        return a
    q["requested_qty"] = pd.to_numeric(q["requested_qty"] if "requested_qty" in q else q.get("count", 1), errors="coerce").fillna(1).clip(lower=1)
    q["size_bucket"] = q["requested_qty"].map(size_bucket)
    e["order_id"] = e["order_id"].astype(str)
    term = set(e.loc[e["event"].astype(str) == "terminal", "order_id"])
    fills = e[e["event"].astype(str) == "fill"].copy()
    if len(fills):
        fq = pd.to_numeric(fills["contracts"] if "contracts" in fills else fills.get("filled", 1), errors="coerce").fillna(0)
        fsum = fills.assign(_fq=fq).groupby("order_id")["_fq"].sum()
    else:
        fsum = pd.Series(dtype=float)
    q["order_id"] = q["order_id"].astype(str)
    q["terminal"] = q["order_id"].isin(term)
    q["filled_qty_terminal"] = q["order_id"].map(fsum).fillna(0.0)
    if "game" not in q:
        q["game"] = q["ticker"].astype(str).str.split("-").str[1]
    if mf:
        m = pd.concat([pd.read_csv(f) for f in mf], ignore_index=True)
        if "order_id" in m and "m30" in m:
            mm = m.assign(order_id=m["order_id"].astype(str), _m30=pd.to_numeric(m["m30"], errors="coerce")).groupby("order_id")["_m30"].mean()
            q["m30"] = q["order_id"].map(mm)
    a = write_authority(q, pol.get("policy_family_id"), os.path.join(here, "exec_size_authority.json"))
    print("\n15b) SIZE AUTHORITY (qp7, current policy family)")
    for b, v in a["buckets"].items():
        print(f"   {b:<9} {v.get('status','RESEARCH'):<10} terminal {v.get('terminal_orders',0):>4} | fills {v.get('fills',0):>3} | games {v.get('games',0):>3} | residual LB {v.get('residual_lcb_c')}")
    return a


def main():
    df = load_predictions(sys.argv[1:] or None)
    res = settlements(df["ticker"].unique())
    df["y"] = df["ticker"].map(res).map({"yes": 1.0, "no": 0.0})
    df = df.dropna(subset=["y"])
    if df.empty:
        print("No settled predictions yet (see the Kalshi error above, if any).")
        return
    df["w"] = 1.0 / df.groupby("ticker")["ticker"].transform("size")
    y, w = df["y"].values, df["w"].values
    print(f"\nGrading {len(df):,} predictions on {df['ticker'].nunique()} settled markets "
          f"({', '.join(sorted(df['source_file'].unique()))})\n")

    print("1) Accuracy (Brier, each market weighted equally; lower is better)")
    print(f"   engine raw         {brier(df['p_model_raw'].values, y, w):.4f}")
    print(f"   model as used      {brier(df['p_model_yes'].values, y, w):.4f}   (after calibration, where one was active)")
    print(f"   market mid         {brier(df['p_mkt_yes'].values, y, w):.4f}")

    print("\n2) Edge from trading toward the model, at the market mid (BEFORE spread and fees; needs ~2-4c to be tradable)")
    _, d = edge_report(df, "all predictions")

    d["gap"] = (d["p_model_yes"] - d["p_mkt_yes"]).abs()
    d["gap_bin"] = pd.cut(d["gap"], [0, .03, .06, .10, .15, 1.0], include_lowest=True)
    print("\n3) Edge by size of disagreement (if the model is right, bigger gaps should earn more)")
    print(d.groupby("gap_bin", observed=True).agg(predictions=("edge", "size"), markets=("ticker", "nunique"),
                                                   edge_c=("edge", "mean")).round(2).to_string())
    print("\n4) By stat")
    rows = []
    for stat, g in d.groupby("stat"):
        rows.append(dict(stat=stat, markets=g["ticker"].nunique(), brier_model=brier(g["p_model_yes"].values, g["y"].values, g["w"].values),
                         brier_market=brier(g["p_mkt_yes"].values, g["y"].values, g["w"].values), edge_c=g["edge"].mean()))
    print(pd.DataFrame(rows).set_index("stat").round(4).to_string())
    d["period"] = d["clock"].astype(str).str.extract(r"^(Q\d|OT\d?)", expand=False).fillna("?")
    print("\n5) By period")
    print(d.groupby("period").agg(predictions=("edge", "size"), edge_c=("edge", "mean")).round(2).to_string())

    phase_report(df)

    if "p_book" in df.columns and df["p_book"].notna().any():
        b = df[df["p_book"].notna()]
        print(f"\n5b) Sportsbook consensus on the same markets ({b['ticker'].nunique()} markets): Brier books "
              f"{brier(b['p_book'].values, b['y'].values, b['w'].values):.4f} | market {brier(b['p_mkt_yes'].values, b['y'].values, b['w'].values):.4f}"
              f" | model {brier(b['p_model_yes'].values, b['y'].values, b['w'].values):.4f}")
    ladder_report(df)
    ci_report(df)

    print(f"\n6) LIVE GATE (only predictions from engine {ENGINE_VERSION} count: other engines cannot authorize it)")
    oos = df[(df["cal_id"] != "pre-v19") & (df["engine"] == ENGINE_VERSION)]
    if oos["ticker"].nunique() == 0:
        gate = dict(markets=0, players=0, games=0, edge_c=None, ci_lo=None, ci_hi=None)
        print("   0 markets so far: paper trade with v2026.19 to build the record.")
    else:
        gate, _ = edge_report(oos, "v2026.19+ predictions, out of sample")
    # v2026.37: the broad gate is ENGINE SANITY (enough graded markets over enough games, and the model's edge at the mid is
    # not demonstrably negative), not "the whole model beats Kalshi everywhere". Local alpha is authorized per cell.
    sanity = dict(passed=bool(gate["markets"] >= GATE_MARKETS and gate["games"] >= GATE_GAMES and gate["ci_hi"] is not None
                              and gate["ci_hi"] > 0), rule=f">= {GATE_MARKETS} markets over >= {GATE_GAMES} games, edge 90% CI upper bound > 0")
    print(f"   ENGINE SANITY: {'PASS' if sanity['passed'] else 'not passed'} ({sanity['rule']})")
    authority_report(df, res)
    strat = strategy_report(res)          # research only: hypothetical eligibility does not authorize money (maker selection)
    fills = fill_report(res)
    giveback, mgate = markout_report()
    matchup_ab(df)
    tier_report(res)
    funnel_report()
    pol = current_policy()
    size_auth = size_authority_report(pol)
    nightly = write_nightly_reports(os.path.dirname(os.path.abspath(__file__)), res)
    auth = production_authorization(sanity, fills, mgate, pol)
    passed = auth["passed"]
    # diagnostic only (v2026.38): "the whole model beats Kalshi everywhere" no longer authorizes anything
    dominance = dict(passed=bool(gate["markets"] >= GATE_MARKETS and gate["games"] >= GATE_GAMES and gate["ci_lo"] is not None
                                 and gate["ci_lo"] > 0 and gate["edge_c"] is not None and gate["edge_c"] >= GATE_EDGE_C),
                     rule=f"diagnostic: global edge 90% CI low > 0 and mean >= {GATE_EDGE_C}c (NOT an authorization gate)")
    print(f"   GLOBAL MODEL DOMINANCE (diagnostic, not a gate): {'yes' if dominance['passed'] else 'no'}")
    status = dict(gate, passed=passed, authorization=auth, engine_sanity=sanity, global_model_dominance=dominance, strategy=strat,
                  fills=fills, markout_gate=mgate, size_authority=size_auth, nightly_reports=nightly, spread_giveback=giveback,
                  quote_policy=QUOTE_POLICY, policy=pol, engine=ENGINE_VERSION, generated=datetime.now().isoformat(timespec="seconds"),
                  rule=AUTH_RULE)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print(f"   PRODUCTION AUTHORIZATION: {'PASS' if passed else 'NOT PASSED (paper or --pilot-live only)'}")
    for k, v in auth["checks"].items():
        print(f"      {'ok ' if v else 'no '} {k}")
    print(f"   rule: {AUTH_RULE}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi_pred_grades.csv")
    d.to_csv(out, index=False)
    print(f"\nDetail: {out}")


if __name__ == "__main__":
    main()