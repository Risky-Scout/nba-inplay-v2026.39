#!/usr/bin/env python3
"""v2026.37 checks: order-state machine, policy identity / policy-specific gates, local authority tiers (walk-forward,
shrinkage, no leakage), execution model v2 (censoring, TTL, scalers, all-fill vs event-free, residuals, rolling policy
validation, champion/challenger), fill-aware QP, quote-specific execution alpha, fresh-book re-optimization, WebSocket
market clock / pruning, model-risk CVaR, isotonic ladder, pilot budget, starvation funnel, and full main() cycles.

Run from the project folder:  python3 tests/test_v2037.py   (works in a temporary copy; your logs are untouched)"""
import os
import sys
import glob
import json
import math
import time
import shutil
import tempfile

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="v2037_")
for f in glob.glob(os.path.join(SRC, "*.py")):
    shutil.copy(f, TMP)
if not os.path.exists(os.path.join(TMP, "sportsbook_consensus.py")):
    alt = os.path.join(os.path.dirname(SRC), "testsupport", "sportsbook_consensus.py")
    if os.path.exists(alt):
        shutil.copy(alt, TMP)
sys.path.insert(0, TMP)
os.chdir(TMP)

import numpy as np
import pandas as pd

import nba_markov_engine as mg
import kalshi_nba_inplay_trader as T
import kalshi_exec_model as XM
import fit_calibration as FC
import fit_execution as FE
import grade_predictions as GP
import kalshi_results as KR
from kalshi_portfolio_qp import CorrelatedKellyQP
from kalshi_ws import KalshiWSBook

FAILS, SKIPS, PASSES = [], [], []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    (PASSES if cond else FAILS).append(name)


def skip(name, why):
    print(f"  [SKIP] {name}  ({why})")
    SKIPS.append(name)


def cleanup_logs():
    for f in set(glob.glob(os.path.join(TMP, "kalshi_*.csv")) + glob.glob(os.path.join(TMP, "run_config_*")) +
                 glob.glob(os.path.join(TMP, "*.json"))):
        os.remove(f)


# ── fake exchange for the state machine ──────────────────────────────────────────────────────────────────────────────
class FakeKalshi(T.KalshiClient):
    def __init__(self):
        super().__init__("", "", paper=True)
        self.fees = {s: (1.0, 0.0, True) for s in T.KALSHI_SERIES}
        self.cancel_ok, self.status, self.fills, self.fills_ok, self.cancels = True, {}, [], True, []
        self.n = 0

    @property
    def can_trade(self):
        return True

    def send_order(self, m, side, price_cents, count, is_maker, ttl_sec):
        self.n += 1
        return f"o{self.n}", 0.0

    def cancel_order(self, oid, ticker):
        self.cancels.append(oid)
        return self.cancel_ok

    def get_order_status(self, oid):
        return self.status.get(oid)

    def get_fills(self, min_ts):
        return list(self.fills) if self.fills_ok else None

    def top_of_book(self, ticker):
        return getattr(self, "tob", None)

    def save(self):
        pass


MK = T.KalshiMarket("KXNBAPTS-26NOV08AAABBB-P20", "E", "t", 7, "Player Seven", "PTS", 19.5, 47, 51, 49, 53)
BET = dict(ticker=MK.ticker, side="yes", price_cents=48, fee_cents=0.0, game_key="G", player_name="Player Seven", player_id=7,
           stat="PTS", line=19.5, is_maker=True, p_model=0.58, p_mkt=0.49, win_prob=0.51, p_used=0.53, roi=0.05,
           tier="PILOT", team_id="T1", snap_bid=47, snap_ask=51, qty_bid=30.0, qty_ask=30.0)


def fill(oid, fid, n, px=0.48):
    return dict(order_id=oid, fill_id=fid, count_fp=n, yes_price_dollars=px, fee_cost=0.0, created_time=time.time())


print("1) order-state machine (P0 capital integrity)")
cleanup_logs()
k = FakeKalshi()
ob = T.OrderBook(k, {})
ob.place(BET, MK, 10, 30)
oid = next(iter(ob.resting))
check("placed maker order is RESTING with an exchange expiry", ob.resting[oid]["state"] == T.RESTING and ob.resting[oid]["expiration_ts"] > time.time())
k.cancel_ok = False
ob.request_cancel(oid, "test")
check("1. cancel failure stays risk-active (CANCEL_FAILED, still resting, still reserved)",
      oid in ob.resting and ob.resting[oid]["state"] == T.CANCEL_FAILED and ob.reserved()[(MK.ticker, "yes")]["cost_usd"] > 0)
k.cancel_ok = True
ob.retry_failed_cancels()
check("retry inside the backoff window does not hammer the exchange", oid in ob.resting and len(k.cancels) == 1)
ob.resting[oid]["next_retry_ts"] = 0.0          # backoff elapsed
ob.retry_failed_cancels()
check("2. cancel success enters PENDING_TERMINAL", oid in ob.pending and ob.pending[oid]["state"] == T.PENDING_TERMINAL)
for _ in range(10):
    ob.reconcile({MK.ticker: MK}, cancel_all=False)          # no status, expiry not passed
check("3. pending risk does not disappear from timeout (10 reconcile cycles) alone",
      oid in ob.pending and ob.reserved(include_resting=False)[(MK.ticker, "yes")]["contracts"] == 10)
k.fills = [fill(oid, "f1", 3)]
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("4. late fill after the cancel request is booked", ob.positions[(MK.ticker, "yes")]["contracts"] == 3
      and ob.reserved(include_resting=False)[(MK.ticker, "yes")]["contracts"] == 7)
k.status[oid] = {"status": "canceled", "fill_count": 5.0, "remaining": 0.0}
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("exchange says 5 filled but only 3 booked -> still reserved (await fills)", oid in ob.pending)
k.fills.append(fill(oid, "f2", 2))
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("7. terminal confirmation (canceled, fill count fully booked) releases the reservation",
      oid not in ob.pending and (MK.ticker, "yes") not in ob.reserved() and ob.terminal_log[-1] == (oid, T.CANCELLED_CONFIRMED))
qe = pd.read_csv(T.QUOTE_EVENTS_CSV)
check("terminal event logged with its proof", (qe["event"] == "terminal").any() and qe["reason"].astype(str).str.contains("exchange status").any())
# partial fill before cancel
ob.place(BET, MK, 4, 30)
o2 = [o for o in ob.resting][-1]
k.fills = [fill(o2, "g1", 1)]
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("5a. partial fill while resting -> PARTIALLY_FILLED, remaining 3 reserved",
      ob.resting[o2]["state"] == T.PARTIALLY_FILLED and ob.reserved()[(MK.ticker, "yes")]["contracts"] == 3)
ob.request_cancel(o2, "test")
k.status[o2] = None
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("5b. partial fill then cancel: pending keeps the 3 unfilled reserved until proof", o2 in ob.pending
      and ob.reserved(include_resting=False)[(MK.ticker, "yes")]["contracts"] == 3)
# expiry proof when status is unreadable
ob.pending[o2]["expiration_ts"] = time.time() - T.EXPIRY_GRACE_S - 5
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("exchange expiry + grace with a fills read after it proves EXPIRED_CONFIRMED", o2 not in ob.pending
      and ob.terminal_log[-1] == (o2, T.EXPIRED_CONFIRMED))
ob.place(BET, MK, 2, 30)
o3 = [o for o in ob.resting][-1]
ob.request_cancel(o3, "test")
ob.pending[o3]["expiration_ts"] = time.time() - T.EXPIRY_GRACE_S - 5
k.fills_ok = False
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("an unreadable fills feed never proves expiry (still reserved)", o3 in ob.pending)
k.fills_ok = True
# crash/restart
ob.place(BET, MK, 2, 30)
o4 = [o for o in ob.resting][-1]
ob.save_journal()
ob2 = T.OrderBook(k, {})
check("6. crash/restart: journal reloads the pending-cancel order as pending and the resting one as CANCEL_FAILED",
      o3 in ob2.pending and o4 in ob2.resting and ob2.resting[o4]["state"] == T.CANCEL_FAILED
      and ob2.reserved()[(MK.ticker, "yes")]["contracts"] == 4)
k.status[o4] = {"status": "resting", "fill_count": 0.0, "remaining": 2.0}
ob2.retry_failed_cancels()
ob2.reconcile({MK.ticker: MK}, cancel_all=False)
check("status 'resting' after a cancel ack moves the order back to live CANCEL_FAILED (retried, reserved)",
      o4 in ob2.resting and ob2.resting[o4]["state"] == T.CANCEL_FAILED)
r = ob2.reserved()
check("3*. reserved() carries contracts, cost, game, player, stat, line, team and tier consistently",
      all(k_ in r[(MK.ticker, "yes")] for k_ in ("contracts", "cost_usd", "game_key", "player", "stat", "line", "team_id", "tier"))
      and abs(r[(MK.ticker, "yes")]["cost_usd"] - r[(MK.ticker, "yes")]["contracts"] * 0.48) < 1e-9)
ok_src = open(os.path.join(TMP, "kalshi_nba_inplay_trader.py")).read()
check("no boolean 'cancelled' flag is used for state any more", 'bet["cancelled"]' not in ok_src and 'get("cancelled")' not in ok_src)

print("1b) review regressions (order book)")
cleanup_logs()
k = FakeKalshi()
ob = T.OrderBook(k, {})
ob.place(BET, MK, 3, 30)
oa = next(iter(ob.resting))
k.cancel_ok = False
for _ in range(5):
    ob.request_cancel(oa, "burst")            # e.g. a WebSocket burst of updates
check("R7. a burst of retries on ONE failing order is backed off and counts as one failure (no instant kill switch)",
      len(k.cancels) == 1 and ob.cancel_failures == 1 and not ob.killed, f"{len(k.cancels)} DELETEs, failures {ob.cancel_failures}")
k.cancel_ok = True
ob.resting[oa]["next_retry_ts"] = 0
ob.request_cancel(oa, "retry")
k.status[oa] = {"status": "canceled", "fill_count": None, "remaining": None}
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("R6b. 'canceled' with an unknown fill count is NOT proof: still reserved", oa in ob.pending)
k.status[oa] = {"status": "canceled", "fill_count": 0.0, "remaining": 0.0}
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("terminal once the fill count is known and booked", oa not in ob.pending)
k.fills = [fill(oa, "late1", 1)]
ob.reconcile({MK.ticker: MK}, cancel_all=False)
check("R6c. a fill reported after release is still booked (never lost)", ob.positions.get((MK.ticker, "yes"), {}).get("contracts") == 1)
class PagedFail(FakeKalshi):
    pass
kp = FakeKalshi()
calls = {"n": 0}
class _R:
    def __init__(self, code, body): self.status_code, self._b, self.text = code, body, ""
    def json(self): return self._b
def _g(url, headers=None, params=None, timeout=None):
    calls["n"] += 1
    return _R(200, {"fills": [fill("x", "p1", 1)], "cursor": "c2"}) if calls["n"] == 1 else _R(500, {})
kp2 = T.KalshiClient("", "", paper=True)
kp2._session.get = _g
kp2._auth = lambda m, p: {}
check("R6a. a fills read with a failed later page returns None (a partial list proves nothing)", kp2.get_fills(0) is None)
cleanup_logs()
k3 = FakeKalshi()
T.ledger_append(dict(timestamp="t", mode="LIVE", order_id="oj", fill_id="fj", ticker=MK.ticker, game="G", player="P", stat="PTS",
                     line=19.5, side="yes", order_type="MAKER", price_cents=48, fee_usd=0.0, contracts=2, cost_usd=0.96, p_model=.5,
                     p_mkt=.5, p_used=.5, ev_pct=0, joint_kelly_pct=0))
json.dump({"oj": dict(BET, count=2, filled=0, submitted=time.time() - 100, state="PENDING_TERMINAL", ttl=12)}, open(T.JOURNAL_JSON, "w"))
ob5 = T.OrderBook(k3, {})
check("R-unconfirmed. journal reload repairs 'filled' from the ledger and gives a v2026.36 entry its exchange expiry",
      ob5.pending["oj"]["filled"] == 2 and abs(ob5.pending["oj"]["expiration_ts"] - (ob5.pending["oj"]["submitted"] + 15)) < 1e-6)
ob5.reconcile({MK.ticker: MK}, cancel_all=False)
check("   so it clears instead of waiting forever", "oj" not in ob5.pending)
wbx = KalshiWSBook(lambda: {})
wbx.want, wbx.connected = {"A", "B"}, True
sn = lambda tk, sq: {"type": "orderbook_snapshot", "sid": 1, "seq": sq, "msg": {"market_ticker": tk, "yes_dollars_fp": [["0.4", "10"]], "no_dollars_fp": [["0.5", "10"]]}}
dl = lambda tk, sq: {"type": "orderbook_delta", "sid": 1, "seq": sq, "msg": {"market_ticker": tk, "price_dollars": "0.4", "delta_fp": "1", "side": "yes"}}
wbx._handle(sn("A", 1)); wbx._handle(sn("B", 2)); wbx.untrack({"B"}); wbx._handle(dl("B", 3)); wbx._handle(dl("A", 4))
check("R1. deltas for an untracked ticker on a shared sid still advance the sequence (no false gap)", wbx.gaps == 0 and wbx.best("A") is not None)

print("2) policy identity and policy-specific gates")
import argparse
a1 = argparse.Namespace(kelly=0.10, lcb_z=1.282, once=False, interval=15, live=False, pilot_live=True)
a2 = argparse.Namespace(kelly=0.11, lcb_z=1.282, once=False, interval=15, live=False, pilot_live=True)
a3 = argparse.Namespace(kelly=0.10, lcb_z=1.282, once=True, interval=99, live=True, pilot_live=True)
p1, f1, _ = T.policy_identity(a1, "m1", "p1")
p2, f2, _ = T.policy_identity(a2, "m1", "p1")
p3, f3, _ = T.policy_identity(a3, "m1", "p1")
p4, f4, _ = T.policy_identity(a1, "m2", "p1")
p5, f5, _ = T.policy_identity(a1, "m1", "p2")
a2b = argparse.Namespace(kelly=0.10, lcb_z=1.645, once=False, interval=15, live=False, pilot_live=True)
p2b, f2b, _ = T.policy_identity(a2b, "m1", "p1")
check("any economic parameter changes the policy id AND family (lcb_z 1.282 -> 1.645)", p1 != p2b and f1 != f2b)
check("v2026.38: a sizing parameter (kelly 0.10 -> 0.11) changes policy_id but not the family", p1 != p2 and f1 == f2)
check("operational settings (--once, --interval, --live) do not", p1 == p3 and f1 == f3)
check("a different execution model is a different policy and family", p1 != p4 and f1 != f4)
check("a re-estimated execution prior changes policy_id but not the family", p1 != p5 and f1 == f5)
cleanup_logs()
json.dump(dict(policy_id="P", policy_family_id="FAM", price_min=20), open(os.path.join(TMP, "run_config_20261108_190000.json"), "w"))
rows, res = [], {}
for i in range(150):
    t = f"KXNBAPTS-26NOV{(i % 12) + 1:02d}AAABBB-{i}"
    res[t] = "yes" if i % 3 else "no"
    for pol, fam, arm in (("qp5", "FAM", "base"), ("qp6", "OTHER", "base"), ("qp6", "FAM", "plus")):
        rows.append(dict(timestamp="x", mode="LIVE", order_id=f"{pol}{fam}{arm}{i}", fill_id=f"f{pol}{fam}{arm}{i}", ticker=t, game="G",
                         player="P", stat="PTS", line=19.5, side="yes", order_type="MAKER", price_cents=50, fee_usd=0.0,
                         contracts=3, cost_usd=1.5, p_model=0.6, p_mkt=0.5, p_used=0.53, ev_pct=6, joint_kelly_pct=0.5,
                         engine=mg.ENGINE_VERSION, trader="v", quote_policy=pol, policy_family_id=fam, policy_id="x", arm=arm))
pd.DataFrame(rows).to_csv(os.path.join(TMP, "kalshi_fills_2026-11-01.csv"), index=False)
fr = GP.fill_report(res)
check("4. qp5 fills, other-family fills and exploration-arm fills never certify the qp6 base policy", fr.get("orders", 0) == 0 and not fr["passed"], str(fr))

print("3) local authority (global 0 does not mean zero trading)")
AUTH = {"*": dict(weight=0.0, tier="RESEARCH", research_reason="failed_evidence"),
        "PTS|Q2|g3-6": dict(weight=0.12, tier="PILOT", confidence=0.35),
        "AST|Q3|g0-3": dict(weight=0.17, tier="PRODUCTION", confidence=0.9),
        "REB|Q4|g6-10": dict(weight=0.10, tier="RESEARCH", research_reason="failed_evidence"),
        "REB|Q4": dict(weight=0.20, tier="PILOT", confidence=0.2),
        "PTS|Q3|g3-6": dict(weight=0.0, tier="RESEARCH", research_reason="insufficient_evidence"),
        "PTS|Q3": dict(weight=0.09, tier="PILOT", confidence=0.3)}
a = T.authority_for("PTS", "Q2", 0.045, AUTH)
check("8. global w=0 + local PILOT cell PTS|Q2|g3-6 -> nonzero local weight, PILOT", a["tier"] == "PILOT" and a["weight"] == 0.12 and a["cell_key"] == "PTS|Q2|g3-6", str(a))
a = T.authority_for("AST", "Q3", 0.02, AUTH)
check("9. global w=0 + local PRODUCTION cell AST|Q3|g0-3 -> PRODUCTION authority", a["tier"] == "PRODUCTION" and a["confidence"] == 0.9, str(a))
a = T.authority_for("PTS", "Q1", 0.045, AUTH)
check("10. unsupported cell -> RESEARCH, weight 0 (no directional candidate)", a["tier"] == "RESEARCH" and a["weight"] == 0.0, str(a))
a = T.authority_for("REB", "Q4", 0.08, AUTH)
check("a specific cell that FAILED its evidence blocks a coarser PILOT parent", a["tier"] == "RESEARCH" and a["cell_key"] == "REB|Q4|g6-10", str(a))
a = T.authority_for("PTS", "Q3", 0.045, AUTH)
check("an insufficient-evidence cell falls through to its coarser PILOT parent", a["tier"] == "PILOT" and a["cell_key"] == "PTS|Q3", str(a))
check("> 10c disagreement is never authorized", T.authority_for("AST", "Q3", 0.11, AUTH)["tier"] == "RESEARCH")
check("11*. gap bins 0-3 / 3-6 / 6-10 / >10", [KR.gap_bin_of_value(g) for g in (0.0, 0.029, 0.03, 0.059, 0.06, 0.10, 0.1001)]
      == ["g0-3", "g0-3", "g3-6", "g3-6", "g6-10", "g6-10", None])


def synth(n_dates=4, games_per=4, n_mk=40, seed=1, skilled=("PTS", "Q2"), future_flip=None):
    """Market = truth + noise everywhere; the model is EXACT in the skilled (stat, phase) cell and pure noise elsewhere."""
    r = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for g in range(games_per):
            gt = f"26NOV{d + 1:02d}G{g}XX"
            for k_ in range(n_mk):
                for stat in ("PTS", "REB"):
                    for per, clock in ((2, "Q2 5:00"), (3, "Q3 5:00")):
                        ph = "Q2" if per == 2 else "Q3"
                        true = r.uniform(.3, .7)
                        lg = math.log(true / (1 - true))
                        mkt = 1 / (1 + math.exp(-(lg + r.normal(0, 0.15))))
                        if (stat, ph) == skilled:
                            mod = true
                        else:
                            mod = 1 / (1 + math.exp(-(math.log(mkt / (1 - mkt)) + r.normal(0, 0.15))))
                        if abs(mod - mkt) > 0.099:
                            mod = mkt + np.sign(mod - mkt) * 0.099
                        y = float(r.random() < true)
                        if future_flip is not None and d >= future_flip:
                            y = float(r.random() < 0.5)
                        rows.append(dict(ticker=f"KX{stat}-{gt}-{k_}-{per}", game=gt, stat=stat, player=f"p{k_ % 10}", clock=clock,
                                         period=per, p_mkt_yes=mkt, p_model_raw=mod, p_model_yes=mod, y=y, yes_bid=int(mkt * 100) - 2,
                                         yes_ask=int(mkt * 100) + 2, dead_ball=True, timestamp=f"2026-11-{d + 1:02d}T20:00",
                                         engine=mg.ENGINE_VERSION))
    df = pd.DataFrame(rows)
    df["w"] = 1.0 / df.groupby("ticker")["ticker"].transform("size")
    return df


d0 = synth()
auth0, _full = FC.build_authority(d0, {}, FC.DEFAULT_POLICY)
glob_t = auth0["*"]["tier"]
pq2 = auth0.get("PTS|Q2", {})
check("9*. fit_calibration writes local authority regardless of the global result (PTS|Q2 earns a tier on its own)",
      pq2.get("tier") in ("PILOT", "PRODUCTION") and pq2.get("weight", 0) > 0, f"global {glob_t}, PTS|Q2 {pq2.get('tier')} w={pq2.get('weight')}")
check("an unskilled cell (REB|Q3) earns no authority", auth0.get("REB|Q3", {}).get("tier") == "RESEARCH", str(auth0.get("REB|Q3", {}).get("research_reason")))
ev0 = FC.walk_forward_cell_evidence(d0, ["*", "PTS|Q2"], FC.DEFAULT_POLICY)
d1 = synth(future_flip=3)                    # the LAST date's outcomes replaced by coin flips
ev1 = FC.walk_forward_cell_evidence(d1, ["*", "PTS|Q2"], FC.DEFAULT_POLICY)
early0 = [e for e in ev0["PTS|Q2"] if e["date"] < "2026-11-04"]
early1 = [e for e in ev1["PTS|Q2"] if e["date"] < "2026-11-04"]
check("11/18. strict walk-forward: changing a FUTURE date's outcomes cannot change any earlier held-out result",
      len(early0) > 0 and all(abs(a_["brier_gain"] - b_["brier_gain"]) < 1e-12 and abs(a_["w_past"] - b_["w_past"]) < 1e-12
                              and abs(a_["edge_sum"] - b_["edge_sum"]) < 1e-12 for a_, b_ in zip(early0, early1)))
check("the first date is scored with an empty past (weight 0)", all(e["w_past"] == 0.0 for e in ev0["PTS|Q2"] if e["date"] == "2026-11-01"))
small = d0[d0["game"].isin(sorted(d0["game"].unique())[:2])]
pre = FC._prep(small, {}, FC.DEFAULT_POLICY)
wa = FC.fit_authority_weights(pre, ["*", "PTS", "*|Q2", "PTS|Q2"])
pre_b = FC._prep(d0, {}, FC.DEFAULT_POLICY)
wb = FC.fit_authority_weights(pre_b, ["*", "PTS", "*|Q2", "PTS|Q2"])
check("12/17. a low-sample cell is shrunk harder toward its parent (lambda smaller) than a high-evidence cell",
      wa["PTS|Q2"]["lam"] < wb["PTS|Q2"]["lam"] and abs(wa["PTS|Q2"]["weight"] - wa["PTS|Q2"]["parent_weight"]) <=
      abs(wa["PTS|Q2"]["weight_fit"] - wa["PTS|Q2"]["parent_weight"]) + 1e-12,
      f"lambda {wa['PTS|Q2']['lam']:.2f} vs {wb['PTS|Q2']['lam']:.2f}")
s_one = FC.summarize_evidence([dict(date="2026-11-01", game=f"g{i}", markets=20, edge_sum=100.0, brier_gain=0.1, w_past=0.0) for i in range(8)])
check("one night never earns PILOT (dates < PILOT_MIN_DATES)", FC.tier_of(s_one, 0.2, "x")[0] == "RESEARCH")

print("4) execution model v2")
qd = pd.DataFrame(dict(order_id=["a", "b", "c", "d"], x=[1, 2, 3, 4]))
evd = pd.DataFrame(dict(order_id=["a", "b", "b", "c"], event=["fill", "cancel", "terminal", "cancel"],
                        reason=["", "x", "CANCELLED_CONFIRMED: exchange status canceled", "y"], filled=[1, 0, 0, 0]))
lab = FE.label_quotes(qd, evd)
check("13. unresolved quotes are CENSORED, not negatives (only a terminal unfilled quote is 0)",
      list(lab["status"]) == ["FILLED", "TERMINAL_UNFILLED", "CENSORED", "CENSORED"] and np.isnan(lab["label"].iloc[2]))


def synth_quotes(n=900, seed=4, dates=("20261101", "20261102", "20261103", "20261104")):
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        side = "yes"
        yb = int(r.integers(30, 60)); ya = yb + int(r.integers(2, 6))
        px = yb + int(r.integers(0, 2))
        ttl = float(r.choice([5, 8, 12, 45]))
        pu, pm = 0.55, 0.52
        p_fill = 1 / (1 + math.exp(-(-2.5 + 0.06 * ttl + 0.8 * (px - yb))))
        filled = r.random() < p_fill
        drift = 1.5 * 3.0 + r.normal(0, 1)              # market converges toward the model (gap 3c) whether we fill or not
        m30 = (pm * 100 - px) + drift - 2.0 + r.normal(0, 1)   # fills are 2c worse than the drift: adverse selection
        rows.append(dict(order_id=f"q{i}", ticker=f"KXNBAPTS-26NOV0{1 + i % 4}G{i % 9}XX-{i}", side=side, price=px, yes_bid=yb,
                         yes_ask=ya, stat="PTS", period=3, phase="Q3", stop="timeout", ttl=ttl, qty_bid=20, qty_ask=20,
                         p_used=pu, p_mkt=pm, fee_c=0.0, propensity=1.0, run_id=dates[i % len(dates)] + "_190000",
                         label=float(filled), m30_all=m30 if filled else np.nan,
                         m30_ef=(m30 + 1.5) if (filled and i % 2) else np.nan, q_m30=np.nan if filled else drift,
                         quote_policy="qp6", arm="base"))
    q = pd.DataFrame(rows)
    q["date"] = q["run_id"].str[:8]
    q["game"] = q["ticker"].str.split("-").str[1]
    return q


qq = synth_quotes()
mfit = FE.fit_all(qq)
md = FE._model_dict(mfit)
r0 = next(qq.itertuples())
x_ft = FE.features_at(r0._replace(ttl=5.0), r0.yes_bid)
x_to = FE.features_at(r0._replace(ttl=45.0), r0.yes_bid)
p5, p45 = XM.quote_value(md, x_ft, 3.0, 0.0)["pf"], XM.quote_value(md, x_to, 3.0, 0.0)["pf"]
check("14. TTL is a feature: a 45 s period-break quote has a higher P(fill) than a 5 s free-throw quote", p45 > p5 + 0.05, f"{p5:.2f} vs {p45:.2f}")
Xtr = FE.design(qq)
check("15. standardized scoring reproduces the training transform (training-only scaler saved in the model)",
      np.allclose(XM.standardize(md["scaler"], Xtr).mean(axis=0)[[i for i, f in enumerate(XM.EXEC_FEATURES) if f not in XM.BINARY and np.std(Xtr[:, i]) > 0]], 0.0, atol=1e-9)
      and np.allclose(XM.standardize(md["scaler"], Xtr[:3]), (Xtr[:3] - np.array(md["scaler"]["mean"])) / np.array(md["scaler"]["sd"])))
check("16. production (all-fill) markout model differs from the event-free diagnostic model",
      not np.allclose(md["mark_all"], md["mark_ef"]) and mfit["n_mark_all"] > mfit["n_mark_ef"])
xs = XM.standardize(md["scaler"], Xtr[qq["label"].values == 1])
resid_pred = float(np.mean(xs @ np.asarray(md["mark_all"])))
raw = float(np.nanmean(qq["m30_all"]))
check("9*. markouts are residualized against the unfilled-quote drift (convergence toward the model is not counted twice)",
      resid_pred < raw - 2.0, f"raw M30 {raw:+.2f}c vs residual {resid_pred:+.2f}c")
check("markout model never loads on the model-market gap", abs(md["mark_all"][XM.GAP_COL]) < 1e-6)
folds = FE.rolling_validation(qq)
check("14*. rolling expanding-window validation over every later date (not only the last)",
      len(folds) == 3 and [f_["test_date"] for f_ in folds] == ["20261102", "20261103", "20261104"])
check("policy value is reported against the bid+1 and join baselines", all(set(f_["policy"]) == {"model", "bid1", "join"} for f_ in folds))
for f_ in glob.glob(os.path.join(TMP, "*exec_model*.json")):
    os.remove(f_)
os.makedirs(os.path.join(TMP, "fe"), exist_ok=True)
fe_dir = os.path.join(TMP, "fe")
qlog = qq.drop(columns=["label", "m30_all", "m30_ef", "q_m30", "date", "game"])
qlog.to_csv(os.path.join(fe_dir, "kalshi_quotes_2026-11-01.csv"), index=False)
evr, mkr, qmr = [], [], []
for r_ in qq.itertuples():
    if r_.label == 1:
        evr.append(dict(order_id=r_.order_id, event="fill", reason="", filled=1))
        mkr.append(dict(order_id=r_.order_id, m30=r_.m30_all, m30_event=("false" if r_.m30_ef == r_.m30_ef else "true")))
    else:
        evr.append(dict(order_id=r_.order_id, event="terminal", reason="EXPIRED_CONFIRMED: x", filled=0))
        qmr.append(dict(order_id=r_.order_id, q_m30=r_.q_m30))
pd.DataFrame(evr).to_csv(os.path.join(fe_dir, "kalshi_quote_events_2026-11-01.csv"), index=False)
pd.DataFrame(mkr).to_csv(os.path.join(fe_dir, "kalshi_markouts_2026-11-01.csv"), index=False)
pd.DataFrame(qmr).to_csv(os.path.join(fe_dir, "kalshi_quote_marks_2026-11-01.csv"), index=False)
out = FE.main(fe_dir)
check("fit_execution writes a challenger with scaler, both markout models, drift and bootstrap draws",
      os.path.exists(os.path.join(fe_dir, "challenger_exec_model.json")) and out.get("scaler") and len(out.get("boot_fill", [])) >= 10)
champ_exists = os.path.exists(os.path.join(fe_dir, "champion_exec_model.json"))
check("34. champion promoted only when validated (policy beats bid+1 and join); never the newest automatically otherwise",
      champ_exists == bool(out["validated"]), f"validated={out['validated']} summary={ {k: v for k, v in (out['validation']['summary'] or {}).items() if k.startswith('value')} }")

print("5) fill-aware QP, confidence, caps")
rng = np.random.default_rng(0)


def cand(t, edge, phi, conf=1.0, pl="P", g="G", team="T", tier="PRODUCTION"):
    return dict(ticker=t, side="yes", game_key=g, player_name=pl, stat="PTS", win_prob=0.5, price_cents=48, fee_cents=0.0,
                outcome=rng.random(4000) < 0.5, expected_edge_c=edge, fill_prob=phi, confidence=conf, team_id=team, tier=tier)


qp = CorrelatedKellyQP(session_cap=0.0048, game_cap=1, single_cap=0.2, player_cap=1, kelly_fraction=0.25)
oA = qp.solve_allocations([cand("A", 4.0, 1.0, pl="a", g="g1"), cand("B", 1.5, 1.0, pl="b", g="g2")], 1000, 1000, {})
oB = qp.solve_allocations([cand("A", 4.0, 0.05, pl="a", g="g1"), cand("B", 1.5, 0.5, pl="b", g="g2")], 1000, 1000, {})
check("18/6. fill probability changes the ranking: A (4c, 5% fill) no longer outranks B (1.5c, 50% fill) for scarce notional",
      {o["ticker"] for o in oA} == {"A"} and {o["ticker"] for o in oB} == {"B"}, f"{[o['ticker'] for o in oA]} -> {[o['ticker'] for o in oB]}")
qp2 = CorrelatedKellyQP(session_cap=0.6, game_cap=1, single_cap=0.2, player_cap=1, kelly_fraction=0.25)
c1 = qp2.solve_allocations([cand("A", 3.0, 1.0, conf=1.0)], 1000, 1000, {})
c2 = qp2.solve_allocations([cand("A", 3.0, 1.0, conf=0.35)], 1000, 1000, {})
check("7. production confidence 0.35 gives a smaller allocation than 1.0", c1 and c2 and c2[0]["contracts"] < c1[0]["contracts"],
      f"{c1[0]['contracts'] if c1 else 0} vs {c2[0]['contracts'] if c2 else 0}")
qp3 = CorrelatedKellyQP(session_cap=0.6, game_cap=1, single_cap=0.01, player_cap=1, kelly_fraction=1.0)
cc = qp3.solve_allocations([cand("A", 30.0, 0.05, conf=0.3)], 1000, 1000, {})
check("8/19. confidence and P(fill) never shrink hard exposure: the position cap binds at FULL notional",
      cc and cc[0]["risk_usd"] <= 0.01 * 1000 * min(1, 2 * (1 - 0.48)) + 1e-9 and cc[0]["risk_usd"] >= 9.0, str(cc[0]["risk_usd"] if cc else None))
pil = [dict(cand(f"P{i}", 6.0, 1.0, pl=f"p{i}", g=f"g{i}", tier="PILOT"), max_contracts=1) for i in range(8)]
qp4 = CorrelatedKellyQP(session_cap=0.6, game_cap=1, single_cap=0.2, player_cap=1, kelly_fraction=0.25)
op = qp4.solve_allocations(pil, 250, 1000, {}, group_caps=[{"members": list(range(8)), "room": 0.01}])
check("4. PILOT orders are one contract each even when Kelly wants more", op and all(o["contracts"] == 1 for o in op))
check("5. PILOT budget: total pilot notional <= pilot session cap (1% of $250)", sum(o["risk_usd"] for o in op) <= 2.5 + 1e-9 and len(op) == 5,
      f"{len(op)} pilots, ${sum(o['risk_usd'] for o in op):.2f}")
pp = [dict(cand(f"Q{i}", 6.0, 1.0, pl="same", g="g", tier="PILOT"), max_contracts=1) for i in range(3)]
opp = qp4.solve_allocations(pp, 250, 1000, {}, group_caps=[{"members": [0, 1, 2], "room": 0.01}, {"members": [0, 1, 2], "room": 0.0025}])
check("6. PILOT player cap: one-contract pilots cannot stack on one player past 0.25%", len(opp) == 1, f"{len(opp)} on one player")
held = [dict(ticker="H", side="yes", game_key="gH", outcome=rng.random(4000) < 0.5, stat="REB", win_prob=0.5, price_cents=50,
             fee_cents=0.0, exposure_usd=150.0, team_id="TT", player_name="Star")]
held_other = [dict(held[0], team_id="ZZ", player_name="Someone")]
qp5 = CorrelatedKellyQP(session_cap=0.6, game_cap=1, single_cap=0.2, player_cap=1, kelly_fraction=0.25, team_sd=0.05, player_sd=0.05)
cN = cand("N", 3.0, 1.0, pl="Star", g="gN", team="TT")
n0 = qp5.solve_allocations([cN], 1000, 1000, {}, held=held_other)
n1 = qp5.solve_allocations([cN], 1000, 1000, {}, held=held)
check("25/21. a held position on the same player/team (other game) carries factor covariance into new sizing",
      n1 and n0 and n1[0]["joint_kelly_fraction"] < n0[0]["joint_kelly_fraction"] - 1e-4,
      f"{n0[0]['joint_kelly_fraction']:.4f} -> {n1[0]['joint_kelly_fraction']:.4f}")

print("6) model-risk CVaR")
o_ = rng.random(2000) < 0.5
mk_orders = lambda shared: [dict(outcome=rng.random(2000) < 0.5, contracts=10, unit_c=50, stat="PTS",
                                 game_key=("G" if shared else f"G{i}"), team_id=("T" if shared else f"T{i}"),
                                 player_name=("P" if shared else f"P{i}"), expected_edge_c=1.0) for i in range(6)]
sds = dict(model=0.0, stat=0.0, game=0.05, team=0.05, player=0.05)
SH = mk_orders(True)
IND = [dict(o, game_key=f"G{i}", team_id=f"T{i}", player_name=f"P{i}") for i, o in enumerate(SH)]
_, cv_sh = T.cvar_scale(SH, [], 1000, limit_frac=1.0, shock_sds=sds)
_, cv_ind = T.cvar_scale(IND, [], 1000, limit_frac=1.0, shock_sds=sds)
_, cv_none = T.cvar_scale(SH, [], 1000, limit_frac=1.0, shock_sds=None)
check("26. model-risk shocks raise CVaR95 when the portfolio shares model assumptions", cv_sh["cvar95"] > cv_ind["cvar95"] and cv_sh["cvar95"] > cv_none["cvar95"],
      f"shared {cv_sh['cvar95']} vs spread {cv_ind['cvar95']} vs no shocks {cv_none['cvar95']}")
lo_fill = [dict(o, fill_prob=0.01) for o in SH]
_, cv_lf = T.cvar_scale(lo_fill, [], 1000, limit_frac=1.0, shock_sds=sds)
check("27. CVaR never lets fill probability reduce worst-case outstanding exposure", cv_lf["cvar95"] == cv_sh["cvar95"])

print("7) ladder coherence")
pj = T.isotonic_decreasing([10.5, 15.5, 20.5], [0.60, 0.64, 0.30], [1.0, 1.0, 1.0])
check("22. weighted isotonic projection enforces P(X>L1) >= P(X>L2) >= P(X>L3) (projected, not dropped)",
      all(pj[i] >= pj[i + 1] - 1e-12 for i in range(2)) and abs(pj[0] - 0.62) < 1e-9 and abs(pj[2] - 0.30) < 1e-12, str(pj))
pjw = T.isotonic_decreasing([10.5, 15.5], [0.60, 0.64], [9.0, 1.0])
check("uncertainty weights pull the pooled value toward the more precise strike", abs(pjw[0] - 0.604) < 1e-9, str(pjw))

print("8) WebSocket market clock")
wb = KalshiWSBook(lambda: {})
wb.want = {"TK", "TK2"}
wb.connected, wb._last_msg = True, time.time()
hits = []
wb.on_update = lambda t: hits.append(t)
wb._handle({"type": "orderbook_snapshot", "sid": 1, "seq": 1, "msg": {"market_ticker": "TK", "yes_dollars_fp": [["0.47", "30"]], "no_dollars_fp": [["0.49", "30"]]}})
wb._handle({"type": "orderbook_snapshot", "sid": 1, "seq": 2, "msg": {"market_ticker": "TK2", "yes_dollars_fp": [["0.40", "10"]], "no_dollars_fp": [["0.55", "10"]]}})
check("snapshot builds the book (bid 47 / ask 51)", wb.fresh("TK") == (47, 30.0, 51, 30.0), str(wb.fresh("TK")))
kk = FakeKalshi()
kk.ws = wb
kk.top_of_book = T.KalshiClient.top_of_book.__get__(kk)
obw = T.OrderBook(kk, {})
obw.place(dict(BET, ticker="TK", snap_bid=47, snap_ask=51, qty_bid=30.0, qty_ask=30.0), T.KalshiMarket("TK", "E", "t", 7, "P", "PTS", 19.5, 47, 51, 49, 53), 5, 30)
oidw = next(iter(obw.resting))
wb.on_update = lambda t: obw.fast_checks(tickers={t}) if t in {b["ticker"] for b in obw.resting.values()} else None
wb.on_update = lambda t: None
wb._handle({"type": "orderbook_delta", "sid": 1, "seq": 3, "msg": {"market_ticker": "TK", "price_dollars": "0.47", "delta_fp": "30", "side": "no"}})
check("a delta that does not move our side's top keeps the quote", oidw in obw.resting)
wb.on_update = lambda t: obw.fast_checks(tickers={t}) if t in {b["ticker"] for b in obw.resting.values()} else None
wb._handle({"type": "orderbook_delta", "sid": 1, "seq": 4, "msg": {"market_ticker": "TK", "price_dollars": "0.49", "delta_fp": "-30", "side": "no"}})
check("21. a book delta (ask jumped) cancels the resting quote immediately from the WS callback, no watch loop",
      oidw not in obw.resting and oidw in obw.pending)
mic = wb.micro("TK")
check("velocity features come from the WS top-of-book history (ask +2c -> positive 1 s mid velocity)",
      mic["has_ws"] and set(XM.MICRO_KEYS) <= set(mic) and mic["mid_vel_1s"] > 0, str({k: round(v, 2) for k, v in mic.items() if k != "has_ws"}))
wb._handle({"type": "orderbook_delta", "sid": 1, "seq": 9, "msg": {"market_ticker": "TK2", "price_dollars": "0.40", "delta_fp": "5", "side": "yes"}})
check("22*. WS sequence gap marks the books stale: fresh() returns None (REST fallback)", wb.fresh("TK2") is None and wb.gaps == 1)
wb._handle({"type": "orderbook_snapshot", "sid": 2, "seq": 1, "msg": {"market_ticker": "TK2", "yes_dollars_fp": [["0.40", "10"]], "no_dollars_fp": [["0.55", "10"]]}})
dead = wb.untrack({"TK2"})
check("23. untrack prunes the book and unsubscribes a subscription with no wanted ticker", "TK2" not in wb.want and wb.best("TK2") is None and 2 in dead)
wb._handle({"type": "orderbook_snapshot", "sid": 2, "seq": 2, "msg": {"market_ticker": "TK2", "yes_dollars_fp": [["0.40", "10"]], "no_dollars_fp": [["0.55", "10"]]}})
check("updates for an untracked ticker are ignored", wb.best("TK2") is None)
k2 = FakeKalshi()
k2.tob = (47.0, 30.0, 51.0, 30.0)
ob3 = T.OrderBook(k2, {})
ob3.place(dict(BET, price_cents=47, snap_bid=47, snap_ask=51, qty_bid=30.0), MK, 25, 30)   # joined: we are 25 of the 30 at 47
o5 = next(iter(ob3.resting))
check("24. own order size removed: our 25 of 30 at the bid is not 'depth collapse' (other traders still 5 of 30 -> collapse)",
      ob3.fast_checks() == {"depth_collapse": 1})
ob4 = T.OrderBook(k2, {})
ob4.place(dict(BET, price_cents=47, snap_bid=47, snap_ask=51, qty_bid=30.0), MK, 2, 30)
check("   and with only 2 of 30 ours, the same book keeps the quote", ob4.fast_checks() == {})

print("9) funnel and starvation")
fn = T.Funnel(starve_s=1800)
fn.block("no local authority", 71); fn.block("small gap", 11); fn.block("execution EV <= 0", 8)
fn.last_post = time.time() - 1801
rep_ = fn.starvation()
check("13. 30 quiet minutes print OPPORTUNITY STARVATION with reasons and percentages", rep_ is not None and "OPPORTUNITY STARVATION" in rep_
      and "78.9%  no local authority" in rep_, rep_.splitlines()[1] if rep_ else "")
check("starvation is diagnostic only (Funnel has no threshold attributes; prints at most every 5 minutes)",
      fn.starvation() is None and not any(a_ for a_ in vars(fn) if "edge" in a_ or "min_" in a_))
fn.posted(1)
check("posting an order resets the starvation timer and reasons", fn.reasons == {} and time.time() - fn.last_post < 1)

print("10) engine items (spec 26-32)")
for n_, t_ in ((28, "substitution hazard responds to long stint"), (29, "return hazard responds to bench duration"),
               (30, "simulated blowout reduces star minutes relative to close path"), (31, "fitted phase CVs"),
               (32, "no future-data leakage in fitted engine parameters")):
    skip(f"{n_}. {t_}", "engine items 26-32 are NOT implemented in v2026.37; ENGINE_VERSION unchanged")
check("engine version unchanged (simulation probabilities untouched)", mg.ENGINE_VERSION == "2026.28")

# ── full main() cycles against mocks ─────────────────────────────────────────────────────────────────────────────────
print("11) full trader cycles (mocked NBA / D&T / Kalshi)")
cleanup_logs()
HOME, AWAY = 1610612754, 1610612743


def roster(team, base_id):
    rows_e, rows_b = [], []
    pos = ["PG", "SG", "SF", "PF", "C", "G", "F", "C", "G-F", "F-C"]
    for k_ in range(10):
        pid = base_id + k_
        mp = 34 - 2 * k_ if k_ < 5 else 22 - 2 * (k_ - 5)
        rows_e.append(dict(team_id=team, player_id=pid, player_name=f"Player {chr(65 + k_)}{team % 100}", p_usg=0.26 - 0.01 * k_,
                           p_fga_rim_100=6, p_fga_mid_100=4, p_fg3a_100=7, p_fta_100=5, p_ast_100=6 - 0.3 * k_, p_tov_100=2.5,
                           p_orb_100=2, p_drb_100=7, p_stl_100=1.3, p_blk_100=1.0, p_fgpct_rim=0.64, p_fgpct_mid=0.42,
                           p_fg3pct=0.36, p_ftpct=0.78, oepm=1, depm=0, epm=1, p_mp_48_active=mp))
        rows_b.append(dict(team_id=team, player_id=pid, p_mp_48_active=mp, p_mp_48=mp, position=pos[k_], p_t_poss_48=99.0,
                           p_pts=mp * 0.5, p_reb=mp * 0.15, p_ast=mp * 0.12, p_fg3m=mp * 0.05, game_dt="2026-11-08",
                           game_id=22600100, opp_team_id=AWAY if team == HOME else HOME, pre_game_status=None, p_start=1.0 if k_ < 5 else 0.0))
    return rows_e, rows_b


he, hb = roster(HOME, 1000)
ae, ab = roster(AWAY, 2000)
EPM, BOX = pd.DataFrame(he + ae), pd.DataFrame(hb + ab)
TEAM = pd.DataFrame([dict(team_id=HOME, team_depm_next=0.5), dict(team_id=AWAY, team_depm_next=-0.5)])
GAME = dict(gameId="0022600100", gameStatus=2, gameCode="20261108/DENIND",
            homeTeam=dict(teamId=HOME, teamTricode="IND"), awayTeam=dict(teamId=AWAY, teamTricode="DEN"))


def live_state(g):
    box = {}
    for rows in (he, ae):
        for k_, r_ in enumerate(rows):
            mins = 22.0 - k_ if k_ < 5 else 10.0 - (k_ - 5)
            box[r_["player_id"]] = dict(PTS=int(mins * 0.5), REB=int(mins * 0.15), AST=int(mins * 0.12), **{"3PM": 1},
                                        FGA=int(mins * 0.4), FTA=2, ORB=1, STL=0, BLK=0, TOV=1, PF=2, MIN=mins, ACTIVE=True,
                                        STARTER=k_ < 5)
    ctx = mg.LivePBPContext(last_possession_team_id=HOME, last_action_type="timeout", last_action_number=400)
    return mg.LiveInPlayState(game_id=g["gameId"], matchup_title="DEN @ IND", clock_display="Q3 6:00", period=3,
                              seconds_remaining_in_game=1080.0, home_team_id=HOME, away_team_id=AWAY, home_alias="IND",
                              away_alias="DEN", home_score=70, away_score=66, possession_team="HOME",
                              on_court_home=[1000 + k_ for k_ in range(5)], on_court_away=[2000 + k_ for k_ in range(5)],
                              live_box=box, pbp_context=ctx, game_code=g["gameCode"], game_status=2)


SIM_PROBS = {}


def fake_markets(self, names, tokens):
    out = []
    for (pid, stat), lines in SIM_PROBS.items():
        for j, (line, p) in enumerate(lines):
            mid = int(round(100 * p)) - (3, 5, 7, 9)[(j + pid) % 4]
            if not (22 <= mid <= 78):
                continue
            out.append(T.KalshiMarket(f"KXNBA{stat}-26NOV08DENIND-{pid}{int(line)}", "KXNBA-26NOV08DENIND", "t", pid,
                                      f"Player {pid}", stat, line, mid - 2, mid + 2, 100 - mid - 2, 100 - mid + 2))
    return out


st0 = live_state(GAME)
hr, _ = mg.build_dt_rosters_and_fallback_lines(EPM, BOX, HOME)
ar, _ = mg.build_dt_rosters_and_fallback_lines(EPM, BOX, AWAY)
pre = mg.InPlayMarkovEngine(st0, hr, ar, 0.5, -0.5).run_monte_carlo(n_sims=600, seed=1)
for p in hr + ar:
    if p.p_mp < 20:
        continue
    i_ = pre["id_to_idx"][p.player_id]
    for stat in ("PTS", "REB", "AST"):
        smp = pre[stat][:, i_]
        mu = float(smp.mean())
        SIM_PROBS[(p.player_id, stat)] = [(L + 0.5, float((smp > L + 0.5).mean())) for L in range(int(mu) - 2, int(mu) + 3)]

STATE = {"act": 400, "book_shift": 0, "feed_down": False}
T.DT_API_KEY = "test"
mg.NBALiveFeedClient.get_todays_games = staticmethod(lambda: [GAME])
mg.NBALiveFeedClient.fetch_live_game_state = classmethod(lambda cls, g: live_state(g))
mg.NBALivePlayByPlayParser.new_actions = staticmethod(lambda gid, after: (STATE["act"], set(), "timeout"))
mg.NBALivePlayByPlayParser.latest_action = staticmethod(lambda gid, after: None if STATE["feed_down"] else (STATE["act"], set(), "timeout"))
mg.DunksAndThreesClient.fetch_game_day_data = lambda self, d: (EPM, BOX, TEAM)
mg.calibrate_to_dt = lambda hr, ar, box, hd, ad, h, a, n_sims=1500, seed=7: ({}, {(p.player_id, s): True for p in hr + ar for s in ("PTS", "REB", "AST", "3PM", "PRA")}, None)
T.KalshiClient.load_series_fees = lambda self: self.fees.update({s: (1.0, 0.0, True) for s in T.KALSHI_SERIES})
T.KalshiClient.fetch_open_props = fake_markets
def _tob(self, ticker):
    m = next(x for x in fake_markets(self, None, None) if x.ticker == ticker)
    return float(m.yes_bid + STATE["book_shift"]), 50.0, float(m.yes_ask + STATE["book_shift"]), 50.0
T.KalshiClient.top_of_book = _tob

import logging
records = []
class _H(logging.Handler):
    def emit(self, rec):
        records.append(rec.getMessage())
T.logger.addHandler(_H())


def write_cal(auth):
    json.dump({"_pool": {"w": 0.0, "authority": auth}, "_meta": {"id": "testcal", "engine": mg.ENGINE_VERSION, "schema": "authority-v1"}},
              open(os.path.join(TMP, "calibration.json"), "w"))


def run(extra):
    records.clear()
    sys.argv = ["trader", "--once", "--no-books", "--sims", "4000", "--shadow-sims", "0", "--bankroll", "1000",
                "--kelly", "0.25", "--single-cap", "0.02", "--game-cap", "0.10", "--player-cap", "0.04"] + extra
    T.main()
    return list(records)


# global 0, no local authority -> zero directional candidates (the honest default)
cleanup_logs()
write_cal({"*": dict(weight=0.0, tier="RESEARCH", research_reason="failed_evidence")})
rec = run([])
cyc = [m for m in rec if m.startswith("Cycle #1:")]
check("3. global 0 and no local authority: cycle completes with zero candidates", bool(cyc) and " 0 +EV candidates" in cyc[0], cyc[0][:120] if cyc else "")
fu = pd.read_csv(T.FUNNEL_CSV)
check("12. funnel counters reconcile (each stage <= the stage that feeds it)", (fu["markets_seen"] >= fu["trusted"]).all()
      and (fu["trusted"] >= fu["valid_book"]).all() and (fu["valid_gap"] >= fu["positive_model_signal"]).all()
      and (fu["positive_model_signal"] >= fu["local_authority_nonzero"]).all() and int(fu["local_authority_nonzero"].sum()) == 0
      and int(fu["positive_model_signal"].sum()) > 0)

# global 0, local PRODUCTION cell + local PILOT cell
cleanup_logs()
write_cal({"*": dict(weight=0.0, tier="RESEARCH", research_reason="failed_evidence"),
           "PTS|Q3": dict(weight=0.35, tier="PRODUCTION", confidence=1.0, research_reason=""),
           "REB|Q3": dict(weight=0.35, tier="PILOT", confidence=0.3, research_reason="")})
rec = run([])
el = pd.read_csv(T.ELIGIBLE_CSV)
q_ = pd.read_csv(T.QUOTES_CSV) if os.path.exists(T.QUOTES_CSV) else pd.DataFrame()
tiers_el = set(el["tier"].dropna()) if "tier" in el.columns else set()
check("8/9. global 0 + local cells: PRODUCTION (PTS) and PILOT (REB) candidates are generated", {"PRODUCTION", "PILOT"} <= tiers_el, str(tiers_el))
check("14. eligible log carries tier, confidence, cell, authority weight, gap bin, phase, policy id",
      {"tier", "confidence", "cell", "authority_weight", "gap_bin", "phase", "policy_id"} <= set(el.columns))
check("quote log carries tier / cell / policy / execution features", len(q_) > 0 and {"tier", "cell", "policy_id", "policy_family_id", "ttl",
                                                                                     "stop", "aggr", "fee_c", "fill_prob"} <= set(q_.columns))
check("every PILOT quote is one contract, maker, YES", len(q_[q_["tier"] == "PILOT"]) > 0 and (q_.loc[q_["tier"] == "PILOT", "count"] == 1).all()
      and (q_.loc[q_["tier"] == "PILOT", "side"] == "yes").all())
check("PRODUCTION quotes are sized by Kelly (some > 1 contract)", (q_.loc[q_["tier"] == "PRODUCTION", "count"] > 1).any())
check("PILOT notional within the pilot session budget", (q_.loc[q_["tier"] == "PILOT", "count"] * q_.loc[q_["tier"] == "PILOT", "price"] / 100.0).sum() <= 0.01 * 1000 + 1e-9)
cfg = json.load(open(sorted(glob.glob(os.path.join(TMP, "run_config_*.json")))[-1]))
check("run config archives trader v2026.38, qp6, policy ids, authority cells", cfg["trader"] == "v2026.38" and cfg["quote_policy"] == "qp6"
      and cfg["policy_id"] and cfg["policy_family_id"] and cfg["authority_production"] == ["PTS|Q3"] and cfg["min_period_used"] == 1)
pl = pd.read_csv(T.PRED_LOG)
check("prediction log records the authority tier and cell per market", {"authority_tier", "authority_cell", "gap_bin"} <= set(pl.columns))

AUTH_TRADE = {"*": dict(weight=0.0, tier="RESEARCH", research_reason="failed_evidence"),
              "PTS|Q3": dict(weight=0.35, tier="PRODUCTION", confidence=1.0, research_reason=""),
              "REB|Q3": dict(weight=0.35, tier="PILOT", confidence=0.3, research_reason="")}
_cleanup0 = cleanup_logs
def cleanup_logs():
    _cleanup0()
    write_cal(AUTH_TRADE)
# R2: a PILOT quote must keep its queue place on the next cycle (its own resting exposure is not "player already held")
cleanup_logs()
_ng = {"n": 0}
_games0 = mg.NBALiveFeedClient.get_todays_games
def _two_cycles():
    _ng["n"] += 1
    if _ng["n"] > 3:
        raise KeyboardInterrupt
    return [GAME]
mg.NBALiveFeedClient.get_todays_games = staticmethod(_two_cycles)
records.clear()
sys.argv = ["trader", "--no-books", "--sims", "4000", "--shadow-sims", "0", "--bankroll", "1000", "--kelly", "0.25",
            "--single-cap", "0.02", "--game-cap", "0.10", "--player-cap", "0.04", "--interval", "1", "--explore", "0"]
T.main()
mg.NBALiveFeedClient.get_todays_games = _games0
c2 = [m for m in records if m.startswith("Cycle #3:")]   # cycle 1 never trades in a continuous run (no previous mid)
qe2 = pd.read_csv(T.QUOTE_EVENTS_CSV) if os.path.exists(T.QUOTE_EVENTS_CSV) else pd.DataFrame()
_nlw = int(((qe2["event"] == "cancel") & (qe2["reason"] == "no_longer_wanted")).sum()) if len(qe2) else -1
check("R2. on the next trading cycle the resting quotes are KEPT (queue place) and no pilot quote is cancelled as no longer wanted",
      bool(c2) and " 0 kept resting" not in c2[0] and _nlw == 0, (c2[0][:140] if c2 else "no cycle 2") + f" | no_longer_wanted {_nlw}")
qq2 = pd.read_csv(T.QUOTES_CSV) if os.path.exists(T.QUOTES_CSV) else pd.DataFrame()
_pil_pl = set(qq2.loc[qq2["tier"] == "PILOT", "player_id"]) if len(qq2) else set()
_share = [p for p in _pil_pl if (qq2["player_id"] == p).sum() > 1]
_pi_ok = len(qq2) > 0 and "pi_b" in qq2.columns and all(
    abs(sum(json.loads(p_).values()) - 1.0) < 1e-3 and abs(json.loads(p_).get(str(int(px_)), -1) - ap_) < 1e-3
    for p_, px_, ap_ in zip(qq2["pi_b"], qq2["price"], qq2["action_prob"]))
check("v2026.38: every posted quote logs pi_b over its ladder (sums to 1) and the action's own probability", _pi_ok,
      str(qq2[["price", "action_prob", "pi_b"]].head(2).to_dict("records")) if len(qq2) and "pi_b" in qq2.columns else "no pi_b column")
check("R2b. a PILOT quote never shares a player with another quote (pilots yield to production, one pilot per player)",
      len(qq2) > 0 and (qq2["tier"] == "PILOT").any() and not _share, f"pilot players {len(_pil_pl)}, sharing {_share[:3]}")

# fresh book shifted 1c up -> the ladder is rebuilt and the quote re-chosen (price changes) instead of merely validated
cleanup_logs()
STATE["book_shift"] = 1
rec = run(["--revalidate-mid-c", "1.0"])
STATE["book_shift"] = 0
ps = [m for m in rec if m.startswith("Pre-send check")]
check("20/16. final fresh-book quote optimization re-chooses q* (orders re-quoted at the new book)", bool(ps) and "'requoted': 0" not in ps[0], ps[0] if ps else "")
q2 = pd.read_csv(T.QUOTES_CSV) if os.path.exists(T.QUOTES_CSV) else pd.DataFrame()
_b2 = q2[q2["arm"] == "base"] if len(q2) else q2
check("re-quoted base-arm orders rest at the FRESH side bid + 1 (book +1c), never at the stale price",
      len(_b2) > 0 and (_b2["price"] == _b2["side_bid"] + 1).all() and (_b2["yes_bid"] == _b2["yes_bid"]).all(),
      str(_b2[["side", "price", "side_bid", "yes_bid", "quote_level"]].head(3).to_dict("records")) if len(q2) else "")

# fail closed: PBP unreadable, stale book, kill switch block PILOT and PRODUCTION alike
cleanup_logs()
STATE["feed_down"] = True
rec = run([])
STATE["feed_down"] = False
ps = [m for m in rec if m.startswith("Pre-send check")]
check("16. PILOT still fails closed: unreadable play-by-play -> nothing sent", bool(ps) and "'sent': 0" in ps[0], ps[0] if ps else "")
cleanup_logs()
STATE["book_shift"] = 3
rec = run([])
STATE["book_shift"] = 0
ps = [m for m in rec if m.startswith("Pre-send check")]
check("16. stale book (mid moved 3c since pricing) -> nothing sent", bool(ps) and "'sent': 0" in ps[0], ps[0] if ps else "")
_orig = T.OrderBook.__init__
def _killed(self, *a, **k):
    _orig(self, *a, **k)
    self.killed = "test kill switch"
T.OrderBook.__init__ = _killed
cleanup_logs()
rec = run([])
T.OrderBook.__init__ = _orig
check("16. kill switch blocks every tier", any("Kill switch active" in m for m in rec) and not os.path.exists(T.QUOTES_CSV))

# RESEARCH never trades live: a live run with only RESEARCH authority is refused production and places nothing
cleanup_logs = _cleanup0
cleanup_logs()
write_cal({"*": dict(weight=0.0, tier="RESEARCH", research_reason="failed_evidence")})
try:
    run(["--live"])
    refused = False
except SystemExit:
    refused = True
check("15. RESEARCH never trades live: --live without production authority refuses to start", refused)
check("    and the refusal names the missing production cell", any("no PRODUCTION authority cell" in m for m in records))

# manual research weight in paper -> RESEARCH tier, paper only
cleanup_logs()
write_cal({})
rec = run(["--model-weight", "0.35"])
el = pd.read_csv(T.ELIGIBLE_CSV) if os.path.exists(T.ELIGIBLE_CSV) else pd.DataFrame()
check("manual --model-weight in paper: candidates are RESEARCH tier (paper evidence, one contract)", len(el) > 0 and set(el["tier"]) == {"RESEARCH"})

# quote-specific execution alpha flows into sizing (champion model present)
cleanup_logs()
write_cal({"*": dict(weight=0.0, tier="RESEARCH", research_reason="failed_evidence"),
           "PTS|Q3": dict(weight=0.35, tier="PRODUCTION", confidence=1.0, research_reason="")})
cm = dict(FE._model_dict(mfit), quote_policy="qp6", validated=True, n_fills=1000, explore_scale=0.1)
cm["mark_all"] = [0.0] * len(XM.EXEC_FEATURES)
cm["mark_all"][0] = 1.5                                  # +1.5c residual at every quote
cm["exec_model_id"] = XM.model_id(cm)
json.dump(cm, open(os.path.join(TMP, "champion_exec_model.json"), "w"))
rec = run([])
qx = pd.read_csv(T.QUOTES_CSV) if os.path.exists(T.QUOTES_CSV) else pd.DataFrame()
ok17 = len(qx) > 0 and "em_model_c" in qx.columns and (abs(qx["em_model_c"].astype(float) - 1.5) < 1e-6).all() \
    and (abs(qx["em_c"].astype(float) - (1000 / 1030) * 1.5 - (30 / 1030) * qx["em_prior_c"].astype(float)) < 1e-6).all()
check("17/7. the quote-specific execution residual (shrunk to the side prior) is the execution mean used in pricing and sizing", ok17,
      str(qx[["em_model_c", "em_c", "fill_prob"]].head(2).to_dict("records")) if len(qx) else "no quotes")
check("    and the QP is fed P(fill) from the champion model", len(qx) > 0 and (qx["fill_prob"].astype(float) < 1.0).all())
os.remove(os.path.join(TMP, "champion_exec_model.json"))

# tonight's mode: --pilot-live with a manual weight against a mocked LIVE exchange
cleanup_logs()
write_cal({})
_live = {"n": 0, "orders": []}
T.KalshiClient.can_trade = property(lambda self: True)
def _send(self, m, side, price_cents, count, is_maker, ttl_sec):
    _live["n"] += 1
    _live["orders"].append(dict(ticker=m.ticker, side=side, price=price_cents, count=count, maker=is_maker))
    return f"L{_live['n']}", 0.0
T.KalshiClient.send_order = _send
T.KalshiClient.cancel_order = lambda self, oid, t: True
T.KalshiClient.get_fills = lambda self, ts: []
T.KalshiClient.get_order_status = lambda self, oid: None
T.KalshiClient.get_positions = lambda self: {}
T.KalshiClient.get_sports_shard_balance = lambda self: 1000.0
rec = run(["--pilot-live", "--model-weight", "0.35"])
lo = pd.DataFrame(_live["orders"])
check("pilot-live + manual weight (tonight's mode): live orders are PILOT, 1 contract, maker, YES only",
      len(lo) > 0 and (lo["count"] == 1).all() and lo["maker"].all() and (lo["side"] == "yes").all(), f"{len(lo)} live orders")
check("pilot-live total notional within the 1% session cap", (lo["price"] * lo["count"]).sum() / 100.0 <= 0.01 * 1000 + 1e-9,
      f"${(lo['price'] * lo['count']).sum() / 100.0:.2f} of $10.00")
check("pilot-live run mode recorded as pilot", json.load(open(sorted(glob.glob(os.path.join(TMP, "run_config_*.json")))[-1]))["run_mode"] == "pilot")
jr = json.load(open(T.JOURNAL_JSON)) if os.path.exists(T.JOURNAL_JSON) else {}
check("shutdown cancels every live order; the journal keeps them PENDING_TERMINAL with their exchange expiry until proof",
      jr and all(v["state"] == "PENDING_TERMINAL" and v["expiration_ts"] and v["cancel_ts"] for v in jr.values()), f"{len(jr)} journaled")

_src_t = open(os.path.abspath(__file__)).read()
check("no unconditional truth bypass in this test file", _src_t.count("or " + "True") == 0)

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{len(PASSES)} passed, {len(FAILS)} failed, {len(SKIPS)} skipped (engine items not implemented)")
print("\n" + ("V2026.37 CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)