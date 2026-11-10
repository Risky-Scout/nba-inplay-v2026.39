#!/usr/bin/env python3
"""v2026.39 checks: one canonical production authorization, policy-family-specific execution evidence (markout gate,
execution prior, sizing excluded from the family), the game-clustered sign-flip null test used for FDR, frozen discovery
-> later confirmation of PRODUCTION cells, behaviour-policy logging with action-support-restricted doubly robust
evaluation, and champion promotion on a clustered lower bound.

Run from the project folder:  python3 tests/test_v2038.py   (works in a temporary copy; your logs are untouched)"""
import os
import sys
import glob
import json
import math
import shutil
import argparse
import tempfile

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="v2038_")
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

FAILS, PASSES = [], []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    (PASSES if cond else FAILS).append(name)


def clean():
    for f in glob.glob(os.path.join(TMP, "*.json")) + glob.glob(os.path.join(TMP, "kalshi_*.csv")):
        os.remove(f)


check("versions: trader v2026.39, quote policy qp7, engine 2026.29",
      T.TRADER_VERSION == "v2026.39" and T.QUOTE_POLICY == "qp7" and FE.QUOTE_POLICY == "qp7" and GP.QUOTE_POLICY == "qp7"
      and mg.ENGINE_VERSION == "2026.29")

# ── P0-1: one canonical production authorization ────────────────────────────────────────────────────────────────────
print("1) canonical production authorization")
clean()
FAM = "famA"


def write_cal(tier="PRODUCTION", engine=mg.ENGINE_VERSION):
    json.dump({"_meta": {"engine": engine}, "_pool": {"authority": {"PTS|Q3": {"tier": tier, "weight": 0.2}}}},
              open(os.path.join(TMP, "calibration.json"), "w"))


ok_s, ok_f = {"passed": True}, {"passed": True, "policy_family_id": FAM}
ok_m = {"passed": True, "policy_family_id": FAM, "exec_prior_earned": True}
pol = {"policy_family_id": FAM}
write_cal()
a = GP.production_authorization(ok_s, ok_f, ok_m, pol)
check("all five terms true -> passed", a["passed"] and a["production_cells"] == ["PTS|Q3"], str(a["checks"]))
check("engine sanity false -> not passed", not GP.production_authorization({"passed": False}, ok_f, ok_m, pol)["passed"])
check("fill gate from another family -> not passed", not GP.production_authorization(ok_s, dict(ok_f, policy_family_id="x"), ok_m, pol)["passed"])
check("markout gate from another family -> not passed", not GP.production_authorization(ok_s, ok_f, dict(ok_m, policy_family_id="x"), pol)["passed"])
check("execution prior not earned -> not passed", not GP.production_authorization(ok_s, ok_f, dict(ok_m, exec_prior_earned=False), pol)["passed"])
write_cal(tier="PILOT")
check("no PRODUCTION cell -> not passed (global dominance can never substitute)", not GP.production_authorization(ok_s, ok_f, ok_m, pol)["passed"])
write_cal(engine="2026.27")
check("a PRODUCTION cell fitted for another engine does not count", GP.production_authority_cells() == [])
check("the obsolete global CI-low / 2c rule is not a term of the authorization",
      not any("dominance" in k.lower() or "ci" in k.lower().split() for k in a["checks"]) and "dominance" not in GP.AUTH_RULE.lower())

# ── P0-2: policy-family-specific execution evidence ──────────────────────────────────────────────────────────────────
print("2) policy-family-specific markout gate and execution prior")
clean()


def marks(fam, mean, n=60, games=12, seed=0):
    r = np.random.default_rng(seed)
    return [dict(order_id=f"{fam}{i}", ticker=f"KXNBAPTS-26NOV0{1 + i % 3}G{i % games:02d}XX-{i}", side="yes", price=50,
                 contracts=1, maker=True, mode="LIVE", quote_policy="qp7", arm="base", policy_family_id=fam,
                 m30=mean + r.normal(0, 0.5), m30_event="false", p_mkt=0.52, m5=0, m15=0, m120=0) for i in range(n)]


pd.DataFrame(marks("famA", +1.5) + marks("famB", -3.0, seed=1)).to_csv(os.path.join(TMP, "kalshi_markouts_2026-11-08.csv"), index=False)
json.dump({"policy_family_id": "famB", "policy_id": "pB"}, open(os.path.join(TMP, "run_config_20261108_190000.json"), "w"))
_, gB = GP.markout_report()
check("gate for family B uses only B's marks (negative -> not passed), A's positive marks cannot help it",
      not gB["passed"] and gB["policy_family_id"] == "famB" and gB["yes_orders"] == 60, str({k: gB.get(k) for k in ("passed", "yes_orders", "live_mean_c")}))
ep = json.load(open(os.path.join(TMP, "exec_priors.json")))
check("exec_priors.json records the policy family it was learned from", ep.get("policy_family_id") == "famB" and ep["yes"]["mean_c"] < 0)
json.dump({"policy_family_id": "famA", "policy_id": "pA"}, open(os.path.join(TMP, "run_config_20261108_200000.json"), "w"))
_, gA = GP.markout_report()
check("gate for family A passes on A's own marks and marks the prior earned", gA["passed"] and gA["exec_prior_earned"] and gA["yes_orders"] == 60)

ns = dict(min_edge_c=0.5, kelly=0.10, session_cap=0.10, player_cap=0.015, pilot_live=False, max_contracts=None, no_exits=False,
          quote_ttl=8, explore=0.05)
p1, f1, _ = T.policy_identity(argparse.Namespace(**ns), "m1", "e1")
p2, f2, _ = T.policy_identity(argparse.Namespace(**dict(ns, kelly=0.10, session_cap=0.01, player_cap=0.0025, pilot_live=True,
                                                         max_contracts=1, no_exits=True)), "m1", "e1")
p3, f3, _ = T.policy_identity(argparse.Namespace(**dict(ns, min_edge_c=1.0)), "m1", "e1")
p4, f4, _ = T.policy_identity(argparse.Namespace(**ns), "m2", "e1")
check("a --pilot-live run and a production run of the same rules share a policy FAMILY (pilot evidence can authorize production)",
      f1 == f2 and p1 != p2, f"{f1} vs {f2}")
check("an economic rule change (min edge) or another champion execution model is a different family", f3 != f1 and f4 != f1)
src = open(os.path.join(TMP, "kalshi_nba_inplay_trader.py")).read()
check("the trader ignores an exec_priors.json from another policy family and requires model_status 'passed'",
      "_ep.get(\"policy_family_id\") != _fam_now" in src and "model_status.json 'passed' is false" in src)

# ── P0-3: game-clustered sign-flip null test ────────────────────────────────────────────────────────────────────────
print("3) sign-flip null test for FDR")
check("exact: 5 games all positive -> p = 1/32", abs(FC.signflip_p([1, 2, 3, 4, 5]) - 1 / 32) < 1e-12)
check("exact: symmetric data -> p well above 0.05", FC.signflip_p([1, -1, 2, -2, 3, -3]) > 0.3)
rng = np.random.default_rng(7)
ps = np.array([FC.signflip_p(rng.normal(0, 1, 12) * rng.integers(5, 30, 12)) for _ in range(800)])
check("under H0 the rejection rate at 5% is ~5% (not the bootstrap's optimistic rate)", 0.02 <= (ps <= 0.05).mean() <= 0.08, f"{(ps <= 0.05).mean():.3f}")
ps_mc = FC.signflip_p(np.r_[np.ones(20), -np.ones(5)])
check("Monte Carlo branch (> 16 games) gives a valid p-value", 0 < ps_mc < 0.05, f"{ps_mc:.4f}")
rows = [dict(date=f"2026-11-0{1 + i % 3}", game=f"g{i}", markets=20, edge_sum=float(e), brier_gain=0.0, w_past=0.0)
        for i, e in enumerate([30, -10, 25, 5, -2, 40, 12, -8])]
se = FC.summarize_evidence(rows)
check("summarize_evidence's FDR p-value IS the sign-flip p (bootstrap kept as p_boot)",
      abs(se["p_value"] - FC.signflip_p([r["edge_sum"] for r in rows])) < 1e-5 and "p_boot" in se)
check("the sign-flip p is more conservative than the observed-centred bootstrap 'p' here", se["p_value"] >= se["p_boot"], f"{se['p_value']} vs {se['p_boot']}")

# ── P0-4: discovery -> freeze -> confirmation ──────────────────────────────────────────────────────────────────────
print("4) discovery / confirmation")


def evrows(dates, per_date, edge, seed=0):
    r = np.random.default_rng(seed)
    return [dict(date=d, game=f"{d}g{j}", markets=20, edge_sum=float(20 * (edge + r.normal(0, 1))), brier_gain=0.0, w_past=0.1)
            for d in dates for j in range(per_date)]


reg = {"engine": mg.ENGINE_VERSION, "cells": {}}
disc = evrows(["2026-11-01", "2026-11-02", "2026-11-03"], 4, 3.0)
e = FC.update_registry(reg, "PTS|Q3", True, disc, "2026-11-03", ["2026-11-01", "2026-11-02", "2026-11-03"])
check("a discovered cell is FROZEN as a CANDIDATE through the last date it was discovered on",
      e["status"] == "CANDIDATE" and e["frozen_through"] == "2026-11-03")
one_more = disc + evrows(["2026-11-04"], 4, 3.0, seed=1)
e = FC.update_registry(reg, "PTS|Q3", True, one_more, "2026-11-04", None)
check("one later date is not enough: still CANDIDATE (needs 6+ games over 2+ dates after the freeze)", e["status"] == "CANDIDATE")
conf = one_more + evrows(["2026-11-05"], 4, 3.0, seed=2)
e = FC.update_registry(reg, "PTS|Q3", True, conf, "2026-11-05", None)
check("a real edge on the later, unseen games -> CONFIRMED", e["status"] == "CONFIRMED" and e["confirm_p"] <= FC.CONFIRM_ALPHA, str(e))
later_bad = conf + evrows(["2026-11-06", "2026-11-07"], 4, -5.0, seed=3)
e2 = FC.update_registry(reg, "PTS|Q3", True, later_bad, "2026-11-07", None)
check("the decision uses a FIXED sample: later games do not reopen the confirmation", e2["status"] == "CONFIRMED" and e2["decided_through"] == "2026-11-05")
reg2 = {"engine": mg.ENGINE_VERSION, "cells": {}}
FC.update_registry(reg2, "REB|Q2", True, disc, "2026-11-03", None)
noise = disc + evrows(["2026-11-04", "2026-11-05"], 4, -0.5, seed=4)
e = FC.update_registry(reg2, "REB|Q2", True, noise, "2026-11-05", ["2026-11-0%d" % i for i in range(1, 6)])
check("a cell whose edge vanishes on the later games -> REJECTED", e["status"] == "REJECTED", str(e))
e = FC.update_registry(reg2, "REB|Q2", True, noise, "2026-11-05", ["2026-11-0%d" % i for i in range(1, 6)])
check("a REJECTED cell cannot be re-discovered from the same data", e["status"] == "REJECTED")
e = FC.update_registry(reg2, "REB|Q2", True, noise, "2026-11-07", ["2026-11-0%d" % i for i in range(1, 8)])
check(f"... only after {FC.REREGISTER_AFTER_DATES} newer dates may it become a fresh candidate (frozen again)",
      e["status"] == "CANDIDATE" and e["frozen_through"] == "2026-11-07" and e["previous"]["status"] == "REJECTED")
with open(os.path.join(TMP, "authority_registry.json"), "w") as _f:
    json.dump({"engine": "2026.27", "cells": {"X": {"status": "CONFIRMED"}}}, _f)
check("another engine's registry is ignored (evidence resets with the engine)", FC.load_registry()["cells"] == {})
os.remove(os.path.join(TMP, "authority_registry.json"))


def synth(n_dates, games_per=4, n_mk=40, seed=1):
    """Calibrated market (truth = market + noise); the model is EXACT in PTS|Q2 and pure noise around the market elsewhere."""
    r = np.random.default_rng(seed)
    out = []
    for d in range(n_dates):
        for g in range(games_per):
            gt = f"26NOV{d + 1:02d}G{g}XX"
            for k_ in range(n_mk):
                for stat in ("PTS", "REB"):
                    for per, clock in ((2, "Q2 5:00"), (3, "Q3 5:00")):
                        mkt = r.uniform(.3, .7)                       # a CALIBRATED market: truth = market + noise
                        lg = math.log(mkt / (1 - mkt))
                        true = 1 / (1 + math.exp(-(lg + r.normal(0, 0.4))))
                        mod = true if (stat, per) == ("PTS", 2) else 1 / (1 + math.exp(-(math.log(mkt / (1 - mkt)) + r.normal(0, 0.15))))
                        if abs(mod - mkt) > 0.099:
                            mod = mkt + np.sign(mod - mkt) * 0.099
                        out.append(dict(ticker=f"KX{stat}-{gt}-{k_}-{per}", game=gt, stat=stat, player=f"p{k_ % 10}", clock=clock,
                                        period=per, p_mkt_yes=mkt, p_model_raw=mod, p_model_yes=mod, y=float(r.random() < true),
                                        yes_bid=int(mkt * 100) - 2, yes_ask=int(mkt * 100) + 2, dead_ball=True,
                                        timestamp=f"2026-11-{d + 1:02d}T20:00", engine=mg.ENGINE_VERSION))
    df = pd.DataFrame(out)
    df["w"] = 1.0 / df.groupby("ticker")["ticker"].transform("size")
    return df


full = synth(8, games_per=6)
early = full[full["timestamp"] < "2026-11-06"]
regA = {"engine": mg.ENGINE_VERSION, "cells": {}}
authA, _ = FC.build_authority(early, {}, FC.DEFAULT_POLICY, registry=regA)
prodA = [k for k, v in authA.items() if v["tier"] == "PRODUCTION"]
candA = [k for k, v in regA["cells"].items() if v["status"] == "CANDIDATE"]
check("first fit: no cell is PRODUCTION on the data that discovered it (it is frozen as a candidate instead)",
      not prodA and "PTS|Q2" in candA, f"candidates {candA}, PTS|Q2 tier {authA.get('PTS|Q2', {}).get('tier')}")
check("a discovered-but-unconfirmed cell can still trade as PILOT on its own evidence",
      authA.get("PTS|Q2", {}).get("tier") == "PILOT" and authA["PTS|Q2"]["confirmation"]["status"] == "CANDIDATE")
authB, _ = FC.build_authority(full, {}, FC.DEFAULT_POLICY, registry=regA)
check("after later unseen dates confirm it, the frozen cell becomes PRODUCTION",
      authB.get("PTS|Q2", {}).get("tier") == "PRODUCTION" and regA["cells"]["PTS|Q2"]["status"] == "CONFIRMED",
      str({k: regA["cells"]["PTS|Q2"].get(k) for k in ("status", "confirm_games", "confirm_edge_c", "confirm_p")}))
check("no REB cell (pure noise around a calibrated market) is confirmed or PRODUCTION",
      not [k for k, v in regA["cells"].items() if k.startswith("REB") and v["status"] == "CONFIRMED"]
      and not [k for k, v in authB.items() if k.startswith("REB") and v["tier"] == "PRODUCTION"],
      str({k: v["status"] for k, v in regA["cells"].items()}))

# ── P0-5: behaviour policy and action support ─────────────────────────────────────────────────────────────────────
print("5) action support in off-policy evaluation")
_pa = {FE.action_key(px, 1, 8): pr for px, pr in {50: 0.05, 51: 0.9, 52: 0.05}.items()}
r_new = pd.DataFrame([dict(price=51, requested_qty=1, ttl=8, propensity=0.9,
                               pi_b=json.dumps({"50": 0.05, "51": 0.9, "52": 0.05}),
                               pi_b_action=json.dumps(_pa))]).itertuples().__next__()
r_old = pd.DataFrame([dict(price=51, requested_qty=1, ttl=8, propensity=0.9)]).itertuples().__next__()
check("logged pi_b gives every level's behaviour probability", FE.behaviour_probs(r_new) == {50: 0.05, 51: 0.9, 52: 0.05})
check("qp7 exact action distribution is price+qty+TTL", FE.behaviour_action_probs(r_new) == _pa)
check("legacy price-only row has NO qp7 action support", FE.behaviour_action_probs(r_old) == {})


def synth_quotes(n=600, seed=4, dates=("20261101", "20261102", "20261103", "20261104"), log_pi=True):
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        yb = int(r.integers(30, 60))
        ya = yb + int(r.integers(3, 6))
        greedy = yb + 1
        p_exp = 0.1
        u = r.random()
        px = greedy - 1 if u < p_exp else (greedy + 1 if u < 2 * p_exp else greedy)
        pib = {str(greedy): 1 - 2 * p_exp, str(greedy - 1): p_exp, str(greedy + 1): p_exp}
        ttl = float(r.choice([5, 8, 12, 45]))
        p_fill = 1 / (1 + math.exp(-(-2.0 + 0.05 * ttl + 0.8 * (px - yb))))
        filled = r.random() < p_fill
        drift = 4.5 + r.normal(0, 1)
        m30 = (52 - px) + drift - 2.0 + r.normal(0, 1)
        out.append(dict(order_id=f"q{i}", ticker=f"KXNBAPTS-26NOV0{1 + i % 4}G{i % 9}XX-{i}", side="yes", price=px, yes_bid=yb,
                        yes_ask=ya, stat="PTS", period=3, phase="Q3", stop="timeout", ttl=ttl, qty_bid=20, qty_ask=20,
                        p_used=0.55, p_mkt=0.52, fee_c=0.0, propensity=float(pib[str(px)]), requested_qty=1,
                        pi_b=json.dumps(pib) if log_pi else None,
                        pi_b_action=(json.dumps({FE.action_key(int(k), 1, ttl): float(v) for k, v in pib.items()}) if log_pi else None),
                        run_id=dates[i % len(dates)] + "_190000",
                        label=float(filled), m30_all=m30 if filled else np.nan, m30_ef=np.nan, q_m30=np.nan if filled else drift,
                        quote_policy="qp7", arm="base"))
    q = pd.DataFrame(out)
    q["date"] = q["run_id"].str[:8]
    q["game"] = q["ticker"].str.split("-").str[1]
    return q


qq = synth_quotes()
m = FE._model_dict(FE.fit_all(qq))
pr = FE.policy_rows(m, qq)
check("with logged pi_b, join (= bid, an exploration arm at 10%) is a supported action on every decision", pr["join_sup"].all())
pv = FE.policy_values(m, qq)
check("policy values report the supported count and overlap share", all(len(v) == 4 for v in pv.values()) and pv["join"][3] == 1.0)
qq0 = synth_quotes(log_pi=False)
pr0 = FE.policy_rows(m, qq0)
check("legacy price-only logs cannot certify qp7 exact actions",
      pr0["join_sup"].sum() == 0, f"join support {pr0['join_sup'].mean():.2f}")
qq_far = qq.copy()
qq_far["pi_b"] = qq_far.apply(lambda r_: json.dumps({str(int(r_.price)): 1.0}), axis=1)
qq_far["pi_b_action"] = qq_far.apply(
    lambda r_: json.dumps({FE.action_key(int(r_.price), int(r_.requested_qty), float(r_.ttl)): 1.0}), axis=1)
pr_far = FE.policy_rows(m, qq_far)
v_far = FE.policy_values(m, qq_far)
check("a level with pi_b < eps contributes nothing: values are computed on supported decisions only",
      v_far["join"][2] == int(pr_far["join_sup"].sum()) and v_far["join"][3] < 0.5)

print("6) champion promotion on a clustered lower bound")
dd = pd.DataFrame(dict(game=np.repeat([f"g{i}" for i in range(12)], 10), date=np.repeat(["d1", "d2"], 60)))
dd["a"], dd["b"] = 1.0 + np.random.default_rng(0).normal(0, 0.5, 120), 0.0
dd["a_sup"], dd["b_sup"] = True, True
cd = FE.clustered_delta(dd, "a", "b")
check("a consistent improvement across games has a positive 90% lower bound", cd["lcb90"] is not None and cd["lcb90"] > 0, str(cd))
dd["a"] = np.random.default_rng(1).normal(0, 3, 120)
cd0 = FE.clustered_delta(dd, "a", "b")
check("noise has a nonpositive lower bound even when its mean is positive by chance", cd0["lcb90"] <= 0, str(cd0))
dd["b_sup"] = np.arange(120) % 3 == 0
cd1 = FE.clustered_delta(dd, "a", "b")
check("the delta is computed only where BOTH policies are supported (overlap reported)", cd1["rows"] == 40 and abs(cd1["overlap"] - 1 / 3) < 1e-9)
later = qq[qq["date"] > "20261102"]
pr_same = FE.challenger_vs_champion(dict(m, validated=True), dict(m, validated=True), later)
check("an identical challenger is never promoted (delta 0: lower bound not > 0)", not pr_same["promote"], str(pr_same["delta"]))
check("promotion requires every check (dates, games, rows, overlap, LB, calibration, markout)",
      set(pr_same["checks"]) >= {f">= {FE.PROMO_MIN_DATES} unseen dates", "delta 90% lower bound > 0",
                                 "fill calibration not worse (log loss)", "adverse-selection prediction not worse (markout MSE)"})

fe_dir = os.path.join(TMP, "fe")
os.makedirs(fe_dir, exist_ok=True)
qq.drop(columns=["label", "m30_all", "m30_ef", "q_m30", "date", "game"]).to_csv(os.path.join(fe_dir, "kalshi_quotes_2026-11-01.csv"), index=False)
evr, mkr, qmr = [], [], []
for r_ in qq.itertuples():
    if r_.label == 1:
        evr.append(dict(order_id=r_.order_id, event="fill", reason="", filled=1))
        mkr.append(dict(order_id=r_.order_id, m30=r_.m30_all, m30_event="false"))
    else:
        evr.append(dict(order_id=r_.order_id, event="terminal", reason="EXPIRED_CONFIRMED: x", filled=0))
        qmr.append(dict(order_id=r_.order_id, q_m30=r_.q_m30))
pd.DataFrame(evr).to_csv(os.path.join(fe_dir, "kalshi_quote_events_2026-11-01.csv"), index=False)
pd.DataFrame(mkr).to_csv(os.path.join(fe_dir, "kalshi_markouts_2026-11-01.csv"), index=False)
pd.DataFrame(qmr).to_csv(os.path.join(fe_dir, "kalshi_quote_marks_2026-11-01.csv"), index=False)
out = FE.main(fe_dir)
summ = out["validation"]["summary"] or {}
check("fit_execution records supported quote offsets (>= 20 logged actions) for the trader",
      out.get("supported_offsets") and set(out["supported_offsets"]) <= {0, 1, 2}, str(out.get("offset_counts")))
check("validation requires a clustered 90% LB vs BOTH baselines and overlap (not a mean comparison)",
      "delta_vs_bid1" in summ and "delta_vs_join" in summ and
      out["validated"] == bool(out["validated"] and summ["delta_vs_bid1"]["lcb90"] > 0 and summ["delta_vs_join"]["lcb90"] > 0))
json.load(open(os.path.join(fe_dir, "challenger_exec_model.json")))
check("the challenger file (with promotion record) is valid JSON", "promotion" in json.load(open(os.path.join(fe_dir, "challenger_exec_model.json"))))
check("trader restricts PRODUCTION quotes to supported offsets and logs pi_b with every quote",
      '"no_action_support"' in src and '"pi_b", "greedy_px", "action_price", "action_prob"' in src)

print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL v2026.39 CHECKS PASSED")