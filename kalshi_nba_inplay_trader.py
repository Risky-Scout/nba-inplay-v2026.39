#!/usr/bin/env python3
"""
Kalshi NBA In-Play Trader (Calibrated Markov + Correlated Kelly + Maker-First)
======================================================================================
Pricing
  - Markov engine calibrated to the D&T projected box; untrusted player/stat pairs are not traded.
  - Model probability is shrunk toward the market mid (--model-weight) before EV is computed.
  - Gates: two-sided book, max spread, price band, mid stable vs last cycle (no trading into news the
    NBA CDN has not shown yet), max model-vs-market disagreement (data/model error, not edge),
    game date in the Kalshi event ticker must match tonight's NBA game.
  - EV uses all-in cost (price + fee); fees from each series' fee_type (maker fees where they apply).
Sizing
  - Correlated half-Kelly QP; correlations measured from the simulated joint outcomes.
  - Caps include positions already held: session (total open risk), per game, per position
    (position cap tapers on favorites above 50c).
Execution / fees (--fee-policy min, default: minimise fees, never block a +EV trade on fees alone)
  - Default route: post-only maker order (bid +1c inside the spread), short expiry, cancelled every
    cycle; real fills reconciled from GET /portfolio/fills. Post-only can never pay a taker fee;
    maker fee is 0 on most prop series and 0.0175*P*(1-P) (max 0.44c) where Kalshi charges it.
  - Taker IOC only when it pays for itself: ROI after the taker fee >= --taker-min-ev (strong edge),
    or in the last 300s of Q4/OT (maker fills unlikely) with ROI after fee >= --min-ev.
  - Every fee is inside the EV calculation, so fee-charging routes are only used when still +EV.
  - --fee-policy zero: maker-only on fee-free series only (strict; may produce no trades).
v2026.29 (README 5l)
  - Edge is what we expect to KEEP after a fill: nominal edge minus half the spread capture (fills are not random),
    Monte Carlo s.e., main-vs-shadow model disagreement and a buffer; the optimizer sizes on that conservative p.
  - Pre-send re-check: no orders for a game with a new play since its simulation; each order re-read against the
    live book. Quotes live 30 s. Event-free 5/15/30/120 s markouts; counterfactual logging of dropped candidates.
Evidence
  - Last-mid drift report at shutdown (NOT execution CLV: basketball happens in between; markouts are the
    execution measure, graded by grade_predictions.py).
  - Ledger records real fills only (paper mode simulates maker fills only if the market trades through).
  - Positions are rebuilt from today's ledger on restart.
"""

import os
import re
import sys
import time
import math
import uuid
import logging
import argparse
import hashlib
import zlib
import json
import subprocess
import threading
import queue
import inspect
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import nba_markov_engine as mg
from sportsbook_consensus import SportsbookConsensus, prob_at_line
from kalshi_portfolio_qp import CorrelatedKellyQP, kalshi_fee_cents
import kalshi_exec_model as XM
from kalshi_exec_model import EXEC_FEATURES, exec_features, quote_ev_c
from kalshi_results import phase_of_period, gap_bin_of_value
from execution_size_authority import size_bucket, highest_authorized_qty, load_authority
from execution_hazard import action_key

try:
    from kalshi_auth import load_private_key, get_auth_headers
except ImportError:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    import base64

    def load_private_key(file_path: str):
        with open(file_path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

    def sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
        clean_path = path.split("?")[0]
        msg = f"{timestamp_ms}{method}{clean_path}".encode("utf-8")
        sig = private_key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return base64.b64encode(sig).decode("utf-8")

    def get_auth_headers(key_id: str, private_key, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sign_request(private_key, ts, method, path),
            "Content-Type": "application/json"
        }

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("KalshiTrader")


def get_keychain_secret(key_name: str) -> str:
    try:
        val = subprocess.check_output(
            ["security", "find-generic-password", "-a", os.environ.get("USER", ""), "-s", key_name, "-w"],
            stderr=subprocess.DEVNULL, text=True).strip()
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key_name, "")


DT_API_KEY = get_keychain_secret("DT_API_KEY")
KALSHI_KEY_ID = get_keychain_secret("KALSHI_API_KEY_ID")
ODDS_API_KEY = get_keychain_secret("ODDS_API_KEY")   # The Odds API (optional): sportsbook consensus
KALSHI_PEM_PATH = os.path.expanduser(os.getenv("KALSHI_PEM_PATH", "~/kalshi/kalshi-mm/Odds_Actuary.pem"))
if not os.path.exists(KALSHI_PEM_PATH):
    KALSHI_PEM_PATH = os.path.join(CURRENT_DIR, "Odds_Actuary.pem")

QUOTE_MAX_AGE = 75   # seconds a resting quote may live before it is refreshed (overridden by --quote-ttl at startup)
PENDING_CHECKS = 4   # cycles a cancelled order stays watched for late fills (~1 min)
TRADER_VERSION = "v2026.39"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")           # joins every log row to its run_config_<RUN_ID>.json
POLICY = {"policy_id": "unset", "policy_family_id": "unset", "exec_model_id": "none", "exec_prior_id": "none"}   # set in main
QUOTE_POLICY = "qp7"   # v2026.37: base quote bid+1 (join when toxic / NO), or argmax LCB[P(fill) x (alpha + execution
                       # residual - fee)] over the maker ladder once a CHAMPION execution model is promoted; the quote is
                       # re-optimized on the final fresh book; exploration +/-1c arms (own predictions, logged propensity)
                       # decay with execution-model uncertainty; fill-aware QP; WS market-clock fast cancels
OLD_POLICY_PRIOR = {"yes": (0.0, 0.71), "no": (-2.3, 1.92)}   # qp3 (Nov 6) measured markouts: used only WEAKLY (below)
TIERS = ("PRODUCTION", "PILOT", "RESEARCH")
EXPIRY_GRACE_S = 30.0       # an order past its EXCHANGE expiry + this, with a fills read after that time, is provably done
OPERATIONAL_ARGS = {"once", "ws_selftest", "write_production_profile", "interval", "books_interval", "override_gate",
                    "live", "watch"}   # everything else in args is economically relevant and enters the policy hash
# v2026.38: SIZE and RISK-BUDGET settings enter policy_id but not policy_family_id. They change how many contracts are bought,
# not which quotes are posted or at what price, and --pilot-live forces them. Were they in the family, pilot evidence could
# never authorize a production run (different family) and a production run could never earn its own evidence (no gate).
SIZING_ARGS = {"pilot_live", "kelly", "session_cap", "player_cap", "game_cap", "single_cap", "max_bet", "bankroll",
               "max_contracts", "pilot_session_cap", "pilot_player_cap", "cvar_limit", "no_exits"}
OLD_PRIOR_WEIGHT = 0.15     # hierarchical prior for a new quote policy: 15% of the old policy's mean ...
NEW_POLICY_SD = {"yes": 2.0, "no": 3.0}   # ... with deliberately wide uncertainty until qp5 earns its own marks
EXEC_MODEL_JSON = None      # set below (needs CURRENT_DIR)
TTL_BY_STOP = {"timeout": 12, "substitution": 8, "freethrow": 5, "period": 45, "instantreplay": 15}
TTL_GENERIC = 8             # any other dead ball
EDGE_RULE = "lcb90>=0.5c+inv_skew"
MARK_H = (5, 15, 30, 120)   # markout horizons, seconds after the fill
PRED_COLS = ['timestamp', 'ticker', 'clock', 'player', 'stat', 'line', 'p_model_yes', 'p_mkt_yes', 'p_model_raw', 'cal_id',
             'model_mean', 'model_sd', 'proj_min', 'current', 'engine', 'p_book', 'n_books', 'p_micro', 'p_shadow',
             'yes_bid', 'yes_ask', 'period', 'dead_ball',   # v2026.29: book + regime, so weights can be fitted where we trade
             'phase', 'starter', 'stint_min', 'pf', 'stints_done', 'sim_pace', 'pace_target',   # v2026.34: H1 attribution
             'authority_tier', 'authority_cell', 'authority_weight', 'gap_bin']                  # v2026.37: local authority
DEAD_BALL = {"timeout", "substitution", "period", "freethrow", "instantreplay", "stoppage", "jumpball", "foul", "violation"}
SERIES_STAT = {"KXNBAPTS": "PTS", "KXNBAREB": "REB", "KXNBAAST": "AST", "KXNBA3PT": "3PM", "KXNBAPRA": "PRA"}
KALSHI_SERIES = list(SERIES_STAT)
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LEDGER_CSV = os.path.join(CURRENT_DIR, f"kalshi_fills_{TODAY_STR}.csv")
PRED_LOG = os.path.join(CURRENT_DIR, f"kalshi_predictions_{TODAY_STR}.csv")
MARKOUT_CSV = os.path.join(CURRENT_DIR, f"kalshi_markouts_{TODAY_STR}.csv")
ELIGIBLE_CSV = os.path.join(CURRENT_DIR, f"kalshi_eligible_{TODAY_STR}.csv")     # every trade-policy-eligible candidate
QUOTES_CSV = os.path.join(CURRENT_DIR, f"kalshi_quotes_{TODAY_STR}.csv")         # every order posted (fill-model data)
QUOTE_EVENTS_CSV = os.path.join(CURRENT_DIR, f"kalshi_quote_events_{TODAY_STR}.csv")  # cancel (with reason) / fill per order
QUOTE_MARKS_CSV = os.path.join(CURRENT_DIR, f"kalshi_quote_marks_{TODAY_STR}.csv")    # v2026.37: 30 s mid move after every quote
JOURNAL_JSON = os.path.join(CURRENT_DIR, "kalshi_orders_journal.json")          # unresolved orders survive a crash
CAL_JSON = os.path.join(CURRENT_DIR, "calibration.json")
EXEC_PRIORS_JSON = os.path.join(CURRENT_DIR, "exec_priors.json")
EXEC_MODEL_JSON = os.path.join(CURRENT_DIR, "champion_exec_model.json")      # v2026.37: only a PROMOTED model trades
CHALLENGER_JSON = os.path.join(CURRENT_DIR, "challenger_exec_model.json")    # scored in shadow on every quote
FUNNEL_CSV = os.path.join(CURRENT_DIR, f"kalshi_funnel_{TODAY_STR}.csv")
PROD_PROFILE_JSON = os.path.join(CURRENT_DIR, "production_profile.json")   # written by grade_predictions.py from LIVE maker orders
STATUS_JSON = os.path.join(CURRENT_DIR, "model_status.json")
SIZE_AUTH_JSON = os.path.join(CURRENT_DIR, "exec_size_authority.json")


def production_profile(args, cal_id: str, min_period: int, side_on: dict) -> dict:
    """The fields a production run must keep identical to its saved, validated profile (v2026.37: EVERY economically
    relevant argument, via the policy-family hash and its field list, plus engine and resolved side/period settings)."""
    _, fam, fields = policy_identity(args, POLICY.get("exec_model_id", "none"), POLICY.get("exec_prior_id", "none"))
    return dict({k: (v if isinstance(v, (int, float, str, bool, type(None))) else json.dumps(v, default=str)) for k, v in fields.items()},
                engine=mg.ENGINE_VERSION, policy_family_id=fam, min_period_used=min_period, no_enabled=bool(side_on["no"]))


def load_exec_model(path: Optional[str] = None) -> Optional[dict]:
    """The CHAMPION execution model (fit_execution.py promotes a challenger only when it beats the champion's rolling
    walk-forward policy value). None if absent, for another quote policy, other features, or not validated."""
    return XM.load_model(path or EXEC_MODEL_JSON, QUOTE_POLICY)


def file_hash(path: str) -> str:
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    except Exception:
        return "none"


def quote_policy_code_hash() -> str:
    """Hash of the source that decides WHERE and HOW LONG we quote: a code change there is a new policy."""
    src = ""
    for obj in (ttl_for, conservative_edge, inventory_skew_c, isotonic_decreasing):
        try:
            src += inspect.getsource(obj)
        except Exception:
            pass
    try:
        src += inspect.getsource(XM)
    except Exception:
        pass
    src += repr(sorted(TTL_BY_STOP.items())) + str(TTL_GENERIC) + QUOTE_POLICY
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def policy_identity(args, exec_model_id: str, exec_prior_id: str) -> Tuple[str, str, dict]:
    """(policy_id, policy_family_id, hashed fields). EVERY argument that can change prediction, eligibility, execution,
    sizing, risk or exits is hashed; only OPERATIONAL_ARGS are excluded.
      policy_id        = everything incl. the execution prior id (exact strategy of this run)
      policy_family_id = the same minus exec_prior_id and minus SIZING_ARGS (v2026.38). exec_priors.json is re-estimated every night by
                         grade_predictions.py; if evidence could only pool under the exact policy_id, no execution gate
                         could ever reach 100 orders over 10 games. The family pools nights of the same RULES; the exact
                         policy_id is still logged on every row and reported."""
    fields = {k: v for k, v in sorted(vars(args).items()) if k not in OPERATIONAL_ARGS}
    fields.update(trader=TRADER_VERSION, quote_policy=QUOTE_POLICY, quote_code=quote_policy_code_hash(),
                  ttl_policy=sorted(TTL_BY_STOP.items()) + [("generic", TTL_GENERIC)], exec_model_id=exec_model_id)
    fam_fields = {k: v for k, v in fields.items() if k not in SIZING_ARGS}
    size_fields = {k: v for k, v in fields.items() if k in SIZING_ARGS}
    fam = hashlib.sha256(json.dumps(fam_fields, sort_keys=True, default=str).encode()).hexdigest()[:12]
    pid = hashlib.sha256((fam + "|" + json.dumps(size_fields, sort_keys=True, default=str) + "|" + exec_prior_id).encode()).hexdigest()[:12]
    return pid, fam, fields


def authority_for(stat: str, phase: str, gap: Optional[float], authority: dict) -> dict:
    """Local model authority for this regime (v2026.37). Priority: stat|phase|gap, stat|phase, *|phase|gap, *|phase,
    stat, global. A cell that is RESEARCH only for lack of data ('insufficient_evidence') falls through to the next
    coarser cell; a cell with enough data that FAILED blocks (evidence against trading there). A PILOT/PRODUCTION cell
    applies even when the global weight is 0. Nothing found -> RESEARCH, weight 0."""
    gb = gap_bin_of_value(gap) if gap is not None else None
    keys = ([f"{stat}|{phase}|{gb}"] if gb else []) + [f"{stat}|{phase}"] + ([f"*|{phase}|{gb}"] if gb else []) + \
        [f"*|{phase}", stat, "*"]
    if gap is not None and gb is None:
        return dict(weight=0.0, tier="RESEARCH", confidence=0.0, cell_key="gt10", source="gap above 10c: never authorized", w_sd=None)
    for k in keys:
        a = authority.get(k)
        if not a:
            continue
        t = a.get("tier", "RESEARCH")
        if t in ("PRODUCTION", "PILOT") and float(a.get("weight", 0.0)) > 0:
            return dict(weight=min(0.35, max(0.0, float(a["weight"]))), tier=t,
                        confidence=float(np.clip(a.get("confidence", 0.0), 0.0, 1.0)), cell_key=k, source="calibration",
                        w_sd=a.get("w_sd"))
        if a.get("research_reason", "insufficient_evidence") != "insufficient_evidence":
            return dict(weight=0.0, tier="RESEARCH", confidence=0.0, cell_key=k, source="failed walk-forward evidence", w_sd=None)
    return dict(weight=0.0, tier="RESEARCH", confidence=0.0, cell_key="none", source="no local authority", w_sd=None)


def isotonic_decreasing(lines: List[float], probs: List[float], weights: List[float]) -> List[float]:
    """Weighted pool-adjacent-violators: the closest (weighted least squares) sequence with
    P(X > L1) >= P(X > L2) >= ... for L1 < L2 < ...  Returns probabilities in the order of `lines` (sorted input)."""
    blocks = []                       # [value, weight, count]
    for p, w in zip(probs, weights):
        blocks.append([float(p), max(1e-9, float(w)), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0] - 1e-15:     # increasing pair violates "decreasing"
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    out = []
    for v, _, n in blocks:
        out += [v] * n
    return out


class Funnel:
    """Opportunity funnel (v2026.37): per-cycle stage counts, rolling totals, and the starvation diagnostic. Diagnostic
    only: nothing here ever changes a threshold."""
    STAGES = ("markets_seen", "in_scope", "trusted", "valid_book", "valid_gap", "positive_model_signal",
              "local_authority_nonzero", "pilot_authority", "production_authority", "side_enabled", "execution_EV_positive",
              "LCB_pass", "QP_positive", "below_one_contract", "integer_lifted", "CVaR_survivors", "fresh_book_survivors",
              "PBP_survivors", "quotes_posted", "fills")

    def __init__(self, starve_s: float = 1800.0):
        self.total = {k: 0 for k in self.STAGES}
        self.cycle = {k: 0 for k in self.STAGES}
        self.reasons: Dict[str, int] = {}
        self.last_post = time.time()
        self.starve_s = starve_s
        self.last_report = 0.0

    def start_cycle(self):
        self.cycle = {k: 0 for k in self.STAGES}

    def add(self, stage: str, n: int = 1):
        self.cycle[stage] = self.cycle.get(stage, 0) + n
        self.total[stage] = self.total.get(stage, 0) + n

    def block(self, reason: str, n: int = 1):
        self.reasons[reason] = self.reasons.get(reason, 0) + n

    def posted(self, n: int):
        if n > 0:
            self.last_post = time.time()
            self.reasons = {}

    def starvation(self, now: Optional[float] = None) -> Optional[str]:
        """The report text when no quote has been posted for starve_s seconds (at most every 5 minutes), else None."""
        now = time.time() if now is None else now
        if now - self.last_post < self.starve_s or now - self.last_report < 300.0:
            return None
        self.last_report = now
        tot = sum(self.reasons.values())
        lines = [f"OPPORTUNITY STARVATION: no quote posted for {(now - self.last_post) / 60:.0f} min (diagnostic only; "
                 f"no threshold is changed). Blocked opportunities by reason:"]
        for k, v in sorted(self.reasons.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"   {100.0 * v / max(1, tot):5.1f}%  {k}  ({v})")
        if not tot:
            lines.append("   (no candidate reached any gate: no live games, markets or in-scope players)")
        return "\n".join(lines)

    def row(self, run_id: str, cycle_no: int) -> dict:
        return dict(time=datetime.now().isoformat(timespec="seconds"), run_id=run_id, cycle=cycle_no, **self.cycle)


def ttl_for(last_action_type: str, cap: int) -> int:
    """State-dependent quote life: a quote posted in a timeout can rest longer than one between two free throws."""
    t = (last_action_type or "").lower()
    for k, v in TTL_BY_STOP.items():
        if k in t:
            return int(min(cap, v))
    return int(min(cap, TTL_GENERIC))


def load_calibration():
    """Per-stat fit from fit_calibration.py: p = sigmoid(a + b*logit(p_raw)). Missing file/stat = no change."""
    try:
        with open(CAL_JSON) as f:
            c = json.load(f)
        if c.get("_meta", {}).get("engine") != mg.ENGINE_VERSION:
            logger.warning(f"calibration.json was fitted on engine {c.get('_meta', {}).get('engine', 'pre-2026.22')}, not "
                           f"{mg.ENGINE_VERSION}: ignored (a fit on a different engine would double-correct). Using identity.")
            return {}, "none"
        return {k: (float(v["a"]), float(v["b"])) for k, v in c.items() if not k.startswith("_")}, c["_meta"]["id"]
    except FileNotFoundError:
        return {}, "none"
    except Exception as e:
        logger.error(f"calibration.json unreadable ({e}); running uncalibrated.")
        return {}, "none"


def calibrate(cal: dict, stat: str, p: float) -> float:
    if stat not in cal:
        return p
    a, b = cal[stat]
    p = min(max(p, 1e-4), 1 - 1e-4)
    return 1.0 / (1.0 + math.exp(-(a + b * math.log(p / (1 - p)))))


def dt_provenance_ok(dt_state: dict, home_id: int, away_id: int) -> Tuple[bool, str]:
    """Fail closed if the D&T rows do not belong to this game: both teams present in /epm and /game-predictions-box,
    and, where D&T returns them, the same date, the same game id for both teams, opponents that match, and fresh updates."""
    box, epm, date = dt_state.get("box"), dt_state.get("epm"), str(dt_state.get("date", ""))
    if box is None or epm is None or box.empty or epm.empty:
        return False, "no D&T data"
    hb, ab = box[box["team_id"] == home_id], box[box["team_id"] == away_id]
    if hb.empty or ab.empty or epm[epm["team_id"] == home_id].empty or epm[epm["team_id"] == away_id].empty:
        return False, "team missing from D&T rows"
    for col in ("game_dt", "game_date"):
        if col in box.columns:
            dates = {str(v)[:10] for v in pd.concat([hb[col], ab[col]]).dropna()}
            if date and dates and dates != {date[:10]}:
                return False, f"projection date {sorted(dates)} != {date[:10]}"
    if "game_id" in box.columns:
        gh, ga = set(hb["game_id"].dropna().astype(str)), set(ab["game_id"].dropna().astype(str))
        if gh and ga and gh != ga:
            return False, "home and away projections carry different game ids"
    for col in ("opp_team_id", "opponent_team_id", "opp_id"):
        if col in box.columns:
            if set(pd.to_numeric(hb[col], errors="coerce").dropna().astype(int)) - {away_id}:
                return False, "projection opponent does not match"
    for col in ("updated_at", "update_dt"):
        if col in box.columns:
            ts = pd.to_datetime(pd.concat([hb[col], ab[col]]), errors="coerce", utc=True).dropna()
            if len(ts) and (pd.Timestamp.now(tz="UTC") - ts.max()).total_seconds() > 36 * 3600:
                return False, f"projections stale (last update {ts.max()})"
    return True, "ok"


def logit_pool(p_model: float, p_mkt: float, w: float) -> float:
    """Market-anchored probability: sigmoid((1-w)*logit(market) + w*logit(model)).
    Fitted on Nov 5 (447 settled markets): w ~ 0.2-0.35, i.e. the market carries most of the information and the
    model adds a measured tilt. A big model/market gap moves the price far less than a linear blend would."""
    lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
    return 1.0 / (1.0 + math.exp(-((1.0 - w) * lg(p_mkt) + w * lg(p_model))))


def pool3(p_model: float, p_mkt: float, p_book: Optional[float], w_model: float, w_book: float) -> float:
    """Logit pool of model, Kalshi and (if available) sportsbook consensus; weights are fitted walk-forward."""
    if p_book is None or w_book <= 0:
        return logit_pool(p_model, p_mkt, w_model)
    lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
    w_k = max(0.0, 1.0 - w_model - w_book)
    return 1.0 / (1.0 + math.exp(-(w_k * lg(p_mkt) + w_model * lg(p_model) + w_book * lg(p_book))))


def conservative_edge(p_used: float, p_mkt: float, price_c: float, fee_c: float, p_model: float, n_sims: int, w_m: float,
                      p_shadow: Optional[float] = None, as_spread_frac: float = 1.0, as_buffer_c: float = 0.5,
                      shadow_z: float = 1.0, n_shadow: int = 0, w_sd: float = 0.0, exec_alpha_c: float = 0.0,
                      exec_sd_c: float = 0.0, lcb_z: float = 1.282) -> Tuple[float, float, dict]:
    """90% lower confidence bound (LCB) of the edge, in cents per contract, that we expect to KEEP after a fill, and the
    matching conservative win probability (for sizing).

    Edge_mean = directional alpha + execution alpha - fee
              = (100*p_used - price - fee) - as_spread_frac x (100*p_mkt - price) + exec_alpha_c
      (as_spread_frac = 1: no spread credit; exec_alpha_c = measured raw 30 s markout of maker fills on this side.)
    Measured stochastic uncertainty (standard errors only, combined in quadrature):
      sigma_MC   = 100 x w x sqrt(p(1-p)/n)                               Monte Carlo
      sigma_cal  = 100 x p_used(1-p_used) x |logit p_model - logit p_mkt| x w_sd   pool-weight estimation error
      sigma_exec = s.e. of the execution prior
    Model-specification uncertainty (not Gaussian, kept separate):
      H_shadow   = shadow_z x w x 100 x max(0, |p_model - p_shadow| - s.e. of the main-shadow difference)
    Deterministic penalties: as_buffer_c.
    LCB90 = Edge_mean - deterministic - lcb_z x sigma_measured - H_shadow   (each uncertainty is charged exactly once)"""
    lg = lambda q: math.log(min(max(q, 1e-4), 1 - 1e-4) / (1 - min(max(q, 1e-4), 1 - 1e-4)))
    pq = max(p_model * (1.0 - p_model), 1e-6)
    nominal = 100.0 * p_used - price_c - fee_c
    spread_part = as_spread_frac * max(0.0, 100.0 * p_mkt - price_c)
    mean = nominal - spread_part + exec_alpha_c
    mc = 100.0 * w_m * math.sqrt(pq / max(1, n_sims))
    cal = 100.0 * p_used * (1.0 - p_used) * abs(lg(p_model) - lg(p_mkt)) * max(0.0, w_sd)
    ex = max(0.0, exec_sd_c)
    sigma = math.sqrt(mc * mc + cal * cal + ex * ex)
    H = 0.0
    if p_shadow is not None and math.isfinite(p_shadow):
        se_diff = math.sqrt(pq / max(1, n_sims) + (pq / n_shadow if n_shadow > 0 else 0.0))
        H = max(0.0, shadow_z) * w_m * 100.0 * max(0.0, abs(p_model - p_shadow) - se_diff)
        H = H if math.isfinite(H) else 0.0
    lcb = mean - as_buffer_c - lcb_z * sigma - H
    p_cons = min(0.999, max(0.001, (price_c + fee_c + lcb) / 100.0))
    return lcb, p_cons, dict(mean_c=mean, nominal_c=nominal, spread_c=spread_part, se_c=mc, cal_c=cal, exec_c=exec_alpha_c,
                             exec_sd_c=ex, sigma_c=sigma, model_c=H, det_c=as_buffer_c, lcb_c=lcb)


def inventory_skew_c(player_exposure_usd: float, bankroll: float, player_cap: float, max_c: float = 1.0) -> float:
    """Extra LCB edge required as a player's exposure approaches its cap: max_c x (exposure / cap)^2, capped at max_c.
    25% of cap -> +0.06c, 50% -> +0.25c, 75% -> +0.56c, 100% -> +1.00c."""
    cap = max(1e-9, player_cap * bankroll)
    return float(min(max_c, max_c * (max(0.0, player_exposure_usd) / cap) ** 2))


def csv_append(path: str, rows: List[dict]):
    """Append rows keeping the file's existing header (a column added by a newer version is dropped for that file rather
    than shifting every later column; a new file gets every column)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path) as f:
                cols = f.readline().strip().split(",")
            df = df.reindex(columns=cols)
            df.to_csv(path, mode="a", header=False, index=False)
            return
        except Exception:
            pass
    df.to_csv(path, mode="w", header=True, index=False)


def apply_fill(d: dict, contracts: float, price_cents: float, fee_usd: float):
    """Average-cost position accounting. Entry: remaining cost basis += contracts x (price + fee). Exit: basis falls by
    the AVERAGE entry cost of the contracts sold (not by the sale proceeds), and realized P&L = proceeds - removed basis
    - exit fee. cost_usd is the remaining cost basis (what the optimizer and caps see)."""
    d.setdefault("realized_pnl", 0.0)
    if contracts >= 0:
        d["contracts"] += contracts
        d["cost_usd"] += contracts * price_cents / 100.0 + fee_usd
    else:
        sold = min(-contracts, max(0.0, d["contracts"]))
        avg = d["cost_usd"] / d["contracts"] if d["contracts"] > 1e-9 else 0.0
        removed = avg * sold
        d["contracts"] -= sold
        d["cost_usd"] = max(0.0, d["cost_usd"] - removed)
        d["realized_pnl"] += sold * price_cents / 100.0 - fee_usd - removed
    d["remaining_cost_basis"] = d["cost_usd"]
    d["avg_entry_price"] = 100.0 * d["cost_usd"] / d["contracts"] if d["contracts"] > 1e-9 else 0.0


def venue_fill_ts(f: dict, now: float) -> Tuple[float, str]:
    """(fill time, source). Kalshi's own fill timestamp when present and plausible; otherwise local receipt time,
    flagged, never silently treated as exact."""
    for k in ("created_time", "ts", "created_ts", "time"):
        v = f.get(k)
        if v in (None, ""):
            continue
        try:
            t = float(v) if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit() else \
                datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
            t = t / 1000.0 if t > 1e11 else t                 # milliseconds
            if now - 6 * 3600 <= t <= now + 5:
                return t, "venue"
        except Exception:
            continue
    logger.warning("Fill has no valid venue timestamp; markout clock starts at local receipt (flagged ts_source=local_fallback).")
    return now, "local_fallback"


def model_shocks(items: List[dict], n: int, sds: Optional[dict], seed: int = 12345) -> np.ndarray:
    """Per-scenario P&L shift (dollars) from shared MODEL errors (v2026.37): in each scenario draw latent shocks
    eps_global, eps_stat[s], eps_game[g], eps_team[t], eps_player[p] ~ N(0, sd) (probability units) and move every item's
    win probability by -(sum of the shocks it is exposed to). The expected P&L of contracts with a lower true probability
    falls by contracts x eps dollars, consistently across every item sharing a factor. Returns an (n_items, n) array."""
    out = np.zeros((len(items), n))
    if not sds or n == 0 or not items or not any(float(v or 0) > 0 for v in sds.values()):
        return out
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, float(sds.get("model", 0.0)), n)
    draws: Dict[tuple, np.ndarray] = {}
    def shock(kind, key):
        if key in (None, "", "None") or float(sds.get(kind, 0.0)) <= 0:
            return 0.0
        k = (kind, str(key))
        if k not in draws:
            draws[k] = rng.normal(0.0, float(sds[kind]), n)
        return draws[k]
    for i, it in enumerate(items):
        e = g + shock("stat", it.get("stat")) + shock("game", it.get("game_key")) + shock("team", it.get("team_id")) \
            + shock("player", it.get("player_name"))
        out[i] = e
    return out


def cvar_scale(orders: List[dict], others: List[dict], bankroll: float, limit_frac: float = 0.02, q: float = 0.95,
               iters: int = 20, shock_sds: Optional[dict] = None) -> Tuple[List[dict], dict]:
    """Tail-loss control after sizing. Every item carries `outcome` (simulated win vector), `contracts`, `unit_c` (all-in
    cost per contract, cents) and its factor tags (stat, game_key, team_id, player_name). others = held positions +
    pending + resting orders not replaced by `orders`. FULL contract quantities are used: fill probability never reduces
    worst-case exposure. Per scenario s:
        PnL_s = sum_i contracts_i x [(100 x win_is - unit_c_i) / 100 - eps_is]
    with eps the shared model-error shocks (model_shocks): basketball outcome AND misspecification uncertainty.
    Finds the largest alpha in [0,1] such that CVaR95 of (others + alpha x new orders, rounded DOWN) <= limit."""
    vecs = [it["outcome"] for it in orders + others if it.get("outcome") is not None]
    n = min((len(v) for v in vecs), default=0)
    if n == 0 or not orders:
        return orders, {"alpha": 1.0, "cvar95": None, "p99": None}
    eps_o = model_shocks(orders + others, n, shock_sds)
    eps_new, eps_oth = eps_o[:len(orders)], eps_o[len(orders):]

    def pnl(items, eps, scale=None):
        tot = np.zeros(n)
        for j, it in enumerate(items):
            k = it["contracts"] if scale is None else math.floor(it["contracts"] * scale + 1e-9)
            if k <= 0:
                continue
            o = it.get("outcome")
            win = np.zeros(n) if o is None else np.asarray(o[:n], dtype=float)   # no outcome vector: assume full loss
            tot += k * ((100.0 * win - it["unit_c"]) / 100.0 - eps[j])
        return tot

    base = pnl(others, eps_oth)
    def risk(alpha):
        L = -(base + pnl(orders, eps_new, alpha))
        var = np.quantile(L, q)
        return float(L[L >= var].mean()), float(np.quantile(L, 0.99))
    limit = limit_frac * bankroll
    cv1, p991 = risk(1.0)
    if cv1 <= limit:
        return orders, {"alpha": 1.0, "cvar95": round(cv1, 2), "p99": round(p991, 2)}
    lo, hi = 0.0, 1.0
    if risk(0.0)[0] > limit:
        return [], {"alpha": 0.0, "cvar95": round(risk(0.0)[0], 2), "p99": round(risk(0.0)[1], 2), "note": "held risk alone exceeds the limit"}
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if risk(mid)[0] <= limit:
            lo = mid
        else:
            hi = mid
    out = []
    for o in orders:
        k = math.floor(o["contracts"] * lo + 1e-9)
        if k >= 1:
            o = dict(o, contracts=k, risk_usd=round(k * o["unit_c"] / 100.0, 2),
                     ev_usd=round(k * float(o.get("expected_edge_c", 0.0)) / 100.0, 2))
            out.append(o)
    cv, p99 = risk(lo)          # re-evaluated on the rounded quantities
    return out, {"alpha": round(lo, 4), "cvar95": round(cv, 2), "p99": round(p99, 2)}


def throttle(dpnl: float, bankroll: float, tox_c: Optional[float], tox_warn: float = -3.0, tox_stop: float = -6.0):
    """(size multiplier, join-bid-only, no-new-player-exposure, reason). Continuous risk response instead of one cliff:
    drawdown zones 0/-1/-2% of bankroll -> 100/65/35% sizing (-3% stays the hard stop), and execution health: if the
    recent fills' markouts are toxic, quote less aggressively and smaller; if very toxic, pause entries."""
    dd = dpnl / max(bankroll, 1e-9)
    mult, join, no_add, why = 1.0, False, False, []
    if dd <= -0.02:
        mult, no_add = 0.35, True; why.append(f"drawdown {dd:.1%}: 35% size, no new player exposure")
    elif dd <= -0.01:
        mult = 0.65; why.append(f"drawdown {dd:.1%}: 65% size")
    if tox_c is not None and tox_c <= tox_stop:
        mult = 0.0; why.append(f"fills toxic (recent markout {tox_c:+.1f}c): entries paused")
    elif tox_c is not None and tox_c <= tox_warn:
        mult, join = mult * 0.5, True; why.append(f"fills toxic (recent markout {tox_c:+.1f}c): half size, join the bid")
    return mult, join, no_add, "; ".join(why)


H1_BUCKETS = ("Q1", "Q2", "HT")


def period_bucket(period: int, per_clock: Optional[float] = None) -> str:
    """Q1, Q2, HT (halftime: Q2 clock 0:00), Q3, Q4 (overtime counts as Q4). One definition: kalshi_results.phase_of_period
    (fit_calibration.py and grade_predictions.py use the same module)."""
    return phase_of_period(period, per_clock)


GAP_BINS = ((0.0, 0.03, "g0-3"), (0.03, 0.06, "g3-6"), (0.06, 1.01, "g6-10"))


def gap_bin(gap: float) -> str:
    for lo, hi, name in GAP_BINS:
        if lo <= gap < hi:
            return name
    return "g6-10"


def weight_for(stat: str, period: int, cell_w: dict, stat_w: dict, w_global: float, per_clock: Optional[float] = None,
               h1_strict: bool = False, gap: Optional[float] = None) -> float:
    """Model weight for this regime: a fitted (stat, period) cell, else (any stat, period), else the stat, else global.
    Cells exist only where they beat the coarser weight on held-out games (fit_calibration.py). h1_strict (weights from
    calibration.json, not a manual research weight): a first-half regime NEVER borrows the second-half/global weight;
    without its own earned cell its model weight is 0."""
    b = period_bucket(period, per_clock)
    keys = ([f"{stat}|{b}|{gap_bin(gap)}"] if gap is not None else []) + [f"{stat}|{b}", f"*|{b}"]
    for k in keys:
        if k in cell_w:
            return cell_w[k]
    if h1_strict and b in H1_BUCKETS:
        return 0.0
    return stat_w.get(stat, w_global)


def side_book(yes_bid: int, yes_ask: int, side: str) -> Tuple[int, int]:
    """(bid, ask) in cents for buying `side`."""
    return (yes_bid, yes_ask) if side == "yes" else (100 - yes_ask, 100 - yes_bid)


def revalidate(bet: dict, snap_yes_bid: int, snap_yes_ask: int, now_yes_bid: int, now_yes_ask: int,
               max_mid_move_c: float = 1.0) -> Tuple[bool, str]:
    """Pre-send check against the book as it is NOW (the cycle snapshot can be many seconds old by order time).
    A maker order must still rest (price below the current ask, never post-only cross); a taker order's ask must be
    unchanged; and the mid may not have moved more than max_mid_move_c since the snapshot the price was computed on."""
    if not (1 <= now_yes_bid < now_yes_ask <= 99):
        return False, "book one-sided or crossed"
    if abs((now_yes_bid + now_yes_ask) - (snap_yes_bid + snap_yes_ask)) / 2.0 > max_mid_move_c + 1e-9:
        return False, "book moved"
    bid, ask = side_book(now_yes_bid, now_yes_ask, bet["side"])
    px = int(round(bet["price_cents"]))
    if bet.get("is_maker", True):
        if px >= ask:
            return False, "would cross"
        if px < bid:
            return False, "outbid"
    elif px != ask:
        return False, "ask changed"
    return True, "ok"


def contradicts(side_a: str, line_a: float, side_b: str, line_b: float) -> bool:
    """True if two bets on the SAME player and stat can never both win: an UNDER (NO) at line L1 together
    with an OVER (YES) at line L2 >= L1. Both lose whenever the result lands between them. (An OVER below an
    UNDER is a "middle" where both can win; that is allowed.)"""
    if side_a == side_b:
        return False
    under_line, over_line = (line_a, line_b) if side_a == "no" else (line_b, line_a)
    return over_line >= under_line


def drop_contradictions(cands: List[dict], positions: Dict[Tuple[str, str], dict]) -> Tuple[List[dict], int]:
    """Keep the best-ROI candidates that contradict neither a held position nor an already-kept candidate."""
    held = [(v.get("player"), v.get("stat"), k[1], v.get("line")) for k, v in positions.items() if v.get("stat")]
    kept, dropped = [], 0
    for c in sorted(cands, key=lambda c: -c["roi"]):
        mine = [(c["player_name"], c["stat"], c["side"], c["line"])]
        book = held + [(k["player_name"], k["stat"], k["side"], k["line"]) for k in kept]
        if any(p == q[0] and s == q[1] and l2 is not None and contradicts(sd, ln, q[2], float(l2))
               for (p, s, sd, ln) in mine for q in book for l2 in [q[3]]):
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped
LEDGER_COLS = ["timestamp", "mode", "order_id", "fill_id", "ticker", "game", "player", "stat", "line", "side",
               "order_type", "price_cents", "fee_usd", "contracts", "cost_usd", "p_model", "p_mkt", "p_used",
               "ev_pct", "joint_kelly_pct", "engine", "trader", "run_id", "quote_policy", "policy_id", "policy_family_id",
               "exec_model_id", "exec_prior_id", "arm", "tier", "confidence", "cell", "team_id", "fill_prob", "cond_edge_c"]


def ledger_append(row: dict):
    """Append one fill. An existing ledger keeps its own header (a ledger started by an older version has no engine
    column; appending wider rows to it would misalign every later read)."""
    row = dict(row, engine=mg.ENGINE_VERSION, trader=TRADER_VERSION, run_id=RUN_ID, quote_policy=QUOTE_POLICY, **POLICY)
    cols = LEDGER_COLS
    new = not os.path.exists(LEDGER_CSV)
    if not new:
        try:
            with open(LEDGER_CSV) as f:
                cols = f.readline().strip().split(",")
        except Exception:
            pass
    pd.DataFrame([{c: row.get(c, "") for c in cols}]).to_csv(LEDGER_CSV, mode="a", header=new, index=False)


def load_positions_from_ledger() -> Dict[Tuple[str, str], dict]:
    pos: Dict[Tuple[str, str], dict] = {}
    if not os.path.exists(LEDGER_CSV):
        return pos
    df = pd.read_csv(LEDGER_CSV)
    for _, r in df.iterrows():
        k = (str(r["ticker"]), str(r["side"]))
        d = pos.setdefault(k, {"contracts": 0.0, "cost_usd": 0.0, "game_key": str(r["game"]), "player": str(r["player"]),
                               "stat": str(r["stat"]), "line": float(r["line"]),
                               "team_id": str(r["team_id"]) if "team_id" in df.columns and pd.notna(r.get("team_id")) else "",
                               "tier": str(r["tier"]) if "tier" in df.columns and pd.notna(r.get("tier")) else "PRODUCTION"})
        if "tier" in df.columns and str(r.get("tier")) == "PRODUCTION":
            d["tier"] = "PRODUCTION"
        apply_fill(d, float(r["contracts"]), float(r["price_cents"]), float(r["fee_usd"] or 0.0))   # same average-cost rule as live
    return {k: v for k, v in pos.items() if v["contracts"] > 1e-9}


def game_date_token(game_code: str) -> str:
    """NBA gameCode '20261105/ORLPHX' -> Kalshi date token '26NOV05'."""
    try:
        return datetime.strptime(game_code.split("/")[0], "%Y%m%d").strftime("%y%b%d").upper()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# KALSHI CLIENT
# ─────────────────────────────────────────────────────────────

@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    title: str
    player_id: int
    player_name: str
    stat: str
    line: float
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    exchange_index: Optional[int] = None

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 200.0


class KalshiClient:
    BASE = "https://external-api.kalshi.com/trade-api/v2"

    def __init__(self, key_id: str, pem_path: str, paper: bool = True):
        self.key_id = key_id
        self.paper = paper
        self.ws = None          # KalshiWSBook when --ws: top_of_book reads the local L2 book first, REST as fallback
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json", "User-Agent": "nba-inplay-trader/2026.16"})
        self._private_key = None
        if pem_path and os.path.exists(pem_path):
            try:
                self._private_key = load_private_key(pem_path)
            except Exception as e:
                logger.error(f"Error loading PEM: {e}")
        self.fees: Dict[str, Tuple[float, float]] = {}
        self._warned: set = set()

    @property
    def can_trade(self) -> bool:
        return (not self.paper) and self._private_key is not None and bool(self.key_id)

    def _auth(self, method: str, path: str) -> Dict[str, str]:
        return get_auth_headers(self.key_id, self._private_key, method, f"/trade-api/v2{path}")

    def _warn_once(self, key: str, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    # ── account ─────────────────────────────────────────────
    def get_sports_shard_balance(self) -> float:
        if not self.can_trade:
            return float("inf")
        path = "/portfolio/balance"
        try:
            resp = self._session.get(f"{self.BASE}{path}", headers=self._auth("GET", path), timeout=5)
            if resp.status_code != 200:
                logger.error(f"[BALANCE {resp.status_code}] {resp.text[:200]}")
                return 0.0
            data = resp.json()
            for b in data.get("balance_breakdown", []) or []:
                if int(b.get("exchange_index", -1)) == 3:
                    return float(b.get("balance", 0.0))
            return float(data.get("balance_dollars", 0.0))
        except Exception as e:
            logger.error(f"Error reading balance: {e}")
            return 0.0

    def load_series_fees(self):
        """fee_type/fee_multiplier per series. Unknown -> assume maker fees apply (conservative)."""
        for s in KALSHI_SERIES:
            taker_m, maker_m, known = 1.0, 1.0, False
            try:
                r = self._session.get(f"{self.BASE}/series/{s}", timeout=8)
                if r.status_code == 200:
                    ser = r.json().get("series", r.json())
                    ft = str(ser.get("fee_type", "quadratic_with_maker_fees"))
                    mult = float(ser.get("fee_multiplier", 1.0) or 1.0)
                    taker_m = mult
                    maker_m = mult if "maker" in ft else 0.0
                    known = "fee_type" in ser
                    logger.info(f"Fees {s}: fee_type={ft} multiplier={mult} -> resting orders "
                                f"{'FREE' if known and maker_m == 0 else 'PAY a maker fee'}")
                else:
                    logger.warning(f"Fee lookup for {s} returned {r.status_code}; treating as fee-charging.")
            except Exception as e:
                logger.warning(f"Fee lookup failed for {s} ({e}); treating as fee-charging.")
            self.fees[s] = (taker_m, maker_m, known)

    def fee_free_series(self) -> set:
        return {s for s, (_, mm, known) in self.fees.items() if known and mm == 0.0}

    def fee_cents(self, ticker: str, price: int, is_maker: bool) -> float:
        tm, mm, _ = self.fees.get(ticker.split("-")[0], (1.0, 1.0, False))
        return kalshi_fee_cents(price, is_maker, tm, mm)

    # ── market data ─────────────────────────────────────────
    @staticmethod
    def _cents(m: dict, f_dollars: str, f_cents: str) -> int:
        v = m.get(f_dollars)
        if v not in (None, ""):
            try:
                return int(round(float(v) * 100))
            except (ValueError, TypeError):
                pass
        v = m.get(f_cents)
        try:
            return int(v) if v is not None else 0
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _threshold(m: dict) -> Optional[int]:
        """Smallest integer stat value that settles YES (YES iff stat >= k)."""
        k_struct = None
        st, fs = str(m.get("strike_type", "")), m.get("floor_strike")
        if fs is not None:
            try:
                fs = float(fs)
                if st == "greater":
                    k_struct = int(math.floor(fs)) + 1
                elif st == "greater_or_equal":
                    k_struct = int(math.ceil(fs))
            except (TypeError, ValueError):
                pass
        k_text = None
        for txt in (m.get("yes_sub_title", ""), m.get("title", ""), m.get("subtitle", "")):
            mt = re.search(r"(\d+)\s*\+", str(txt or ""))
            if mt:
                k_text = int(mt.group(1))
                break
            mo = re.search(r"\bover\s+(\d+(?:\.\d+)?)", str(txt or ""), re.I)
            if mo:
                k_text = int(math.floor(float(mo.group(1)))) + 1
                break
        if k_struct is not None and k_text is not None and k_struct != k_text:
            return None   # structured strike and title disagree -> do not guess
        return k_struct if k_struct is not None else k_text

    def fetch_open_props(self, names: List[Tuple[str, int, str]], date_tokens: set) -> List[KalshiMarket]:
        markets = []
        for series in KALSHI_SERIES:
            cursor, pages = "", 0
            while pages < 10:
                pages += 1
                params = {"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                try:
                    resp = self._session.get(f"{self.BASE}/events", params=params, timeout=8)
                    if resp.status_code != 200:
                        logger.warning(f"[EVENTS {series} {resp.status_code}] {resp.text[:150]}")
                        break
                    data = resp.json()
                except Exception as e:
                    logger.warning(f"[EVENTS {series}] {e}")
                    break
                for ev in data.get("events", []) or []:
                    ev_ticker = str(ev.get("event_ticker", "")).upper()
                    if not any(tok and tok in ev_ticker for tok in date_tokens):
                        continue   # not tonight's game (e.g. tomorrow's props already listed)
                    for m in ev.get("markets", []) or []:
                        km = self._parse_market(m, ev, series, names)
                        if km:
                            markets.append(km)
                cursor = data.get("cursor") or ""
                if not cursor:
                    break
        return markets

    def _parse_market(self, m: dict, ev: dict, series: str, names: List[Tuple[str, int, str]]) -> Optional[KalshiMarket]:
        ticker = str(m.get("ticker", ""))
        if not ticker.startswith(series) or str(m.get("status", "active")) != "active":
            return None
        stat = SERIES_STAT[series]

        # Player: longest whole-word name match, most specific field first; ambiguity -> skip.
        matched = None
        for field in (m.get("yes_sub_title", ""), m.get("title", ""), ev.get("title", ""), ev.get("sub_title", "")):
            text = " " + re.sub(r"[^a-z0-9]+", " ", mg.normalize_player_name(str(field or ""))) + " "
            hits = [(len(n), pid, orig) for n, pid, orig in names if f" {n} " in text]
            if hits:
                best = max(h[0] for h in hits)
                top = {(pid, orig) for L, pid, orig in hits if L == best}
                if len(top) == 1:
                    matched = top.pop()
                break
        if not matched:
            return None

        k = self._threshold(m)
        if k is None or k < 1:
            self._warn_once(f"strike:{ticker}", f"[SKIP] {ticker}: strike unclear (strike_type={m.get('strike_type')}, "
                                                f"floor={m.get('floor_strike')}, yes_sub_title={m.get('yes_sub_title')!r})")
            return None

        yb = self._cents(m, "yes_bid_dollars", "yes_bid")
        ya = self._cents(m, "yes_ask_dollars", "yes_ask")
        nb = self._cents(m, "no_bid_dollars", "no_bid") or (100 - ya if 0 < ya < 100 else 0)
        na = self._cents(m, "no_ask_dollars", "no_ask") or (100 - yb if 0 < yb < 100 else 0)
        if not (1 <= yb < ya <= 99):
            return None   # one-sided or crossed book: no reliable fair value
        ex = m.get("exchange_index")
        return KalshiMarket(ticker=ticker, event_ticker=str(ev.get("event_ticker", "")), title=str(m.get("title", "")),
                            player_id=matched[0], player_name=matched[1], stat=stat, line=k - 0.5,
                            yes_bid=yb, yes_ask=ya, no_bid=nb, no_ask=na,
                            exchange_index=int(ex) if ex is not None else None)

    # ── orders ──────────────────────────────────────────────
    def send_order(self, m: KalshiMarket, side: str, price_cents: int, count: int, is_maker: bool,
                   ttl_sec: int) -> Tuple[Optional[str], float]:
        """Returns (order_id, contracts filled immediately). Live only."""
        path = "/portfolio/events/orders"
        yes_px = price_cents if side == "yes" else 100 - price_cents
        payload = {
            "ticker": m.ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": "bid" if side == "yes" else "ask",      # V2: YES leg only; ask = sell YES = buy NO
            "count": str(int(count)),
            "price": f"{yes_px / 100.0:.4f}",
            "time_in_force": "good_till_canceled" if is_maker else "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": bool(is_maker),
            "cancel_order_on_pause": True,
        }
        if is_maker:
            payload["expiration_time"] = int(time.time()) + int(ttl_sec)   # backstop if cancel fails
        if m.exchange_index is not None:
            payload["exchange_index"] = m.exchange_index
        try:
            resp = self._session.post(f"{self.BASE}{path}", headers=self._auth("POST", path), json=payload, timeout=5.0)
            if resp.status_code in (200, 201):
                d = resp.json()
                d = d.get("order", d)
                return str(d.get("order_id", "")), float(d.get("fill_count", 0) or 0)
            logger.error(f"[ORDER ERROR {resp.status_code}] {m.ticker}: {resp.text[:300]}")
        except Exception as e:
            logger.error(f"[NETWORK ERROR] {e}")
        return None, 0.0

    def send_exit(self, m: KalshiMarket, side: str, price_cents: int, count: int) -> Tuple[Optional[str], float]:
        """Sell a held position: IOC + reduce_only, so it can only shrink the position, never flip it."""
        path = "/portfolio/events/orders"
        yes_px = price_cents if side == "yes" else 100 - price_cents
        payload = {"ticker": m.ticker, "client_order_id": str(uuid.uuid4()),
                   "side": "ask" if side == "yes" else "bid",   # sell YES / sell NO (= bid on YES)
                   "count": str(int(count)), "price": f"{yes_px / 100.0:.4f}", "time_in_force": "immediate_or_cancel",
                   "self_trade_prevention_type": "taker_at_cross", "post_only": False, "reduce_only": True}
        if m.exchange_index is not None:
            payload["exchange_index"] = m.exchange_index
        try:
            resp = self._session.post(f"{self.BASE}{path}", headers=self._auth("POST", path), json=payload, timeout=5.0)
            if resp.status_code in (200, 201):
                d = resp.json()
                d = d.get("order", d)
                return str(d.get("order_id", "")), float(d.get("fill_count", 0) or 0)
            logger.error(f"[EXIT ERROR {resp.status_code}] {m.ticker}: {resp.text[:300]}")
        except Exception as e:
            logger.error(f"[NETWORK ERROR] {e}")
        return None, 0.0

    def get_positions(self) -> Optional[Dict[str, float]]:
        """Exchange-authoritative net positions {ticker: +YES / -NO contracts}; None if they cannot be read."""
        path, out, cursor = "/portfolio/positions", {}, ""
        try:
            for _ in range(20):
                params = {"limit": 1000, "count_filter": "position"}
                if cursor:
                    params["cursor"] = cursor
                r = self._session.get(f"{self.BASE}{path}", headers=self._auth("GET", path), params=params, timeout=8)
                if r.status_code != 200:
                    logger.error(f"[POSITIONS {r.status_code}] {r.text[:200]}")
                    return None
                d = r.json()
                for p in d.get("market_positions", []) or []:
                    v = p.get("position_fp", p.get("position", 0))
                    try:
                        v = float(v or 0)
                    except (TypeError, ValueError):
                        continue
                    if abs(v) > 1e-9:
                        out[str(p.get("ticker", ""))] = v
                cursor = d.get("cursor") or ""
                if not cursor:
                    break
            return out
        except Exception as e:
            logger.error(f"[POSITIONS] {e}")
            return None

    @staticmethod
    def parse_top(j: dict) -> Optional[Tuple[float, float, float, float]]:
        """(yes_bid_c, qty_bid, yes_ask_c, qty_ask) from an /orderbook response; None if one-sided or unreadable."""
        ob = j.get("orderbook_fp") or j.get("orderbook") or {}
        def top(levels):
            best = None
            for lv in levels or []:
                px, q = float(lv[0]), float(lv[1])
                px = px * 100.0 if px <= 1.0 else px          # dollars or cents
                if q > 0 and (best is None or px > best[0]):
                    best = (px, q)
            return best
        yb = top(ob.get("yes_dollars") or ob.get("yes"))
        nb = top(ob.get("no_dollars") or ob.get("no"))
        if not yb or not nb:
            return None
        bid, qb, ask, qa = yb[0], yb[1], 100.0 - nb[0], nb[1]
        if not (0 < bid < ask < 100):
            return None
        return bid, qb, ask, qa

    def ws_headers(self) -> Dict[str, str]:
        h = get_auth_headers(self.key_id, self._private_key, "GET", "/trade-api/ws/v2")
        h.pop("Content-Type", None)
        return h

    def top_of_book(self, ticker: str) -> Optional[Tuple[float, float, float, float]]:
        """Current top of book, fetched now. The order book is public: signed only when a key is loaded, so paper and
        live read the same data (v2026.28 signed unconditionally, so paper mode never saw the book)."""
        if self.ws is not None:
            w = self.ws.fresh(ticker)
            if w is not None:
                return float(w[0]), float(w[1]), float(w[2]), float(w[3])
        path = f"/markets/{ticker}/orderbook"
        try:
            hdr = self._auth("GET", path) if self._private_key is not None and self.key_id else {}
            r = self._session.get(f"{self.BASE}{path}", headers=hdr, params={"depth": 1}, timeout=4)
            if r.status_code != 200:
                return None
            return self.parse_top(r.json())
        except Exception:
            return None

    def microprice(self, ticker: str) -> Optional[float]:
        """Size-weighted YES fair price from the top of book: (ask*Q_bid + bid*Q_ask)/(Q_bid + Q_ask). None on any doubt.
        Same midpoint, different depth = different information (800 bid vs 12 offered is not 12 vs 800)."""
        tb = self.top_of_book(ticker)
        if tb is None:
            return None
        bid, qb, ask, qa = tb
        if qb < 5 or qa < 5:          # thin or spoofable top of book: use the plain mid
            return None
        return (ask * qb + bid * qa) / (qb + qa) / 100.0

    def cancel_order(self, order_id: str, ticker: str) -> bool:
        path = f"/portfolio/events/orders/{order_id}"
        try:
            resp = self._session.delete(f"{self.BASE}{path}", headers=self._auth("DELETE", path),
                                        params={"market_ticker": ticker}, timeout=5.0)
            return resp.status_code in (200, 404)    # 404: already filled / expired
        except Exception as e:
            logger.error(f"[CANCEL ERROR] {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """Exchange-authoritative state of one order: {"status": resting|canceled|executed|expired, "fill_count": float or
        None, "remaining": float or None}; None if it cannot be read (then nothing is concluded from it)."""
        if not self.can_trade:
            return None
        for path in (f"/portfolio/orders/{order_id}", f"/portfolio/events/orders/{order_id}"):
            try:
                r = self._session.get(f"{self.BASE}{path}", headers=self._auth("GET", path), timeout=5)
            except Exception as e:
                logger.debug(f"[ORDER STATUS] {order_id}: {e}")
                continue
            if r.status_code != 200:
                continue
            try:
                d = r.json()
                d = d.get("order", d)
            except Exception:
                continue
            st = str(d.get("status", "") or "").lower()
            if not st:
                continue
            def num(*ks):
                for k in ks:
                    v = d.get(k)
                    if v not in (None, ""):
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return None
            return {"status": st, "fill_count": num("fill_count_fp", "fill_count"),
                    "remaining": num("remaining_count_fp", "remaining_count")}
        return None

    def get_fills(self, min_ts: int) -> Optional[List[dict]]:
        """Every fill since min_ts. None if the FIRST page cannot be read (an unreadable fills feed proves nothing; the
        state machine must not treat it as 'no fills')."""
        path, fills, cursor = "/portfolio/fills", [], ""
        for page in range(20):
            params = {"min_ts": int(min_ts), "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = self._session.get(f"{self.BASE}{path}", headers=self._auth("GET", path), params=params, timeout=8)
                if resp.status_code != 200:
                    logger.error(f"[FILLS {resp.status_code}] {resp.text[:200]}")
                    return None          # a partial fills list proves nothing (review fix): never release on it
                d = resp.json()
            except Exception as e:
                logger.error(f"[FILLS] {e}")
                return None
            fills.extend(d.get("fills", []) or [])
            cursor = d.get("cursor") or ""
            if not cursor:
                break
        return fills


# ─────────────────────────────────────────────────────────────
# ORDER BOOK-KEEPING (live + paper share one path)
# ─────────────────────────────────────────────────────────────

def position_mismatches(local: Dict[Tuple[str, str], dict], exch: Dict[str, float], open_tickers: Optional[set] = None) -> List[str]:
    """Compare local net positions with the exchange for today's/yesterday's NBA prop tickers (|difference| >= 1).
    Only OPEN markets are compared: once a market settles the exchange shows 0 while the ledger still shows the fill."""
    now = datetime.now()
    toks = {(now - timedelta(days=d)).strftime("%y%b%d").upper() for d in (0, 1)}
    scope = lambda t: t.startswith("KXNBA") and any(k in t for k in toks) and (open_tickers is None or t in open_tickers)
    net: Dict[str, float] = {}
    for (t, side), v in local.items():
        net[t] = net.get(t, 0.0) + (v["contracts"] if side == "yes" else -v["contracts"])
    out = []
    for t in sorted({t for t in set(net) | set(exch) if scope(t)}):
        if abs(net.get(t, 0.0) - exch.get(t, 0.0)) >= 1.0:
            out.append(f"{t}: local {net.get(t, 0.0):+.0f} vs exchange {exch.get(t, 0.0):+.0f}")
    return out


QUOTE_KEYS = ("pi_b", "greedy_px", "action_price", "action_prob", "tier", "confidence", "cell", "authority_weight", "gap_bin", "phase", "fee_c", "aggr", "stop", "ttl",
              "queue_ahead", "requested_qty", "size_bucket", "qty_over_best_depth", "qty_over_queue_ahead", "log_order_qty", "pi_b_action", "t_since_action_s", "alpha_c", "em_c", "em_model_c", "em_prior_c", "qev_c", "qev_sd_c",
              "fill_prob", "cond_edge_c", "challenger_pf", "challenger_qev", "is_probe", "side_bid", "side_ask",
              "mid_vel_250ms", "mid_vel_1s", "spread_vel", "imb_vel", "bid_depletion", "ask_replenish", "has_ws", "team_id")


def quote_log(oid: str, bet: dict, m: "KalshiMarket", count: int):
    """Every posted order with the book around it; joined later to its terminal event (order_id) to fit
    P(fill before TTL | quote, state) and the execution residual. v2026.37: every execution feature, the authority tier,
    and the policy identity are logged so a model is trained on exactly what it will be scored with."""
    bet.setdefault("requested_qty", int(count))
    bet.setdefault("size_bucket", size_bucket(count))
    row = {"time": datetime.now().isoformat(timespec="seconds"), "order_id": oid, "ticker": bet["ticker"], "side": bet["side"],
           "price": bet["price_cents"], "count": count, "yes_bid": bet.get("snap_bid", m.yes_bid),
           "yes_ask": bet.get("snap_ask", m.yes_ask), "book_age_ms": None if bet.get("book_ts") is None else round(1000.0 * (time.time() - bet["book_ts"])),
           "topup_of": bet.get("topup_of", 0), "run_id": RUN_ID, "quote_policy": QUOTE_POLICY, **POLICY,
           "spread": bet.get("snap_ask", m.yes_ask) - bet.get("snap_bid", m.yes_bid), "p_used": round(float(bet.get("p_used") if bet.get("p_used") is not None else bet["win_prob"]), 4),
           "exp_edge_c": round(float(bet.get("expected_edge_c") or 0.0), 3), "p_mkt": round(bet["p_mkt"], 4),
           "act_submit": bet.get("act_submit"), "qty_bid": bet.get("qty_bid"), "qty_ask": bet.get("qty_ask"),
           "stat": bet["stat"], "clock": bet.get("clock", ""), "period": bet.get("period"), "player_id": bet.get("player_id"),
           "arm": bet.get("arm", "base"), "propensity": bet.get("propensity", 1.0), "quote_level": bet.get("quote_level"),
           "pfill": bet.get("pfill"), "lcb_edge_c": bet.get("lcb_edge_c"), "expiration_ts": bet.get("expiration_ts"),
           **{k: bet.get(k) for k in QUOTE_KEYS}}
    try:
        csv_append(QUOTES_CSV, [row])
    except Exception:
        pass


def quote_event(oid: str, bet: dict, event: str, reason: str = "", contracts: float = 0.0):
    """Quote lifecycle: cancel REQUESTS (with reason), fills, and v2026.37 TERMINAL events (state FILLED /
    CANCELLED_CONFIRMED / EXPIRED_CONFIRMED with the proof). Only a terminal event makes an unfilled quote a negative
    label; a quote without one is right-censored."""
    now = time.time()
    sub = float(bet.get("submitted") or now)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "order_id": oid, "ticker": bet.get("ticker"),
           "side": bet.get("side"), "price": bet.get("price_cents"), "event": event, "reason": reason,
           "age_s": round(now - sub, 2), "contracts": contracts, "act_submit": bet.get("act_submit"), "run_id": RUN_ID,
           "state": bet.get("state"), "filled": bet.get("filled", 0.0), "count": bet.get("count"),
           "requested_qty": bet.get("requested_qty", bet.get("count")), "size_bucket": bet.get("size_bucket"),
           "queue_ahead_initial": bet.get("queue_ahead_initial"), "queue_ahead_terminal": bet.get("queue_ahead_terminal"),
           "queue_depletion_rate": bet.get("queue_depletion_rate"), "seconds_at_price": bet.get("seconds_at_price"),
           "price_level_resets": bet.get("price_level_resets"),
           "policy_id": POLICY["policy_id"], "policy_family_id": POLICY["policy_family_id"], "quote_policy": QUOTE_POLICY}
    try:
        csv_append(QUOTE_EVENTS_CSV, [row])
    except Exception:
        pass


def daily_pnl(by_ticker: Dict[str, "KalshiMarket"], last_mid: Dict[str, float], fee_fn=None, liquidation: bool = False) -> float:
    """Today's realized + mark-to-market P&L from the ledger. liquidation=True marks each open position at what it can be
    SOLD for now: the side's bid minus the taker fee (falls back to the last mid only when the market is not in this
    cycle's snapshot). Risk switches use liquidation value: a mid mark on a 6c spread looks several cents per contract
    healthier than the position really is, so the throttle and the stop would engage late."""
    if not os.path.exists(LEDGER_CSV):
        return 0.0
    try:
        df = pd.read_csv(LEDGER_CSV)
    except Exception:
        return 0.0
    pnl = -float(df["cost_usd"].sum())
    for (t, side), c in df.groupby(["ticker", "side"])["contracts"].sum().items():
        if abs(c) <= 1e-9:
            continue
        m = by_ticker.get(t)
        if liquidation and m is not None:
            bid = m.yes_bid if side == "yes" else m.no_bid
            fee = fee_fn(t, bid, False) if (fee_fn and bid >= 1) else 0.0
            pnl += c * max(0.0, bid - fee) / 100.0
            continue
        mid = m.mid if m else last_mid.get(t)
        if mid is not None:
            pnl += c * (mid if side == "yes" else 1.0 - mid)
    return pnl


# Order states (v2026.37). An order's risk is released ONLY by exchange proof of a terminal state.
RESTING, PARTIALLY_FILLED, CANCEL_REQUESTED, CANCEL_FAILED = "RESTING", "PARTIALLY_FILLED", "CANCEL_REQUESTED", "CANCEL_FAILED"
PENDING_TERMINAL, FILLED, CANCELLED_CONFIRMED, EXPIRED_CONFIRMED = "PENDING_TERMINAL", "FILLED", "CANCELLED_CONFIRMED", "EXPIRED_CONFIRMED"
LIVE_STATES = (RESTING, PARTIALLY_FILLED, CANCEL_REQUESTED, CANCEL_FAILED)
TERMINAL_STATES = (FILLED, CANCELLED_CONFIRMED, EXPIRED_CONFIRMED)


def _locked(fn):
    def w(self, *a, **k):
        with self.lock:
            return fn(self, *a, **k)
    w.__name__, w.__doc__ = fn.__name__, fn.__doc__
    return w


class OrderBook:
    """Order and position book-keeping, live and paper on one path, with an explicit order-state machine:
        RESTING / PARTIALLY_FILLED   live on the exchange (self.resting); full remaining risk reserved
        CANCEL_REQUESTED             transient while the DELETE is in flight
        CANCEL_FAILED                DELETE failed: still resting, still reserved, still fast-monitored, retried
        PENDING_TERMINAL             DELETE acknowledged (or exchange expiry passed): reserved until proven terminal
        FILLED / CANCELLED_CONFIRMED / EXPIRED_CONFIRMED   terminal: risk released, logged as a terminal quote event
    Terminal proof (live) is the exchange's own order status (canceled / executed / expired, with its fill count fully
    booked) or, when the status cannot be read, the exchange-enforced expiration_time passed by EXPIRY_GRACE_S with a
    fills read after that time. Elapsed polling cycles are never proof. Late fills after a cancel request still book.
    The journal persists every non-terminal order with its state across crashes. All mutating methods hold self.lock
    (the WebSocket market clock cancels from its own thread)."""

    def __init__(self, kalshi: KalshiClient, positions: Dict[Tuple[str, str], dict],
                 allowed_series: Optional[set] = None, zero_fee: bool = False):
        self.k = kalshi
        self.lock = threading.RLock()
        self.positions = positions
        self.allowed_series = allowed_series if allowed_series is not None else set()
        self.zero_fee = zero_fee
        self.resting: Dict[str, dict] = {}      # order_id -> bet context (state RESTING / PARTIALLY_FILLED / CANCEL_FAILED)
        self.pending: Dict[str, dict] = {}      # order_id -> bet context (state PENDING_TERMINAL)
        self.terminal_log: List[Tuple[str, str]] = []   # (order_id, terminal state) for diagnostics and tests
        self.released: Dict[str, dict] = {}     # terminal orders (kept so a fill reported after release still books)
        self.markouts: List[dict] = []          # entry fills, re-marked 5/15/30/120 s later
        self.qmarks: List[dict] = []            # every posted quote, mid re-marked 30 s after submit (drift counterfactual)
        self.seen_fills: set = set()
        self.session_ts = int(time.time()) - 5
        self.killed: Optional[str] = None       # reason new entries are disabled (exits still run)
        self.cancel_failures = 0
        self.last_fills_read = 0.0
        self.action_of = None                   # game_key -> latest NBA play number seen (set by main; markout event flags)
        if os.path.exists(LEDGER_CSV):           # a restart must never book the same fill twice
            try:
                self.seen_fills = set(pd.read_csv(LEDGER_CSV, usecols=["fill_id"])["fill_id"].dropna().astype(str))
            except Exception:
                pass
        if self.k.can_trade and os.path.exists(JOURNAL_JSON):
            try:
                with open(JOURNAL_JSON) as f:
                    j = json.load(f)
                j = {oid: bet for oid, bet in j.items() if isinstance(bet, dict) and bet.get("ticker") and bet.get("side")}
                booked = {}
                if os.path.exists(LEDGER_CSV):
                    try:
                        _l = pd.read_csv(LEDGER_CSV, usecols=["order_id", "contracts"])
                        _l = _l[pd.to_numeric(_l["contracts"], errors="coerce") > 0]
                        booked = _l.groupby(_l["order_id"].astype(str))["contracts"].sum().to_dict()
                    except Exception:
                        pass
                for oid, bet in j.items():
                    # a crash between booking a fill and saving the journal leaves 'filled' stale: the ledger is the truth
                    bet["filled"] = max(float(bet.get("filled") or 0.0), float(booked.get(str(oid), 0.0)))
                    if not bet.get("expiration_ts"):
                        # pre-v2026.37 journal: every maker order was sent with an exchange expiry <= submit + quote TTL + 3
                        bet["expiration_ts"] = float(bet.get("submitted") or time.time()) + float(bet.get("ttl") or 30) + 3.0
                    if bet.get("state") == PENDING_TERMINAL:
                        self.pending[oid] = bet          # cancel already acknowledged: wait for terminal proof
                    else:
                        # state unknown after a crash: treat as live, cancel it (idempotent), keep its risk reserved
                        bet["state"] = CANCEL_FAILED
                        self.resting[oid] = bet
                if j:
                    self.session_ts = int(min(float(b.get("submitted") or time.time()) for b in j.values())) - 60
                    logger.warning(f"Order journal: {len(j)} unresolved order(s) from the last run reloaded "
                                   f"({len(self.pending)} pending terminal proof, {len(self.resting)} to cancel); their risk "
                                   f"stays reserved until the exchange proves them terminal.")
            except Exception as e:
                logger.error(f"Order journal unreadable ({e}).")

    # ── fills ──────────────────────────────────────────────────────────────────────────────────────────────────────
    @_locked
    def _book_fill(self, bet: dict, order_id: str, fill_id: str, contracts: float, price_cents: float, fee_usd: float, mode: str,
                   fill_ts: Optional[float] = None, ts_source: str = "local"):
        cost = contracts * price_cents / 100.0 + fee_usd          # ledger: cash flow of this fill
        key = (bet["ticker"], bet["side"])
        d = self.positions.setdefault(key, {"contracts": 0.0, "cost_usd": 0.0, "game_key": bet["game_key"],
                                            "player": bet["player_name"], "stat": bet["stat"], "line": float(bet["line"]),
                                            "team_id": bet.get("team_id", ""), "player_id": bet.get("player_id"),
                                            "tier": bet.get("tier", "PRODUCTION")})
        if bet.get("tier") == "PRODUCTION":
            d["tier"] = "PRODUCTION"            # a key with any production fill is production exposure
        d.setdefault("team_id", bet.get("team_id", ""))
        apply_fill(d, contracts, price_cents, fee_usd)
        recv = time.time()
        fill_ts = recv if fill_ts is None else fill_ts
        d["last_fill"] = fill_ts
        d["last_px"] = price_cents
        if contracts > 0:
            act0 = bet.get("act_submit")
            if act0 is None:
                try:
                    act0 = self.action_of(bet["game_key"]) if self.action_of else None
                except Exception:
                    pass
            row = dict(ts=fill_ts, time=datetime.fromtimestamp(fill_ts).isoformat(timespec="seconds"), order_id=order_id, fill_id=fill_id,
                       receipt_ts=round(recv, 3), reporting_delay_ms=round(1000.0 * (recv - fill_ts)), ts_source=ts_source,
                       run_id=RUN_ID, quote_policy=QUOTE_POLICY, **POLICY,
                       ticker=bet["ticker"],
                       game_key=bet["game_key"], side=bet["side"], stat=bet.get("stat"), price=price_cents, contracts=contracts,
                       maker=bet.get("is_maker"), gap=round(abs(bet.get("p_model", 0) - bet.get("p_mkt", 0)), 4),
                       p_used=round(float(bet.get("win_prob", 0)), 4), p_mkt=round(float(bet.get("p_mkt", 0)), 4),
                       spread_c=bet.get("spread_c"), act0=act0, engine=mg.ENGINE_VERSION, mode=mode,
                       arm=bet.get("arm", "base"), propensity=bet.get("propensity", 1.0), quote_level=bet.get("quote_level"),
                       snap_bid=bet.get("snap_bid"), snap_ask=bet.get("snap_ask"), tier=bet.get("tier"),
                       cell=bet.get("cell"), team_id=bet.get("team_id"))
            for h in MARK_H:
                row.update({f"m{h}": None, f"m{h}_age": None, f"m{h}_event": None})
            self.markouts.append(row)
        if d["contracts"] <= 1e-9:
            self.positions.pop(key, None)
        ledger_append({
            "timestamp": datetime.now().isoformat(), "mode": mode, "order_id": order_id, "fill_id": fill_id,
            "ticker": bet["ticker"], "game": bet["game_key"], "player": bet["player_name"], "stat": bet["stat"],
            "line": bet["line"], "side": bet["side"], "order_type": "MAKER" if bet["is_maker"] else "TAKER",
            "price_cents": round(price_cents, 2), "fee_usd": round(fee_usd, 4), "contracts": contracts,
            "cost_usd": round(cost, 4), "p_model": round(bet["p_model"], 4), "p_mkt": round(bet["p_mkt"], 4),
            "p_used": round(float(bet.get("p_used") if bet.get("p_used") is not None else bet["win_prob"]), 4), "ev_pct": round(bet["roi"] * 100.0, 2),
            "joint_kelly_pct": round(bet.get("joint_kelly_fraction", 0.0) * 100.0, 3), "arm": bet.get("arm", "base"),
            "tier": bet.get("tier"), "confidence": bet.get("confidence"), "cell": bet.get("cell"), "team_id": bet.get("team_id"),
            "fill_prob": bet.get("fill_prob"), "cond_edge_c": bet.get("cond_edge_c"),
            "requested_qty": bet.get("requested_qty", bet.get("count", abs(contracts))),
            "size_bucket": bet.get("size_bucket", size_bucket(max(1, int(abs(contracts) or 1))))})
        logger.warning(f"✅ FILL {mode} {bet['player_name']} {bet['stat']} {bet['side'].upper()} {bet['line']} "
                       f"{contracts:g}x @ {price_cents:.0f}¢ (cost ${cost:.2f}) {bet['ticker']} [{bet.get('tier', '?')}]")

    # ── state transitions ─────────────────────────────────────────────────────────────────────────────────────────
    def _terminal(self, oid: str, bet: dict, state: str, proof: str):
        _ws = getattr(self.k, "ws", None)
        if _ws is not None and hasattr(_ws, "stop_queue_estimate"):
            _q = _ws.stop_queue_estimate(oid)
            if _q:
                bet["queue_ahead_initial"] = _q.get("queue_ahead_initial")
                bet["queue_ahead_terminal"] = _q.get("queue_ahead_est")
                bet["queue_depletion_rate"] = _q.get("queue_depletion_rate")
                bet["seconds_at_price"] = _q.get("seconds_at_price")
                bet["price_level_resets"] = _q.get("price_level_resets")
        bet["state"] = state
        self.resting.pop(oid, None)
        self.pending.pop(oid, None)
        self.terminal_log.append((oid, state))
        bet["released_ts"] = time.time()
        self.released[oid] = bet
        if len(self.released) > 2000:
            self.released.pop(next(iter(self.released)))
        quote_event(oid, bet, "terminal", f"{state}: {proof}", float(bet.get("filled", 0.0)))

    @_locked
    def request_cancel(self, oid: str, reason: str) -> bool:
        """Ask the exchange to cancel a live order. Success -> PENDING_TERMINAL (still reserved). Failure ->
        CANCEL_FAILED: stays live in self.resting, keeps its full reservation, is retried and fast-monitored."""
        bet = self.resting.get(oid)
        if bet is None:
            return oid in self.pending
        if bet.get("state") == CANCEL_FAILED and time.time() < float(bet.get("next_retry_ts", 0.0)):
            return False                         # backing off: a WS burst must not hammer DELETE (review fix)
        if bet.get("state") != CANCEL_FAILED:
            quote_event(oid, bet, "cancel", reason)
        bet["state"], bet["cancel_reason"] = CANCEL_REQUESTED, reason
        if not self.k.can_trade:
            # Paper: a cancel is instant and a cancelled paper quote can never fill late
            self._terminal(oid, bet, CANCELLED_CONFIRMED, "paper cancel")
            return True
        ok = self.k.cancel_order(oid, bet["ticker"])
        if ok:
            bet["state"], bet["cancel_ts"] = PENDING_TERMINAL, time.time()
            self.pending[oid] = self.resting.pop(oid)
        else:
            bet["state"] = CANCEL_FAILED
            bet["cancel_fail_n"] = int(bet.get("cancel_fail_n", 0)) + 1
            bet["next_retry_ts"] = time.time() + min(30.0, 2.0 ** bet["cancel_fail_n"])
            if bet["cancel_fail_n"] == 1:
                self.cancel_failures += 1        # kill switch counts ORDERS that failed, not retries of one order
            elif bet["cancel_fail_n"] >= 6 and not self.killed:
                self.killed = f"order {oid} could not be cancelled after {bet['cancel_fail_n']} attempts"
            logger.error(f"[CANCEL FAILED] {oid} {bet['ticker']} ({reason}): still resting, risk still reserved; will retry")
        return ok

    def _cancel_resting(self, oid: str, reason: str):
        self.request_cancel(oid, reason)

    @_locked
    def retry_failed_cancels(self) -> int:
        n = 0
        for oid, bet in list(self.resting.items()):
            if bet.get("state") == CANCEL_FAILED and time.time() >= float(bet.get("next_retry_ts", 0.0)):
                n += 1
                self.request_cancel(oid, bet.get("cancel_reason") or "retry")
        return n

    def reconcile(self, markets_by_ticker: Dict[str, KalshiMarket], cancel_all: bool = True):
        """Book every fill (incl. late fills on cancel-requested orders and on orders already released), move
        exchange-expired orders to PENDING_TERMINAL, and release risk only on terminal proof. cancel_all: cancel
        everything live first. Exchange reads (fills, order status) happen OUTSIDE the book lock, so the WebSocket market
        clock is never stalled behind a slow REST call; every state change happens under it."""
        now = time.time()
        if not self.k.can_trade:
            with self.lock:
                self._reconcile_paper(markets_by_ticker, cancel_all, now)
            return
        with self.lock:
            recent = any(now - float(b.get("released_ts", 0.0)) < 600.0 for b in self.released.values())
            if not self.resting and not self.pending and not recent:
                return                           # nothing live, pending or released in the last 10 min: no fills to look for
            if cancel_all:
                for oid in list(self.resting):
                    self.request_cancel(oid, "shutdown")
            self.retry_failed_cancels()
            for oid, bet in list(self.resting.items()):
                if bet.get("expiration_ts") and now > float(bet["expiration_ts"]) and bet.get("state") in (RESTING, PARTIALLY_FILLED):
                    bet["state"] = PENDING_TERMINAL
                    self.pending[oid] = self.resting.pop(oid)
            session_ts = self.session_ts
        fills = self.k.get_fills(session_ts)                       # network, outside the lock
        t_fills = time.time()
        with self.lock:
            fills_ok = fills is not None
            if fills_ok:
                self.last_fills_read = t_fills
            for f in fills or []:
                oid, fid = str(f.get("order_id", "")), str(f.get("fill_id") or f.get("trade_id") or "")
                if fid in self.seen_fills:
                    continue
                bet = self.pending.get(oid) or self.resting.get(oid) or self.released.get(oid)
                if bet is None:
                    continue
                self.seen_fills.add(fid)
                n_f = float(f.get("count_fp", 0) or 0)
                if oid in self.released:
                    logger.error(f"[LATE FILL] {oid} {bet['ticker']}: {n_f:g} contract(s) filled after the order was released; booked now")
                bet["filled"] = float(bet.get("filled", 0.0)) + n_f
                quote_event(oid, bet, "fill", "late (after cancel request)" if oid in self.pending else
                            ("after release" if oid in self.released else ""), n_f)
                px = float(f.get("yes_price_dollars" if bet["side"] == "yes" else "no_price_dollars") or 0) * 100.0
                fee_paid = float(f.get("fee_cost", 0) or 0)
                _fts, _src = venue_fill_ts(f, time.time())
                self._book_fill(bet, oid, fid, n_f, px or bet["price_cents"], fee_paid, "LIVE", fill_ts=_fts, ts_source=_src)
                series = bet["ticker"].split("-")[0]
                if self.zero_fee and fee_paid > 0 and series in self.allowed_series:
                    self.allowed_series.discard(series)
                    logger.error(f"FEE TRIPWIRE: fill on {bet['ticker']} charged ${fee_paid:.4f}; "
                                 f"{series} removed from tradeable series for this session.")
                if oid in self.resting:
                    if bet["filled"] >= float(bet.get("count", 0)) - 1e-9:
                        self._terminal(oid, bet, FILLED, "fully filled (fills feed)")
                    elif bet.get("state") == RESTING:
                        bet["state"] = PARTIALLY_FILLED
            to_check = []
            for oid, bet in list(self.pending.items()):
                if float(bet.get("filled", 0.0)) >= float(bet.get("count", 0)) - 1e-9:
                    self._terminal(oid, bet, FILLED, "fully filled")
                else:
                    to_check.append(oid)
        statuses = {oid: self.k.get_order_status(oid) for oid in to_check}    # network, outside the lock
        with self.lock:
            now = time.time()
            for oid in to_check:
                bet = self.pending.get(oid)
                if bet is None:
                    continue
                st = statuses.get(oid)
                if st is not None:
                    s_ = st["status"]
                    if s_ in ("canceled", "cancelled", "executed", "expired"):
                        fc = st.get("fill_count")
                        if fc is not None and float(bet.get("filled", 0.0)) >= fc - 1e-9:
                            self._terminal(oid, bet, {"executed": FILLED, "expired": EXPIRED_CONFIRMED}.get(s_, CANCELLED_CONFIRMED),
                                           f"exchange status {s_}, fill count {fc}")
                            continue
                        # fill count unknown, or fills not booked yet: not proof (review fix); keep reserved, try expiry below
                        bet["await_fills"] = fc
                    elif s_ == "resting":
                        bet["state"] = CANCEL_FAILED      # the DELETE did not take: live again, retried, reserved
                        self.resting[oid] = self.pending.pop(oid)
                        continue
                exp = bet.get("expiration_ts")
                if exp and fills_ok and now > float(exp) + EXPIRY_GRACE_S and self.last_fills_read > float(exp) + EXPIRY_GRACE_S:
                    fc = (st or {}).get("fill_count")
                    if fc is not None and float(bet.get("filled", 0.0)) < fc - 1e-9:
                        if now - float(exp) > 600:
                            logger.error(f"[ORDER {oid}] exchange reports {fc:g} filled, {bet.get('filled', 0):g} booked, 10+ min "
                                         f"after expiry: still reserved; check the Kalshi fills page")
                        continue
                    self._terminal(oid, bet, EXPIRED_CONFIRMED,
                                   f"exchange expiry {datetime.fromtimestamp(float(exp)).isoformat(timespec='seconds')} + {EXPIRY_GRACE_S:.0f}s, full fills read after")
            if self.cancel_failures >= 3 and not self.killed:
                self.killed = f"cancel failures on {self.cancel_failures} orders"
            self.save_journal()

    def _reconcile_paper(self, markets_by_ticker, cancel_all, now):
        """Paper: a resting bid counts as filled only if the market has since traded through it; expiry is instant."""
        for oid, bet in list(self.resting.items()):
            m = markets_by_ticker.get(bet["ticker"])
            if m is not None and bet.get("state") in (RESTING, PARTIALLY_FILLED):
                ask_now = m.yes_ask if bet["side"] == "yes" else m.no_ask
                if ask_now <= bet["price_cents"]:
                    fee = self.k.fee_cents(bet["ticker"], bet["price_cents"], True) / 100.0 * bet["count"]
                    quote_event(oid, bet, "fill", "paper: traded through", float(bet["count"]))
                    bet["filled"] = float(bet["count"])
                    self._book_fill(bet, oid, oid, float(bet["count"]), bet["price_cents"], fee, "PAPER", ts_source="paper")
                    self._terminal(oid, bet, FILLED, "paper trade-through")
                    continue
            if bet.get("expiration_ts") and now > float(bet["expiration_ts"]):
                self._terminal(oid, bet, EXPIRED_CONFIRMED, "paper expiry")
        if cancel_all:
            for oid in list(self.resting):
                self.request_cancel(oid, "shutdown")
        for oid, bet in list(self.pending.items()):
            self._terminal(oid, bet, CANCELLED_CONFIRMED, "paper")

    @_locked
    def save_journal(self):
        if not self.k.can_trade:
            return
        keep = ("ticker", "side", "count", "filled", "price_cents", "fee_cents", "game_key", "player_name", "player_id",
                "stat", "line", "is_maker", "submitted", "p_model", "p_mkt", "win_prob", "p_used", "roi", "spread_c",
                "expected_edge_c", "act_submit", "arm", "propensity", "quote_level", "ttl", "snap_bid", "snap_ask",
                "state", "cancel_ts", "cancel_reason", "expiration_ts", "tier", "team_id", "confidence", "cell")
        j = {oid: {k: (float(b[k]) if isinstance(b.get(k), (np.floating, np.integer)) else b.get(k)) for k in keep}
             for oid, b in list(self.resting.items()) + list(self.pending.items())}
        try:
            with open(JOURNAL_JSON + ".tmp", "w") as f:
                json.dump(j, f)
            os.replace(JOURNAL_JSON + ".tmp", JOURNAL_JSON)
        except Exception as e:
            logger.error(f"Order journal write failed: {e}")

    @_locked
    def reserved(self, include_resting: bool = True, only_resting: bool = False, states: Optional[tuple] = None) -> Dict[Tuple[str, str], dict]:
        """Full possible cost AND quantity of every order not proven terminal (resting, cancel-failed, pending).
        states: keep only orders in these states."""
        out: Dict[Tuple[str, str], dict] = {}
        src = list(self.resting.values()) if only_resting else \
            (list(self.resting.values()) if include_resting else []) + list(self.pending.values())
        if states is not None:
            src = [b for b in src if b.get("state") in states]
        for b in src:
            rem = max(0.0, float(b.get("count", 0)) - float(b.get("filled", 0.0)))
            if rem <= 0:
                continue
            d = out.setdefault((b["ticker"], b["side"]), {"contracts": 0.0, "cost_usd": 0.0, "game_key": b.get("game_key", ""),
                                                         "player": b.get("player_name", ""), "stat": b.get("stat"), "line": b.get("line"),
                                                         "team_id": b.get("team_id", ""), "tier": b.get("tier", "PRODUCTION")})
            d["contracts"] += rem
            d["cost_usd"] += rem * (float(b.get("price_cents", 0)) + float(b.get("fee_cents", 0))) / 100.0
            if b.get("tier") == "PRODUCTION":
                d["tier"] = "PRODUCTION"
        return out

    # ── markouts ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _mark(self, r: dict, yes_mid_c: float, now: float) -> bool:
        """Record every horizon this fill has reached. m{h} = side mid minus fill price (cents; negative = picked off),
        m{h}_age = actual seconds at the mark, m{h}_event = an NBA play happened between fill and mark."""
        mid = yes_mid_c if r["side"] == "yes" else 100.0 - yes_mid_c
        act = None
        try:
            act = self.action_of(r.get("game_key", "")) if self.action_of else None
        except Exception:
            pass
        changed = False
        for h in MARK_H:
            age = now - r["ts"]
            if r[f"m{h}"] is None and age >= h:
                r[f"m{h}"], r[f"m{h}_age"] = round(mid - r["price"], 2), round(age, 1)
                r[f"m{h}_event"] = None if (act is None or r.get("act0") is None) else bool(act > r["act0"])
                changed = True
        return changed

    def _qmark(self, r: dict, yes_mid_c: float, now: float) -> bool:
        """Quote-level counterfactual: side mid 30 s after SUBMIT minus side mid at submit, for every posted quote."""
        if r.get("q_m30") is not None or now - r["ts"] < 30.0:
            return False
        mid = yes_mid_c if r["side"] == "yes" else 100.0 - yes_mid_c
        r["q_m30"], r["q_m30_age"] = round(mid - r["mid0"], 2), round(now - r["ts"], 1)
        try:
            act = self.action_of(r.get("game_key", "")) if self.action_of else None
            r["q_m30_event"] = None if (act is None or r.get("act0") is None) else bool(act > r["act0"])
        except Exception:
            r["q_m30_event"] = None
        return True

    def toxicity(self, n: int = 8) -> Optional[float]:
        """Contract-weighted markout of the last n entry fills (30s mark, else 15s). None until 3 fills are marked."""
        rows = []
        for r in self.markouts[::-1]:
            for h in (30, 15):
                v = r.get(f"m{h}")
                if v is not None:
                    rows.append((float(v), float(r.get("contracts") or 1.0)))
                    break
            if len(rows) >= n:
                break
        if len(rows) < 3:
            return None
        return sum(v * w for v, w in rows) / sum(w for _, w in rows)

    def _write_markouts(self):
        if self.markouts:
            pd.DataFrame(self.markouts).drop(columns=["ts"]).to_csv(MARKOUT_CSV, index=False)
        if self.qmarks:
            pd.DataFrame(self.qmarks).drop(columns=["ts"]).to_csv(QUOTE_MARKS_CSV, index=False)

    @_locked
    def update_markouts(self, by_ticker: Dict[str, "KalshiMarket"]):
        now, changed = time.time(), False
        for r in self.markouts:
            m = by_ticker.get(r["ticker"])
            if m is not None:
                changed |= self._mark(r, m.mid * 100.0, now)
        for r in self.qmarks:
            m = by_ticker.get(r["ticker"])
            if m is not None:
                changed |= self._qmark(r, m.mid * 100.0, now)
        if changed:
            self._write_markouts()

    def fresh_markouts(self) -> List[dict]:
        now = time.time()
        return [r for r in self.markouts if now - r["ts"] < MARK_H[-2] + 10 and any(r[f"m{h}"] is None for h in MARK_H[:-1])] + \
               [r for r in self.qmarks if r.get("q_m30") is None and now - r["ts"] < 45.0]

    @_locked
    def mark_fresh(self, tickers: Optional[set] = None):
        """Between cycles (and on WS updates): mark recent fills at 5s/15s/30s and quotes at 30s from the live book."""
        now, changed = time.time(), False
        for r in self.fresh_markouts():
            if tickers is not None and r["ticker"] not in tickers:
                continue
            due = ("q_m30" in r and now - r["ts"] >= 30.0) or any(r.get(f"m{h}") is None and now - r["ts"] >= h for h in MARK_H if f"m{h}" in r)
            if not due:
                continue
            tb = self.k.top_of_book(r["ticker"])
            if tb is not None:
                mid = (tb[0] + tb[2]) / 2.0
                changed |= self._qmark(r, mid, now) if "q_m30" in r else self._mark(r, mid, now)
        if changed:
            self._write_markouts()

    # ── quoting ────────────────────────────────────────────────────────────────────────────────────────────────────
    @_locked
    def sync_quotes(self, desired: List[dict]) -> Tuple[List[dict], int]:
        """Quantity-aware reconciliation of LIVE quotes with the TOTAL target per (ticker, side, price):
          target == resting remaining -> keep (queue place kept);
          target  > resting remaining -> keep the resting part, post only the difference;
          target  < resting remaining -> cancel (oldest kept first while it fits) and post the shortfall;
          no target                   -> cancel.
        CANCEL_FAILED orders are never 'kept' as quotes (their cancel is retried) but stay reserved."""
        want = {(o["ticker"], o["side"], int(round(o["price_cents"]))): dict(o) for o in desired}
        kept, now = 0, time.time()
        by_key: Dict[tuple, list] = {}
        for oid, bet in list(self.resting.items()):
            if bet.get("state") == CANCEL_FAILED:
                continue
            key = (bet["ticker"], bet["side"], int(round(bet["price_cents"])))
            stale = now - float(bet.get("submitted") or now) > float(bet.get("ttl") or QUOTE_MAX_AGE)   # state-dependent life
            if stale or key not in want:
                self.request_cancel(oid, "expired_refresh" if stale else "no_longer_wanted")
            else:
                by_key.setdefault(key, []).append(oid)
        out = []
        for key, o in want.items():
            target, have = int(o["contracts"]), 0
            for oid in sorted(by_key.get(key, []), key=lambda x: float(self.resting[x].get("submitted") or 0)):
                rem = max(0, int(round(float(self.resting[oid].get("count", 0)) - float(self.resting[oid].get("filled", 0.0)))))
                if have + rem <= target:
                    have += rem
                    kept += 1
                else:
                    self.request_cancel(oid, "shrink_to_target")
            if target - have >= 1:
                o["contracts"] = target - have
                o["topup_of"] = have
                out.append(o)
        self.save_journal()
        return out, kept

    @_locked
    def fast_checks(self, jump_c: float = 1.0, depth_frac: float = 0.25, unreadable_limit: int = 3,
                    tickers: Optional[set] = None) -> Dict[str, int]:
        """Pull a live quote when the OTHER traders' book changed under it (our own resting size is removed first).
        Called every --watch tick and, with --ws, from the market clock on every synchronized book update (tickers =
        the updated market). CANCEL_FAILED orders are checked too (and their cancel retried). Rules, on the quote's own
        side (NO mapped through 100 - YES):
          market_jump     the opposite (ask-side) best moved >= jump_c, or another trader now bids above our price
          depth_collapse  a JOINED quote whose other-trader depth at its price fell below depth_frac of what it was
          book_unreadable no readable book for unreadable_limit consecutive checks"""
        out: Dict[str, int] = {}
        for oid, bet in list(self.resting.items()):
            if tickers is not None and bet["ticker"] not in tickers:
                continue
            if bet.get("state") == CANCEL_FAILED:
                self.request_cancel(oid, bet.get("cancel_reason") or "retry")      # honours the retry backoff
                continue
            tb = self.k.top_of_book(bet["ticker"])
            why = None
            if tb is None:
                bet["_unread"] = int(bet.get("_unread", 0)) + 1
                if bet["_unread"] >= unreadable_limit:
                    why = "book_unreadable"
            else:
                bet["_unread"] = 0
                yb, qb, ya, qa = tb
                sb, sa = bet.get("snap_bid"), bet.get("snap_ask")
                if bet["side"] == "yes":
                    bid, qbid, ask = yb, qb, ya
                    s_bid, s_ask, q0 = sb, sa, bet.get("qty_bid")
                else:
                    bid, qbid, ask = 100.0 - ya, qa, 100.0 - yb
                    s_bid = None if sa is None else 100.0 - sa
                    s_ask = None if sb is None else 100.0 - sb
                    q0 = bet.get("qty_ask")
                px = float(bet["price_cents"])
                own = max(0.0, float(bet.get("count", 0)) - float(bet.get("filled", 0.0)))
                other_at_bid = qbid - own if abs(bid - px) < 1e-9 else qbid
                if s_ask is not None and abs(ask - s_ask) >= jump_c - 1e-9:
                    why = "market_jump"
                elif bid > px + 1e-9:
                    why = "market_jump"
                elif s_bid is not None and abs(px - s_bid) < 1e-9 and q0 and float(q0) >= 20 and other_at_bid < depth_frac * float(q0):
                    why = "depth_collapse"
            if why:
                self.request_cancel(oid, why)
                out[why] = out.get(why, 0) + 1
        return out

    @_locked
    def cancel_game(self, game_key: str, reason: str = "new_play") -> int:
        """Any new play changes every price in that game: request cancellation of all its live quotes."""
        n = 0
        for oid, bet in list(self.resting.items()):
            if bet.get("game_key") == game_key:
                self.request_cancel(oid, reason)
                n += 1
        return n

    @_locked
    def cancel_players(self, pids: set, reason: str = "player_in_play") -> int:
        n = 0
        for oid, bet in list(self.resting.items()):
            if bet.get("player_id") in pids:
                self.request_cancel(oid, reason)
                n += 1
        return n

    def exit(self, bet: dict, market: KalshiMarket, count: int) -> float:
        """Sell `count` held contracts at the current bid. Returns $ proceeds after fee."""
        px = bet["price_cents"]
        if self.k.can_trade:
            oid, filled = self.k.send_exit(market, bet["side"], px, count)
            if not oid or filled <= 0:
                logger.info(f"[EXIT] no fill at {px}¢ on {bet['ticker']}")
                return 0.0
        else:
            oid, filled = f"paper-exit-{uuid.uuid4().hex[:8]}", float(count)
        fee = self.k.fee_cents(bet["ticker"], px, False) / 100.0 * filled
        self._book_fill(bet, oid, f"{oid}-exit", -filled, px, fee, "LIVE" if self.k.can_trade else "PAPER")
        return filled * px / 100.0 - fee

    @_locked
    def place(self, bet: dict, market: KalshiMarket, count: int, ttl_sec: int) -> float:
        """Returns $ committed (filled cost or resting notional). Maker orders carry their exchange expiration_ts."""
        notional = count * (bet["price_cents"] + bet["fee_cents"]) / 100.0
        now = time.time()
        live = self.k.can_trade
        if live:
            oid, filled = self.k.send_order(market, bet["side"], bet["price_cents"], count, bet["is_maker"], ttl_sec)
            if not oid:
                return 0.0
        else:
            oid, filled = f"paper-{uuid.uuid4().hex[:10]}", 0.0
        if bet["is_maker"]:
            b = dict(bet, count=count, requested_qty=int(count), size_bucket=size_bucket(count),
                     submitted=now, filled=0.0, state=RESTING, expiration_ts=int(now) + int(ttl_sec))
            _ws = getattr(self.k, "ws", None)
            if _ws is not None and hasattr(_ws, "start_queue_estimate"):
                _initial = float(bet.get("queue_ahead") or 0.0)
                _q = _ws.start_queue_estimate(oid, bet["ticker"], bet["side"], int(bet["price_cents"]), initial_depth=_initial)
                b["queue_ahead_initial"] = _q.get("queue_ahead_initial")
            self.resting[oid] = b
            self.save_journal()
            quote_log(oid, b, market, count)
            yb, ya = bet.get("snap_bid", market.yes_bid), bet.get("snap_ask", market.yes_ask)
            mid0 = (yb + ya) / 2.0 if bet["side"] == "yes" else 100.0 - (yb + ya) / 2.0
            self.qmarks.append(dict(ts=now, order_id=oid, ticker=bet["ticker"], side=bet["side"], game_key=bet["game_key"],
                                    price=bet["price_cents"], mid0=mid0, act0=bet.get("act_submit"), q_m30=None,
                                    q_m30_age=None, q_m30_event=None, run_id=RUN_ID, quote_policy=QUOTE_POLICY, **POLICY))
            return notional
        if live:
            if filled > 0:
                fee = self.k.fee_cents(bet["ticker"], bet["price_cents"], False) / 100.0 * filled
                self._book_fill(bet, oid, f"{oid}-ioc", filled, bet["price_cents"], fee, "LIVE")
                return notional
            logger.info(f"[IOC] no fill at {bet['price_cents']}¢ on {bet['ticker']}")
            return 0.0
        fee = self.k.fee_cents(bet["ticker"], bet["price_cents"], False) / 100.0 * count
        self._book_fill(bet, oid, oid, float(count), bet["price_cents"], fee, "PAPER")
        return notional


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Kalshi NBA In-Play Trader")
    ap.add_argument("--bankroll", type=float, default=250.0, help="Risk capital for the night in USD (Kelly base)")
    ap.add_argument("--max-bet", type=float, default=5.0, help="Max $ per order")
    ap.add_argument("--min-order", type=float, default=2.0,
                    help="Skip orders/top-ups smaller than this many $ (avoids paying rounded fees on dust)")
    ap.add_argument("--min-ev", type=float, default=1.0,
                    help="Min ROI on all-in cost after every haircut, percent (v2026.28: 3 on the nominal edge)")
    ap.add_argument("--min-edge-c", type=float, default=0.5,   # applied to the 90% LCB (plus inventory skew)
                    help="Min ROBUST edge in cents (after no spread credit, fee, MC, calibration uncertainty, buffer). v2026.28 "
                         "used 2c on the nominal edge as a proxy for fill costs, which are now subtracted explicitly")
    ap.add_argument("--model-weight", type=float, default=0.0,
                    help="RESEARCH ONLY when given: logit-pool weight on the model. Default 0 = no directional trading until "
                         "calibration.json (current engine, trade regime, out of sample) earns a weight. Paper only unless "
                         "--override-model-weight-risk")
    ap.add_argument("--override-model-weight-risk", action="store_true",
                    help="Allow a manual --model-weight with --live (bypasses the evidence hierarchy; not recommended)")
    ap.add_argument("--allow-no", action="store_true",
                    help="RESEARCH ONLY: allow NO (under) maker entries (off until exec_priors.json shows the NO side passing live)")
    ap.add_argument("--exec-alpha-no-sd-c", type=float, default=None,
                    help="S.e. of the NO execution prior (default: hierarchical qp5 prior, 3.0c until qp5 marks exist)")
    ap.add_argument("--exec-alpha-yes-sd-c", type=float, default=None,
                    help="S.e. of the YES execution prior (default: hierarchical qp5 prior, 2.0c until qp5 marks exist)")
    ap.add_argument("--cvar-limit", type=float, default=0.02, help="Tail limit: scenario CVaR95 of the night's portfolio <= this x bankroll")
    ap.add_argument("--inv-skew-c", type=float, default=1.0, help="Extra LCB edge (cents) required at the player cap: x (exposure/cap)^2")
    ap.add_argument("--exec-alpha-no-c", type=float, default=None,
                    help="Expected 30s markout of a NO maker fill, cents (default: hierarchical qp5 prior; qp3/Nov 6 was -2.3c)")
    ap.add_argument("--exec-alpha-yes-c", type=float, default=None, help="Same for YES (default: hierarchical qp5 prior)")
    ap.add_argument("--tox-n", type=int, default=8, help="Execution-health breaker: number of recent fills in the rolling markout")
    ap.add_argument("--tox-warn-c", type=float, default=-3.0, help="Rolling markout at/below this: half size, join the bid")
    ap.add_argument("--tox-stop-c", type=float, default=-6.0, help="Rolling markout at/below this: no new entries (10 min)")
    ap.add_argument("--max-disagree", type=float, default=0.10, help="Skip if |p_model - p_mkt| exceeds this (Nov 5: 0 edge above 0.10)")
    ap.add_argument("--min-gap", type=float, default=0.01,
                    help="Skip if |p_model - p_mkt| is below this. v2026.35: 0.01 (was 0.03): the 90%% edge LCB already charges "
                         "MC, calibration, weight, execution and shadow uncertainty, so a hard 3c floor double-filtered")
    ap.add_argument("--stats", default="PTS,REB,AST", help="Stats to trade (Nov 5: 3PM model added nothing, PRA biased high)")
    ap.add_argument("--min-period", type=int, default=None,
                    help="First quarter to trade. Default: Q3 until calibration.json has fitted period weights, then Q1 "
                         "(weak periods then get small model weights instead of a ban)")
    ap.add_argument("--taker", action="store_true", help="Allow crossing the spread (Nov 5: takers lost $74.62 of $75.68)")
    ap.add_argument("--watch", type=float, default=2.0, help="Seconds between play-by-play checks (event-driven cancels)")
    ap.add_argument("--live-ev-premium", type=float, default=2.0, help="Extra min ROI (pct) for new orders while the ball is live")
    ap.add_argument("--exit-margin", type=float, default=0.03, help="Sell when bid minus fee beats model value by this much")
    ap.add_argument("--no-exits", action="store_true", help="Hold every position to settlement")
    ap.add_argument("--refill-cooldown", type=float, default=180.0, help="Seconds before adding to a position again")
    ap.add_argument("--max-spread", type=int, default=6, help="Max yes spread in cents")
    ap.add_argument("--price-min", type=int, default=20, help="Min price paid in cents (longshot filter)")
    ap.add_argument("--price-max", type=int, default=80, help="Max price paid in cents")
    ap.add_argument("--tail-min", type=int, default=10, help="Tail zone floor: prices tail-min..price-min need the extra bar")
    ap.add_argument("--tail-max", type=int, default=90, help="Tail zone ceiling: prices price-max..tail-max need the extra bar")
    ap.add_argument("--tail-extra-c", type=float, default=1.0,
                    help="Extra LCB (cents) required in the 10-20c / 80-90c tails, where the model was overconfident before")
    ap.add_argument("--quote-ladder", choices=["mid", "bid1"], default="bid1",
                    help="bid1 (default): base quote bid+1, exploration arms +/-1c, or the fitted execution model's argmax. "
                         "mid = v2026.35 midpoint walk: RESEARCH ONLY (paper), since edge-at-mid ignores fill selection")
    ap.add_argument("--no-probe", type=int, default=0,
                    help="LIVE NO probes: up to N one-contract NO orders per run that JOIN the bid, so the NO side can re-earn "
                         "its place with real fills (0 = off; NO otherwise stays disabled live)")
    ap.add_argument("--max-mid-move", type=float, default=0.03, help="Skip a market whose mid moved more than this since last cycle")
    ap.add_argument("--session-cap", type=float, default=0.10, help="Max total open risk / bankroll for the night")
    ap.add_argument("--game-cap", type=float, default=0.03)
    ap.add_argument("--single-cap", type=float, default=0.01)
    ap.add_argument("--player-cap", type=float, default=0.015, help="Max total exposure / bankroll on one player")
    ap.add_argument("--fee-policy", choices=["min", "zero"], default="min",
                    help="min: maker-first, taker only when the edge pays for the fee; zero: fee-free maker only")
    ap.add_argument("--taker-min-ev", type=float, default=10.0,
                    help="Min ROI (percent, after taker fee) to cross the spread before the last 300s of Q4/OT")
    ap.add_argument("--no-fav-taper", action="store_true", help="Disable the favorite position-cap taper")
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--sims", type=int, default=8000, help="Paths per game state (states are cached; same state = same paths)")
    ap.add_argument("--kelly", type=float, default=0.10, help="Base Kelly fraction; LIVE is hard-capped at 0.10 in v2026.39")
    ap.add_argument("--adaptive-kelly", action="store_true", help="Optional: multiply base Kelly by prediction/execution health in [0,1]; never exceeds base")
    ap.add_argument("--as-buffer-c", type=float, default=0.5, help="Adverse-selection buffer, cents, added to the edge bar")
    ap.add_argument("--as-spread-frac", type=float, default=1.0,
                    help="Share of the spread captured by a resting bid that is assumed to be given back on fills "
                         "(0 = fills are random [v2026.28], 1 = count no spread as edge). grade_predictions.py estimates it from markouts")
    ap.add_argument("--shadow-z", type=float, default=1.0,
                    help="Specification haircut on main-vs-shadow disagreement beyond Monte Carlo noise, applied automatically "
                         "whenever the shadow is cached for the exact state (it runs after orders, never on the execution path)")
    ap.add_argument("--lcb-z", type=float, default=1.282, help="Edge rule: Edge_mean - penalties - z x sigma_measured - H_shadow (1.282 = 90%% LCB)")
    ap.add_argument("--model-weight-sd", type=float, default=0.15,
                    help="Pool-weight s.d. when calibration.json has none (one graded night: 0.31-0.34 in sample, mixed held out)")
    ap.add_argument("--matchup-strength", type=float, default=0.0,
                    help="Extra matchup layer on D&T game-optimized skills, 0..1 (0 = D&T only until the A/B earns it weight)")
    ap.add_argument("--quote-ttl", type=int, default=30,
                    help="Seconds a resting quote lives on the exchange. Quotes are posted in dead balls; a short life limits the "
                         "time a quote sits after play resumes and before the (lagging) NBA feed shows it")
    ap.add_argument("--no-revalidate", action="store_true", help="Skip the pre-send order-book and play-by-play re-check (not recommended)")
    ap.add_argument("--revalidate-mid-c", type=float, default=1.0, help="Pre-send check: max mid move (cents) since the priced snapshot")
    ap.add_argument("--model-risk-sd", type=float, default=0.01, help="Sizing: shared model-error s.d. (probability) across ALL bets")
    ap.add_argument("--stat-risk-sd", type=float, default=0.01, help="Sizing: extra shared model-error s.d. within one stat")
    ap.add_argument("--game-risk-sd", type=float, default=0.01, help="Sizing: extra shared model-error s.d. within one game")
    ap.add_argument("--min-books", type=int, default=2, help="Sportsbook gate needs at least this many fresh books")
    ap.add_argument("--book-max-age", type=float, default=90.0, help="Ignore a book whose market update is older than this (s)")
    ap.add_argument("--daily-loss", type=float, default=0.03, help="Kill switch: stop new entries at this loss / bankroll (realized + MTM)")
    ap.add_argument("--shadow-sims", type=int, default=2000, help="Paths for the untraded no-matchup-layer shadow engine (0 = off)")
    ap.add_argument("--book-max-line-diff", type=float, default=0.0,
                    help="Sportsbook gate only on book lines this close to the Kalshi line (0 = exact; translated lines are logged only)")
    ap.add_argument("--books-interval", type=float, default=90.0, help="Seconds between sportsbook refreshes per game (credits!)")
    ap.add_argument("--no-books", action="store_true", help="Ignore sportsbook consensus even if ODDS_API_KEY is set")
    ap.add_argument("--book-weight", type=float, default=0.0,
                    help="Logit-pool weight on sportsbook consensus (0 = confirmation gate + logging only until fitted)")
    ap.add_argument("--micro", action="store_true",
                    help="Fetch order-book depth for microprice inside the pricing loop (off: one REST call per market adds latency)")
    ap.add_argument("--no-micro", action="store_true", help=argparse.SUPPRESS)   # v2026.28 flag, now the default
    ap.add_argument("--allow-live-ball", action="store_true",
                    help="Allow NEW orders while the ball is live (default: dead balls only, until residual possessions are modelled)")
    ap.add_argument("--pregame", action="store_true", help="Also trade scheduled (not started) games")
    ap.add_argument("--min-minutes", type=float, default=20.0, help="Trade players projected for at least this many minutes")
    ap.add_argument("--once", action="store_true", help="Run a single cycle and exit (dry-run preview)")
    ap.add_argument("--live", action="store_true", help="Submit real-money orders (default: paper)")
    ap.add_argument("--override-gate", action="store_true",
                    help="(v2026.36: no longer authorizes production; an unvalidated live run must be --pilot-live)")
    ap.add_argument("--pilot-live", action="store_true",
                    help="PILOT LIVE (real money, data collection): allowed without a passed gate / earned weight, but forced to "
                         "1 contract per order, 1%% session risk, 0.25%% player risk, YES only, no taker, Kelly <= 0.10")
    ap.add_argument("--write-production-profile", action="store_true",
                    help="After a validated start, save this configuration as production_profile.json (later production runs "
                         "must match it)")
    ap.add_argument("--kelly-mean", choices=["auto", "lcb", "posterior"], default="auto",
                    help="Expected return fed to the optimizer: lcb (the 90%% bound) or posterior (mean - buffer; uncertainty "
                         "then lives only in the covariance and CVaR). auto = posterior once exec_model.json is fitted, else lcb")
    ap.add_argument("--explore", type=float, default=0.10,
                    help="Probability of EACH +/-1c exploration arm (one contract, propensity logged); decays as the execution "
                         "model accumulates quotes")
    ap.add_argument("--ws", action="store_true",
                    help="Use the Kalshi WebSocket order book (pre-send checks and fast cancels); REST is the fallback. "
                         "Verify first with --ws-selftest")
    ap.add_argument("--ws-selftest", action="store_true", help="Connect the WebSocket, compare with REST on live markets, exit")
    ap.add_argument("--team-risk-sd", type=float, default=0.01, help="Model-risk factor shared by one team's candidates (rotation)")
    ap.add_argument("--player-risk-sd", type=float, default=0.01, help="Model-risk factor shared by one player's candidates (role)")
    ap.add_argument("--pilot-session-cap", type=float, default=0.01,
                    help="PILOT tier budget: max open risk / bankroll across all PILOT orders tonight (not a quota)")
    ap.add_argument("--pilot-player-cap", type=float, default=0.0025, help="PILOT tier budget per player / bankroll")
    ap.add_argument("--pilot-min-mean-edge-c", type=float, default=1.0, help="PILOT admission: posterior mean edge (cents) above this")
    ap.add_argument("--pilot-lcb-z", type=float, default=0.84, help="PILOT admission: z of the weaker lower bound (0.84 ~ 80%% one-sided)")
    ap.add_argument("--pilot-min-lcb-c", type=float, default=0.0, help="PILOT admission: that lower bound must be >= this (cents)")
    ap.add_argument("--qev-z", type=float, default=1.282, help="Quote choice: argmax QEV - z x sd(QEV) for PRODUCTION; QEV_LCB > 0 required")
    ap.add_argument("--exec-shrink-k", type=float, default=30.0,
                    help="Quote-specific execution residual shrunk toward the side prior: weight n_fills / (n_fills + k)")
    ap.add_argument("--explore-prod-cap", type=float, default=0.02, help="Exploration probability cap per arm in PRODUCTION")
    ap.add_argument("--max-strikes", type=int, default=1, help="Strikes per (player, stat, side) offered to the optimizer")
    ap.add_argument("--starve-min", type=float, default=30.0, help="Print the OPPORTUNITY STARVATION report after this many quiet minutes")
    args = ap.parse_args()
    global QUOTE_MAX_AGE
    QUOTE_MAX_AGE = max(5, args.quote_ttl - 3)
    mw_manual = "--model-weight" in sys.argv
    if args.pilot_live:
        # PILOT LIVE: real money to collect execution evidence for an unvalidated configuration, at the smallest size that
        # still produces it. Never production; every limit below is forced, whatever else was passed.
        args.live, args.taker, args.allow_no, args.no_probe = True, False, False, 0
        args.no_exits = True        # exits sell at the bid (taker): not in a maker-only pilot; positions settle
        args.kelly = min(args.kelly, 0.10)
        args.session_cap, args.player_cap = min(args.session_cap, 0.01), min(args.player_cap, 0.0025)
    args.max_contracts = 1 if args.pilot_live else None
    if args.live and mw_manual and not (args.override_model_weight_risk or args.pilot_live):
        sys.exit("--model-weight with --live is refused: a manual weight bypasses the evidence hierarchy. The live weight comes "
                 "only from calibration.json (current engine, trade regime, out of sample). (--override-model-weight-risk exists.)")
    research = [f for f, on in (("MODEL WEIGHT " + f"{args.model_weight:.0%}", mw_manual and not args.live),
                                ("MIDPOINT QUOTE LADDER", getattr(args, "quote_ladder", "bid1") == "mid"),
                                ("NO ENABLED", args.allow_no),
                                ("EXECUTION PENALTY CHANGED", "--exec-alpha-no-c" in sys.argv or "--exec-alpha-yes-c" in sys.argv)) if on]
    if research and args.live:
        logger.error("Research-only overrides cannot trade live: forcing PAPER mode.")
        args.live = False
    if research:
        args.pilot_live = False
        args.max_contracts = None
        print("!" * 92 + f"\n RESEARCH OVERRIDE: {', '.join(research)}. NOT VALIDATED FOR LIVE CAPITAL. PAPER MODE ONLY.\n" + "!" * 92)
    if args.pilot_live:
        print("#" * 92 + "\n PILOT LIVE: REAL MONEY, NOT PRODUCTION. Unvalidated configuration collecting execution evidence: 1 contract per\n"
              f" order, session risk {args.session_cap:.0%}, player risk {args.player_cap:.2%}, YES only, no taker, Kelly {args.kelly:.2f}.\n" + "#" * 92)
    mg.MATCHUP_STRENGTH = float(np.clip(args.matchup_strength, 0.0, 1.0))
    shadow_strength = 1.0 if mg.MATCHUP_STRENGTH < 0.5 else 0.0   # the A/B arm is the other end   # refresh before the exchange expiry, so the bot never "keeps" an expired quote

    print("=" * 92)
    print(f" KALSHI NBA IN-PLAY TRADER {TRADER_VERSION} (MARKET-ANCHORED MARKOV | CORRELATED FRACTIONAL KELLY | MAKER-ONLY)")
    print(f" Mode:     {'LIVE (REAL MONEY)' if args.live else 'PAPER'}   Bankroll ${args.bankroll:.0f}   "
          f"Caps: session {args.session_cap:.0%} | game {args.game_cap:.0%} | player {args.player_cap:.2%} | position {args.single_cap:.0%}")
    print(f" Pricing:  logit-pool model weight {args.model_weight:.2f} (calibration.json may override) | min EV {args.min_ev:.1f}% + {args.min_edge_c:.1f}c | price {args.price_min}-{args.price_max}¢ | "
          f"spread <= {args.max_spread}¢ | disagree <= {args.max_disagree:.2f}")
    print(f" Fees:     {'ZERO (fee-free maker only)' if args.fee_policy == 'zero' else ('MAKER + TAKER (taker at ROI >= ' + format(args.taker_min_ev, '.0f') + '% after fee)' if args.taker else 'MAKER ONLY (post-only, no taker fees)')}"
          f"   Sizing: correlated {args.kelly:.2f}-Kelly")
    print(f" Edge:     90% LCB of (directional + execution alpha - fee) >= {args.min_edge_c:.1f}c + inventory skew | buffer {args.as_buffer_c:.1f}c | "
          f"quote life {args.quote_ttl}s | pre-send re-check {'OFF' if args.no_revalidate else 'ON'}")
    print(f" Ledger:   {LEDGER_CSV}")
    print("=" * 92)

    if not DT_API_KEY:
        logger.error("DT_API_KEY missing from Keychain or environment.")
        sys.exit(1)

    if os.path.exists(PRED_LOG):
        with open(PRED_LOG) as f:
            if "yes_ask" not in f.readline():      # v2026.29 adds book and regime columns
                os.replace(PRED_LOG, PRED_LOG.replace(".csv", "_oldformat.csv"))
    cal, cal_id = load_calibration()
    authority: Dict[str, dict] = {}
    trade_stats = {s.strip().upper() for s in args.stats.split(",") if s.strip()}
    pool_w, stat_w, book_w, cell_w, w_sd = args.model_weight, {}, args.book_weight, {}, args.model_weight_sd
    book_cell_w: Dict[str, float] = {}
    try:
        with open(CAL_JSON) as _f:
            _pool = json.load(_f).get("_pool", {})
        if cal_id != "none" and "--model-weight" not in sys.argv and _pool.get("w") is not None:
            pool_w = min(0.35, max(0.0, float(_pool["w"])))
            stat_w = {k: min(0.35, max(0.0, float(v))) for k, v in (_pool.get("stat_w") or {}).items()}
            cell_w = {k: min(0.35, max(0.0, float(v))) for k, v in (_pool.get("cell_w") or {}).items()}
            if _pool.get("w_sd") is not None and "--model-weight-sd" not in sys.argv:
                w_sd = max(0.0, float(_pool["w_sd"]))
        if cal_id != "none" and "--book-weight" not in sys.argv and _pool.get("w_book") is not None:
            book_w = min(0.5, max(0.0, float(_pool["w_book"])))
        if cal_id != "none" and "--book-weight" not in sys.argv:
            book_cell_w = {k: min(0.5, max(0.0, float(v))) for k, v in (_pool.get("book_cell_w") or {}).items()}
        if cal_id != "none" and not mw_manual:
            if isinstance(_pool.get("authority"), dict):
                authority = _pool["authority"]
            elif pool_w > 0 or stat_w or cell_w:
                # pre-v2026.37 calibration: its earned weights become PILOT cells (one contract) until refitted
                for _k, _v in list(cell_w.items()) + list(stat_w.items()) + ([("*", pool_w)] if pool_w > 0 else []):
                    authority[_k] = dict(weight=_v, tier="PILOT", confidence=0.35, research_reason="", w_sd=None)
                logger.warning("calibration.json has no local-authority block (pre-v2026.37): its earned weights are treated as "
                               "PILOT cells. Run fit_calibration.py to build tiers.")

    except Exception:
        pass

    def bw_for(stat: str, period: int, per_clock: Optional[float] = None) -> float:
        """Sportsbook weight for this (stat, phase) cell if one was earned walk-forward, else the global book weight."""
        b_ = period_bucket(period, per_clock)
        for k in (f"{stat}|{b_}", f"*|{b_}"):
            if k in book_cell_w:
                return book_cell_w[k]
        return book_w
    # v2026.37: with local authority every phase must EARN its own cell (Q1/Q2/HT included), so the default is Q1; a
    # manual research weight has no phase evidence and keeps the second-half default.
    min_period = args.min_period if args.min_period is not None else (1 if (authority and not mw_manual) else 3)
    n_tier = {t: sum(1 for v in authority.values() if v.get("tier") == t) for t in TIERS}
    # Execution priors (v2026.36): a quote policy's fill population is its own. qp5 starts from a WEAK hierarchical prior
    # (15% of the qp3 mean, wide s.d.) and is updated by its own live marks (exec_priors.json, same policy) by precision
    # weighting; an explicit CLI value replaces it (research).
    exec_prior = {sd_: (OLD_PRIOR_WEIGHT * OLD_POLICY_PRIOR[sd_][0], NEW_POLICY_SD[sd_]) for sd_ in ("yes", "no")}
    exec_prior_own = {"yes": 0, "no": 0}
    side_on = {"yes": True, "no": bool(args.allow_no)}
    probe_left = max(0, int(args.no_probe)) if not side_on["no"] else 0
    prior_src = f"hierarchical {QUOTE_POLICY} prior ({OLD_PRIOR_WEIGHT:.0%} of qp3 mean, wide s.d.)"
    # v2026.38: the execution prior is POLICY-FAMILY specific (the family does not depend on the prior itself)
    _em0 = load_exec_model()
    _fam_now = policy_identity(args, _em0["exec_model_id"] if _em0 else "none", "n/a")[1]
    try:
        with open(EXEC_PRIORS_JSON) as _f:
            _ep = json.load(_f)
        if _ep.get("quote_policy") == QUOTE_POLICY and _ep.get("policy_family_id") != _fam_now:
            logger.warning(f"exec_priors.json is for policy family {_ep.get('policy_family_id')}, this run is {_fam_now}: ignored "
                           "(execution evidence belongs to the family that produced it).")
        elif _ep.get("quote_policy") == QUOTE_POLICY:
            for _sd in ("yes", "no"):
                _v = _ep.get(_sd, {})
                if _v.get("orders", 0) >= 10 and _v.get("se_c"):
                    m0, s0 = exec_prior[_sd]
                    m1, s1 = float(_v["mean_c"]), max(0.25, float(_v["se_c"]))
                    w0, w1 = 1.0 / s0 ** 2, 1.0 / s1 ** 2
                    exec_prior[_sd] = ((w0 * m0 + w1 * m1) / (w0 + w1), (w0 + w1) ** -0.5)
                    exec_prior_own[_sd] = int(_v.get("orders", 0))
                if _v.get("status") == "fail":
                    side_on[_sd] = False
                elif _v.get("status") == "pass" and _sd == "no":
                    side_on["no"] = True
            prior_src = f"exec_priors.json ({_ep.get('fitted')})"
        else:
            logger.warning(f"exec_priors.json is for quote policy {_ep.get('quote_policy')}, not {QUOTE_POLICY}: ignored "
                           "(a different quoting rule selects different fills).")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"exec_priors.json unreadable ({e}); using the hierarchical prior.")
    if args.pilot_live:
        side_on["no"] = False       # the pilot is YES only, whatever exec_priors.json says
    for _sd in ("yes", "no"):
        _m, _s = getattr(args, f"exec_alpha_{_sd}_c"), getattr(args, f"exec_alpha_{_sd}_sd_c")
        if _m is not None or _s is not None:
            exec_prior[_sd] = (exec_prior[_sd][0] if _m is None else _m, exec_prior[_sd][1] if _s is None else _s)
            prior_src = "CLI override"
    if any(exec_prior_own.values()) and prior_src != "CLI override":
        prior_src = f"{QUOTE_POLICY} live marks (YES {exec_prior_own['yes']}, NO {exec_prior_own['no']} orders) + weak qp3 prior"
    if args.as_spread_frac < 1.0 and any(abs(exec_prior[_sd][0]) > 1e-12 for _sd in ("yes", "no")):
        sys.exit("--exec-alpha-*-c is the RAW 30s markout, which already contains whatever spread survives a fill. Combining it "
                 "with spread credit (--as-spread-frac < 1) counts the spread twice. Use one: --as-spread-frac 1 with execution "
                 "alpha (default), or --as-spread-frac g with --exec-alpha-no-c 0 --exec-alpha-yes-c 0.")
    exec_model = load_exec_model()
    challenger = XM.load_model(CHALLENGER_JSON, QUOTE_POLICY)
    run_is_production = bool(args.live and not args.pilot_live)
    run_is_live = bool(args.live)
    POLICY["exec_model_id"] = exec_model["exec_model_id"] if exec_model else "none"
    POLICY["exec_prior_id"] = file_hash(EXEC_PRIORS_JSON) if (prior_src.startswith("exec_priors") or prior_src.startswith(QUOTE_POLICY)) else \
        hashlib.sha256(json.dumps(exec_prior, sort_keys=True).encode()).hexdigest()[:12]
    POLICY["policy_id"], POLICY["policy_family_id"], policy_fields = policy_identity(args, POLICY["exec_model_id"], POLICY["exec_prior_id"])
    size_authority = load_authority(SIZE_AUTH_JSON, POLICY["policy_family_id"])
    shock_sds = dict(model=args.model_risk_sd, stat=args.stat_risk_sd, game=args.game_risk_sd, team=args.team_risk_sd,
                     player=args.player_risk_sd)
    print("=" * 92)
    print(f" ENGINE: {mg.ENGINE_VERSION}   TRADER: {TRADER_VERSION}   MODE: {'LIVE' if args.live else 'PAPER'}   RUN: {RUN_ID}")
    print(f" MODEL WEIGHT: {pool_w:.2f} / {'earned (' + cal_id + ')' if (cal_id != 'none' and not mw_manual) else ('RESEARCH OVERRIDE' if mw_manual else 'none earned -> no directional trading')}")
    print(f" NO SIDE: {'enabled' if side_on['no'] else 'disabled'}   YES SIDE: {'enabled' if side_on['yes'] else 'disabled'}   MIN PERIOD: Q{min_period}")
    print(f" EDGE RULE: 90% LCB >= {args.min_edge_c:.1f}c (+ inventory skew)   TAIL LIMIT: CVaR95 <= {args.cvar_limit:.0%} bankroll")
    print(f" EXECUTION PRIORS: {prior_src}: YES {exec_prior['yes'][0]:+.2f}c (s.e. {exec_prior['yes'][1]:.2f}) | NO {exec_prior['no'][0]:+.2f}c (s.e. {exec_prior['no'][1]:.2f}) | quote policy {QUOTE_POLICY}")
    print(f" RESEARCH OVERRIDES: {', '.join(research) if research else 'none'}")
    print(f" LOCAL AUTHORITY: PRODUCTION {n_tier['PRODUCTION']} | PILOT {n_tier['PILOT']} | RESEARCH {n_tier['RESEARCH']} cells"
          f"{'  (manual weight: ' + ('PILOT' if args.pilot_live else 'RESEARCH/paper') + ')' if mw_manual else ''}"
          f"   PILOT BUDGET: {args.pilot_session_cap:.2%} session, {args.pilot_player_cap:.2%} per player")
    print(f" POLICY: {QUOTE_POLICY} id {POLICY['policy_id']} (family {POLICY['policy_family_id']}) | execution model "
          f"{POLICY['exec_model_id']} | challenger {challenger['exec_model_id'] if challenger else 'none'}")
    print("=" * 92)
    books = SportsbookConsensus("" if args.no_books else ODDS_API_KEY, mg.normalize_player_name, interval=args.books_interval,
                                max_age=args.book_max_age)
    logger.info(f"Sportsbook consensus: {'ON (The Odds API, every ' + format(args.books_interval, '.0f') + 's per game)' if books.enabled else 'OFF (set ODDS_API_KEY to enable)'}"
                f" | book weight {book_w:.2f} | per-stat model weights {stat_w or 'none (global)'}")
    logger.info(f"Regime weights {cell_w or 'none fitted'} | pool-weight s.d. {w_sd:.2f} | matchup strength {mg.MATCHUP_STRENGTH:g} "
                f"(A/B shadow at {shadow_strength:g}, after orders)")
    logger.info(f"Probability: logit pool, model weight {pool_w:.2f} | trading {sorted(trade_stats)} from Q{min_period} | "
                f"gap {args.min_gap:.2f}-{args.max_disagree:.2f} | min edge {args.min_edge_c:.1f}c | {'maker+taker' if args.taker else 'maker only'}")
    logger.info(f"Calibration: {cal_id} " + (str({k: f'a={a:+.2f} b={b:.2f}' for k, (a, b) in cal.items()}) if cal else
                "(none: raw engine probabilities; run fit_calibration.py)"))
    gate_status = "n/a (paper)"
    if args.live:
        try:
            with open(STATUS_JSON) as f:
                gate = json.load(f)
        except Exception:
            gate = {"passed": False, "markets": 0}
        if gate.get("engine") != mg.ENGINE_VERSION:
            gate["passed"] = False      # evidence from another engine version cannot authorize this one
        # v2026.36 production authorization: ALL of these, or the run is not production.
        why_not = []
        # v2026.37 (cell-aware): the broad model gate is ENGINE SANITY (enough graded markets, not demonstrably worse than
        # the market), not "the whole model must beat Kalshi everywhere"; local alpha is authorized per cell. Execution
        # authority (fills + markouts for this policy family) and risk stay required.
        _san = gate.get("engine_sanity", {}) or {}
        if gate.get("engine") != mg.ENGINE_VERSION or not _san.get("passed"):
            _na = lambda v, u="": "n/a" if v is None else f"{v}{u}"
            why_not.append(f"engine sanity not passed for engine {mg.ENGINE_VERSION} ({gate.get('markets', 0) if gate.get('engine') == mg.ENGINE_VERSION else 0} "
                           f"graded markets; edge {_na(gate.get('edge_c'), 'c')}, 90% CI high {_na(gate.get('ci_hi'))})")
        if not (gate.get("fills", {}) or {}).get("passed"):
            why_not.append("fill gate not passed (settled LIVE base-arm orders of this policy family)")
        _mg = gate.get("markout_gate", {}) or {}
        if not _mg.get("passed"):
            why_not.append("markout gate not passed (YES side, this policy family)")
        elif _mg.get("policy_family_id") != POLICY["policy_family_id"]:
            why_not.append(f"markout-gate evidence is for policy family {_mg.get('policy_family_id')}, this run is {POLICY['policy_family_id']}")
        if not gate.get("passed"):
            why_not.append("model_status.json 'passed' is false (the grader's canonical production authorization)")
        if cal_id == "none":
            why_not.append("no calibration for this engine (calibration: none)")
        if mw_manual:
            why_not.append(f"manual model weight {args.model_weight:.2f} (production weight must be earned in calibration.json)")
        if exec_prior_own["yes"] < 30 or prior_src == "CLI override":
            why_not.append(f"execution prior not earned under quote policy {QUOTE_POLICY} ({exec_prior_own['yes']} live YES orders; need 30)")
        if not n_tier["PRODUCTION"]:
            why_not.append("no PRODUCTION authority cell in calibration.json (local cells must earn it walk-forward)")
        _fg = gate.get("fills", {}) or {}
        if _fg.get("policy_family_id") not in (None, POLICY["policy_family_id"]):
            why_not.append(f"fill-gate evidence is for policy family {_fg.get('policy_family_id')}, this run is {POLICY['policy_family_id']}")
        production_ok = not why_not
        gate_status = ("PILOT (authorization checks: " + ("passed" if production_ok else "not passed") + ")") if args.pilot_live \
            else ("PASSED (production)" if production_ok else "not passed")
        if not production_ok and not args.pilot_live:
            logger.error("PRODUCTION AUTHORIZATION FAILED:\n  - " + "\n  - ".join(why_not) +
                         "\n  --override-gate / --override-model-weight-risk no longer authorize production. To collect live "
                         "evidence at minimum size use --pilot-live; otherwise run without --live (paper).")
            sys.exit(1)
        if not production_ok:
            logger.warning("PILOT LIVE (not production) because:\n  - " + "\n  - ".join(why_not))
        if production_ok and not args.pilot_live and os.path.exists(PROD_PROFILE_JSON) and not args.write_production_profile:
            with open(PROD_PROFILE_JSON) as _pf:
                _prof = json.load(_pf)
            _now = production_profile(args, cal_id, min_period, side_on)
            _diff = [f"{k}: profile {_prof.get(k)!r} vs now {_now.get(k)!r}" for k in sorted(_now) if _prof.get(k) != _now.get(k)]
            if _diff:
                logger.error("PRODUCTION PROFILE MISMATCH (production_profile.json):\n  " + "\n  ".join(_diff))
                sys.exit(1)
        if production_ok and not args.pilot_live and args.write_production_profile:
            with open(PROD_PROFILE_JSON, "w") as _pf:
                json.dump(production_profile(args, cal_id, min_period, side_on), _pf, indent=2)
            logger.info("production_profile.json written: later production runs must match it.")
    kalshi = KalshiClient(KALSHI_KEY_ID, KALSHI_PEM_PATH, paper=not args.live)
    if args.live and not kalshi.can_trade:
        logger.error("--live requested but Kalshi key/PEM not loaded. Refusing to start.")
        sys.exit(1)
    if args.ws or args.ws_selftest:
        if kalshi._private_key is None or not kalshi.key_id:
            logger.error("--ws needs the Kalshi key and PEM (the order-book channel is authenticated).")
            sys.exit(1)
        from kalshi_ws import KalshiWSBook, ws_selftest
        if args.ws_selftest:
            tks = []
            for _ser in ("KXNBAPTS", "KXNBAREB", "KXNBAAST"):
                try:
                    _r = kalshi._session.get(f"{kalshi.BASE}/markets", params={"series_ticker": _ser, "status": "open", "limit": 5}, timeout=8)
                    tks += [x["ticker"] for x in (_r.json().get("markets") or [])][:2]
                except Exception as e:
                    logger.error(f"market list failed: {e}")
            if not tks:
                logger.error("No open NBA prop markets to test against right now; run the self-test while markets are listed.")
                sys.exit(1)
            sys.exit(0 if ws_selftest(kalshi.ws_headers, tks, kalshi.top_of_book) else 1)
        kalshi.ws = KalshiWSBook(kalshi.ws_headers)
        kalshi.ws.start()
        logger.info("Kalshi WebSocket order book: ON (REST is the fallback whenever a book is stale or out of sequence)")
    kalshi.load_series_fees()
    if args.fee_policy == "zero":
        allowed_series = kalshi.fee_free_series()
        logger.info(f"Zero-fee policy: tradeable series = {sorted(allowed_series) or 'NONE'}")
        if not allowed_series:
            logger.warning("No NBA prop series is fee-free for resting orders right now; the bot will price "
                           "markets but place no orders. Run with the default --fee-policy min to trade.")
    else:
        allowed_series = set(KALSHI_SERIES)
        logger.info(f"Fee-minimising policy: fee-free maker series = {sorted(kalshi.fee_free_series()) or 'none'}; "
                    f"others trade with the maker fee priced into EV.")

    dt_client = mg.DunksAndThreesClient(DT_API_KEY)
    base_kelly_live = min(max(0.0, float(args.kelly)), 0.10) if args.live else max(0.0, float(args.kelly))
    prediction_health = 1.0 if run_is_production else (0.75 if args.pilot_live else 1.0)
    execution_health = 1.0 if exec_model else (0.65 if args.pilot_live else 1.0)
    effective_kelly = base_kelly_live * prediction_health * execution_health if args.adaptive_kelly else base_kelly_live
    effective_kelly = min(base_kelly_live, max(0.0, effective_kelly))
    qp = CorrelatedKellyQP(session_cap=args.session_cap, game_cap=args.game_cap, single_cap=args.single_cap,
                           half_kelly=True, fav_taper=not args.no_fav_taper, player_cap=args.player_cap,
                           kelly_fraction=effective_kelly, model_sd=args.model_risk_sd, stat_sd=args.stat_risk_sd,
                           game_sd=args.game_risk_sd, team_sd=args.team_risk_sd, player_sd=args.player_risk_sd)
    explore_rng = np.random.default_rng(int(RUN_ID.replace("_", "")) % (2 ** 32))
    base_gamma = qp.gamma
    logger.info(f"Kelly: configured {args.kelly:.3f} | live base {base_kelly_live:.3f} | effective {effective_kelly:.3f}" + (" (adaptive)" if args.adaptive_kelly else ""))
    tox_pause_until = 0.0
    positions = load_positions_from_ledger()
    if positions:
        logger.info(f"Restored {len(positions)} positions (${sum(v['cost_usd'] for v in positions.values()):.2f}) from ledger.")
    book = OrderBook(kalshi, positions, allowed_series, zero_fee=args.fee_policy == "zero")
    try:   # every run archives exactly what it traded with
        code_hash = hashlib.sha1(b"".join(open(os.path.join(CURRENT_DIR, f), "rb").read() for f in sorted(os.listdir(CURRENT_DIR))
                                          if f.endswith(".py"))).hexdigest()[:12]
        cfg = dict(vars(args), trader=TRADER_VERSION, engine=mg.ENGINE_VERSION, calibration_id=cal_id, pool_weight=pool_w,
                   cell_weights=cell_w, pool_weight_sd=w_sd, min_period_used=min_period, run_id=RUN_ID, quote_policy=QUOTE_POLICY,
                   edge_rule=EDGE_RULE, cvar_limit=args.cvar_limit, no_enabled=side_on["no"], yes_enabled=side_on["yes"],
                   exec_priors=exec_prior, exec_prior_source=prior_src, research_overrides=research,
                   run_mode=("research" if research else ("pilot" if args.pilot_live else ("production" if args.live else "paper"))),
                   stat_weights=stat_w, book_weight=book_w, books_enabled=books.enabled, code_hash=code_hash,
                   exec_model=bool(exec_model), gate_status=gate_status, ws=kalshi.ws is not None,
                   policy_id=POLICY["policy_id"], policy_family_id=POLICY["policy_family_id"], exec_model_id=POLICY["exec_model_id"],
                   exec_prior_id=POLICY["exec_prior_id"], policy_fields=policy_fields, authority_cells=n_tier,
                   authority_production=[k for k, v in authority.items() if v.get("tier") == "PRODUCTION"],
                   authority_pilot=[k for k, v in authority.items() if v.get("tier") == "PILOT"],
                   challenger_model_id=challenger["exec_model_id"] if challenger else None,
                   started=datetime.now().isoformat(timespec="seconds"))
        with open(os.path.join(CURRENT_DIR, f"run_config_{RUN_ID}.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        logger.info(f"Run config archived (code {code_hash}, engine {mg.ENGINE_VERSION}).")
        print("-" * 92 + "\n STARTUP MANIFEST (authoritative; also in " + f"run_config_{RUN_ID}.json)")
        for _k, _v in (("TRADER", TRADER_VERSION), ("ENGINE", mg.ENGINE_VERSION), ("CODE HASH", code_hash), ("RUN MODE", cfg["run_mode"].upper()),
                       ("CALIBRATION ID", cal_id), ("MODEL WEIGHT SOURCE", "earned" if (cal_id != "none" and not mw_manual) else ("manual" if mw_manual else "none")),
                       ("EXECUTION PRIOR SOURCE", prior_src), ("EXECUTION MODEL", "fitted (argmax P(fill) x EV)" if exec_model else "none (base bid+1 + exploration)"),
                       ("QUOTE POLICY", f"{QUOTE_POLICY} (policy {POLICY['policy_id']}, family {POLICY['policy_family_id']})"),
                       ("LOCAL AUTHORITY", f"PRODUCTION {n_tier['PRODUCTION']} / PILOT {n_tier['PILOT']} / RESEARCH {n_tier['RESEARCH']} cells"),
                       ("LIVE GATE", gate_status), ("BANKROLL", f"${args.bankroll:.0f}"),
                       ("CAPS", f"session {args.session_cap:.2%} game {args.game_cap:.2%} player {args.player_cap:.2%} position {args.single_cap:.2%}"
                                + (" | 1 contract/order" if args.max_contracts == 1 else "")),
                       ("MODEL PERIODS", f"Q{min_period}+"), ("NO STATUS", "enabled" if side_on["no"] else "disabled"),
                       ("CVaR LIMIT", f"{args.cvar_limit:.0%} of bankroll"), ("DAILY STOP", f"{args.daily_loss:.0%} of bankroll"),
                       ("KELLY", f"{args.kelly:.2f} ({'posterior mean' if (args.kelly_mean == 'posterior' or (args.kelly_mean == 'auto' and exec_model)) else '90% LCB'} as expected return)"),
                       ("BOOK FEED", "WebSocket + REST fallback" if kalshi.ws is not None else "REST")):
            print(f"   {_k:<24}{_v}")
        print("-" * 92)
    except Exception as e:
        logger.error(f"Run config archive failed: {e}")
    def exchange_check(open_tickers: set):
        if not kalshi.can_trade:
            return
        ex = kalshi.get_positions()
        if ex is None:
            logger.warning("Exchange positions unreadable this check (no change to trading state).")
            return
        mm = position_mismatches(positions, ex, open_tickers)
        if mm and not book.killed:
            book.killed = "exchange position mismatch"
            logger.error("KILL SWITCH: local positions differ from Kalshi -> no new entries until restart after review:\n  " + "\n  ".join(mm[:12]))
        elif not mm:
            logger.info(f"Exchange reconciliation OK ({len(ex)} open position(s) on Kalshi).")
        # v2026.36: a local position whose market is no longer open and which Kalshi no longer holds has settled. It stops
        # counting as open risk (v2026.35 kept a settled $0.70 position reserved all night). The ledger keeps the fill.
        gone = [k for k in list(positions) if k[0] not in open_tickers and abs(ex.get(k[0], 0.0)) < 1.0]
        for k in gone:
            v = positions.pop(k)
            logger.info(f"Settled/closed: {k[0]} {k[1].upper()} {v.get('contracts', 0):g}x (cost ${v.get('cost_usd', 0):.2f}) "
                        f"removed from open risk.")

    dt_state = {"ts": 0.0}
    calib_cache: Dict[str, Tuple[dict, dict]] = {}
    prev_mid: Dict[str, float] = {}
    watch_ids: Dict[str, int] = {}
    watch_titles: Dict[str, str] = {}
    provenance_warned: Dict[str, str] = {}
    sim_cache: Dict[str, tuple] = {}
    sim_act: Dict[str, Tuple[str, int]] = {}    # game title -> (NBA game id, play number the current prices were built on)

    def outcome_of(b: dict):
        """Simulated win vector for a resting/pending order from the current simulation of its game (None if unavailable:
        the tail check then assumes a full loss)."""
        try:
            gid = sim_act.get(b.get("game_key"), (None, None))[0]
            sim = sim_cache.get(gid, (None, None, None))[1]
            smp = sim[b["stat"]][:, sim["id_to_idx"][b["player_id"]]]
            return (smp > float(b["line"])) if b["side"] == "yes" else (smp < float(b["line"]))
        except Exception:
            return None
    def price_side(c: dict, bid: int, ask: int, p_mkt: float, p_used: float, tox_c: Optional[float]) -> Tuple[Optional[dict], str]:
        """Admission, quote choice and economics for one (market, side) on a given book (v2026.37). Used for the first
        pass AND for the final fresh-book re-optimization, so both pick the quote with the same rule.
        For every maker level from the bid up to min(ask - 1, mid):
            execution mean  = shrink(model residual E[M30 - drift | fill, quote], side/policy prior, model sample)
                              (no champion model: the side/policy prior)
            edge            = 90% LCB (PRODUCTION) / 80% LCB + posterior-mean bar (PILOT, paper RESEARCH)
            QuoteEV         = P(fill) x (directional alpha + execution mean - fee), with bootstrap s.d.
        Choice: champion model -> argmax QEV_LCB (PRODUCTION; QEV_LCB > 0 required) or argmax QEV (PILOT; QEV > 0);
        no model -> the base policy (bid+1, join when toxic / NO / probe). Exploration arms use their OWN predictions."""
        tier, side = c["tier"], c["side"]
        alpha = 100.0 * (p_used - p_mkt)
        join = c["join"] or ask - bid < 2
        top = min(ask - 1, int(math.floor((bid + ask) / 2.0)))
        levels = [bid] if join else list(range(bid, max(bid, top) + 1))
        lcb_z = args.lcb_z if tier == "PRODUCTION" else args.pilot_lcb_z
        prior_m, prior_sd = exec_prior[side][0], c["adm_sd"]
        n_mod = float(exec_model.get("n_fills", 0)) if exec_model else 0.0
        lam = n_mod / (n_mod + args.exec_shrink_k) if exec_model else 0.0
        inv_x = c["inv_x"]
        if args.taker and args.fee_policy == "min" and tier == "PRODUCTION" and args.tail_min <= ask <= args.tail_max:
            # taker route (only with --taker, only PRODUCTION cells): no spread credit, no execution alpha, certain fill
            fee_t = kalshi.fee_cents(c["ticker"], ask, False)
            lcb_t, pc_t, pt_t = conservative_edge(p_used, p_mkt, ask, fee_t, c["p_model"], c["n"], c["w_m"], c["p_sh"],
                                                  as_spread_frac=0.0, as_buffer_c=args.as_buffer_c, shadow_z=args.shadow_z,
                                                  n_shadow=c["n_shadow"], w_sd=c["w_sd"], lcb_z=lcb_z)
            roi_t = (pc_t * 100.0) / (ask + fee_t) - 1.0
            need_t = c["need"] if c["late"] else max(c["need"], args.taker_min_ev)
            tail_t = 0.0 if args.price_min <= ask <= args.price_max else args.tail_extra_c
            if roi_t >= need_t / 100.0 and lcb_t >= args.min_edge_c + inv_x + tail_t:
                mean_t = pt_t["mean_c"] - pt_t["det_c"]
                kmean = args.kelly_mean if args.kelly_mean != "auto" else ("posterior" if exec_model is not None else "lcb")
                return dict(px=ask, fee=fee_t, roi=roi_t, p_cons=pc_t, lcb=lcb_t, parts=pt_t, pf=1.0, em=0.0, em_model=None,
                            qev=mean_t, qev_sd=pt_t["sigma_c"], qev_lcb=mean_t - args.qev_z * pt_t["sigma_c"], mean_e=mean_t,
                            arm="base", prop=1.0, kelly_mean=kmean, cond_edge_c=mean_t if kmean == "posterior" else lcb_t,
                            quote_level="take", aggr=1.0, is_maker=False), "ok"
        scored = {}
        for px in levels:
            if not (args.tail_min <= px <= args.tail_max):
                continue
            tail_x = 0.0 if args.price_min <= px <= args.price_max else args.tail_extra_c
            fee_c = kalshi.fee_cents(c["ticker"], px, True)
            pf, em_model, qv = None, None, None
            if exec_model is not None:
                x = exec_features(side, c["stat"], c["period"], bid, ask, px, alpha, ttl_s=c["ttl"], stop=c["stop"],
                                  phase=c["phase"], qty_bid=c["qty_side_bid"], qty_ask=c["qty_side_ask"],
                                  queue_ahead=c["qty_side_bid"] if px == bid else 0.0, book_age_s=c.get("book_age_s", 0.0),
                                  t_since_action_s=c["t_since_action_s"], micro=c["micro"])
                qv = XM.quote_value(exec_model, x, alpha, fee_c, z=args.qev_z)
                pf, em_model = qv["pf"], qv["em"]
                em = lam * em_model + (1.0 - lam) * prior_m
                ex_sd = math.sqrt((lam * qv["em_sd"]) ** 2 + ((1.0 - lam) * prior_sd) ** 2)
            else:
                em, ex_sd = prior_m, prior_sd
            lcb, p_cons, parts = conservative_edge(
                p_used, p_mkt, px, fee_c, c["p_model"], c["n"], c["w_m"], c["p_sh"], as_spread_frac=args.as_spread_frac,
                as_buffer_c=args.as_buffer_c, exec_alpha_c=em, exec_sd_c=ex_sd, shadow_z=args.shadow_z,
                n_shadow=c["n_shadow"], w_sd=c["w_sd"], lcb_z=lcb_z)
            roi = (p_cons * 100.0) / (px + fee_c) - 1.0
            mean_e = parts["mean_c"] - parts["det_c"]
            if tier == "PRODUCTION":
                ok = roi >= c["need"] / 100.0 and lcb >= args.min_edge_c + inv_x + tail_x
            else:
                ok = (mean_e >= args.pilot_min_mean_edge_c + tail_x and lcb >= args.pilot_min_lcb_c + inv_x
                      and (tox_c is None or tox_c > args.tox_warn_c))
            if not ok:
                continue
            if pf is not None:
                qev = pf * mean_e
                qev_sd = math.sqrt(qv["qev_sd"] ** 2 + (pf * parts["sigma_c"]) ** 2)
            else:
                qev, qev_sd = mean_e, parts["sigma_c"]
            scored[px] = dict(px=px, fee=fee_c, roi=roi, p_cons=p_cons, lcb=lcb, parts=parts, pf=pf, em=em,
                              em_model=em_model, qev=qev, qev_sd=qev_sd, qev_lcb=qev - args.qev_z * qev_sd, mean_e=mean_e)
        if not scored:
            return None, "edge"
        # v2026.38 action support: a PRODUCTION quote may only use a quote offset the champion was trained on with enough
        # logged actions; an unsupported level is extrapolation and is left to PILOT exploration.
        _sup = exec_model.get("supported_offsets") if exec_model is not None else None
        if tier == "PRODUCTION" and _sup is not None:
            scored = {px_: v_ for px_, v_ in scored.items() if (px_ - bid) in set(_sup)}
            if not scored:
                return None, "no_action_support"
        if exec_model is not None:
            key = (lambda d: d["qev_lcb"]) if tier == "PRODUCTION" else (lambda d: d["qev"])
            best = max(scored.values(), key=key)
            if (tier == "PRODUCTION" and best["qev_lcb"] <= 0.0) or best["qev"] <= 0.0:
                return dict(best, rejected=True), "execution_EV"
        elif args.quote_ladder == "mid" and not run_is_live:
            best = scored[max(scored)]
        else:
            base = bid if join else bid + 1
            best = scored[base] if base in scored else scored[min(scored)]
        arm, prop = "base", 1.0
        # v2026.38 action support: the behaviour policy pi_b(q | X) over EVERY admissible level on this book. The greedy
        # level gets 1 - p_exp x (#arms); each available +/-1c arm gets p_exp; every other level gets 0. It is logged with
        # the quote so off-policy evaluation can tell observed support from extrapolation.
        if c["is_probe"]:
            p_exp = 0.0
        else:
            p_exp = min(0.25, max(0.0, args.explore)) * (float(exec_model.get("explore_scale", 1.0)) if exec_model else 1.0)
            if run_is_production and tier == "PRODUCTION":
                p_exp = min(p_exp, args.explore_prod_cap)
        greedy = best["px"]
        arms_all = [(a_, p_) for a_, p_ in (("minus", greedy - 1), ("plus", greedy + 1)) if p_ in scored]
        pi_b = {int(q_): 0.0 for q_ in scored}
        pi_b[int(greedy)] = 1.0 - p_exp * len(arms_all)
        for _a, _p in arms_all:
            pi_b[int(_p)] = p_exp
        forced = c.get("force_arm")
        if forced is not None:
            px_f = best["px"] + (-1 if forced == "minus" else 1) if forced in ("minus", "plus") else best["px"]
            if forced != "base":
                if px_f not in scored:
                    return None, "arm_unavailable"
                best, arm = scored[px_f], forced
            prop = pi_b.get(int(best["px"]), c.get("force_prop", 1.0))   # probability under the FINAL (fresh) book
        elif not c["is_probe"]:
            arms = arms_all
            u, acc = float(explore_rng.random()), 0.0
            chosen = None
            for a_, p_ in arms:
                acc += p_exp
                if u < acc:
                    chosen = (a_, p_)
                    break
            if chosen is not None:
                best, arm, prop = scored[chosen[1]], chosen[0], p_exp
            else:
                prop = 1.0 - p_exp * len(arms)
        kmean = args.kelly_mean if args.kelly_mean != "auto" else ("posterior" if exec_model is not None else "lcb")
        cond = best["mean_e"] if kmean == "posterior" else best["lcb"]
        px = best["px"]
        best.update(arm=arm, prop=prop, kelly_mean=kmean, cond_edge_c=cond, pi_b=pi_b, ladder=sorted(pi_b), greedy_px=int(greedy),
                    quote_level="join" if px == bid else ("bid+1" if px == bid + 1 else f"bid+{px - bid}"),
                    aggr=(px - bid) / max(ask - bid, 1), is_maker=True)
        return best, "ok"

    def challenger_preds(c: dict, bid: int, ask: int, px: int, alpha: float, fee_c: float) -> Tuple[Optional[float], Optional[float]]:
        """Shadow score of the challenger execution model for the quote actually chosen (never used to decide)."""
        if challenger is None:
            return None, None
        try:
            x = exec_features(c["side"], c["stat"], c["period"], bid, ask, px, alpha, ttl_s=c["ttl"], stop=c["stop"],
                              phase=c["phase"], qty_bid=c["qty_side_bid"], qty_ask=c["qty_side_ask"],
                              queue_ahead=c["qty_side_bid"] if px == bid else 0.0, t_since_action_s=c["t_since_action_s"],
                              micro=c["micro"])
            q = XM.quote_value(challenger, x, alpha, fee_c)
            return round(q["pf"], 4), round(q["qev"], 3)
        except Exception:
            return None, None

    calib_shadow: Dict[str, dict] = {}
    funnel = Funnel(starve_s=60.0 * args.starve_min)
    act_seen_ts: Dict[str, float] = {}         # NBA game id -> when we first saw its current play number
    ws_want: set = set()
    if kalshi.ws is not None:
        # MARKET CLOCK (v2026.37): every synchronized WebSocket book update for a market with a live quote triggers the
        # fast checks for that market immediately (tens of ms), instead of waiting for the --watch loop. The Markov
        # model is never rerun here: only stale-quote protection and markouts.
        _mq: "queue.Queue[str]" = queue.Queue()
        def _on_book(tk: str):
            _mq.put(tk)
        kalshi.ws.on_update = _on_book
        def _market_clock():
            while True:
                tk = _mq.get()
                if tk is None:
                    return
                pend = {tk}
                try:
                    while True:
                        pend.add(_mq.get_nowait())
                except queue.Empty:
                    pass
                pend.discard(None)
                try:
                    with book.lock:
                        live_tk = {b["ticker"] for b in book.resting.values()}
                        hit = pend & live_tk
                        if hit:
                            _fc = book.fast_checks(tickers=hit)
                            if _fc:
                                logger.info(f"[MARKET CLOCK] book changed under resting quotes: {_fc}")
                        book.mark_fresh(tickers=pend)
                except Exception as e:
                    logger.warning(f"market clock: {e}")
        threading.Thread(target=_market_clock, name="market-clock", daemon=True).start()
    ttl = args.quote_ttl   # exchange-side expiry; quotes are refreshed after QUOTE_MAX_AGE (= ttl - 3)
    cycle = 0
    gid_of_title = lambda t: next((g for g, tt in watch_titles.items() if tt == t), None)
    book.action_of = lambda title: watch_ids.get(gid_of_title(title))

    def refresh_dt(date_str: str) -> bool:
        try:
            epm_opt, box_preds, team_epm = dt_client.fetch_game_day_data(date_str)
            if epm_opt.empty or box_preds.empty:
                raise RuntimeError("empty D&T response")
        except Exception as e:
            logger.error(f"D&T fetch for {date_str} failed: {e}" + (" (keeping previous data)" if "epm" in dt_state else ""))
            dt_state["ts"] = time.time() - 840   # retry in about a minute
            return False
        dt_state.update(ts=time.time(), date=date_str, epm=epm_opt, box=box_preds, team={
            int(r["team_id"]): float(r.get("team_depm_next") or r.get("team_depm_game_optimized", 0.0) or 0.0)
            for _, r in team_epm.iterrows()})
        names = {}
        for _, r in epm_opt.iterrows():
            n = re.sub(r"[^a-z0-9]+", " ", mg.normalize_player_name(str(r["player_name"]))).strip()
            names.setdefault(n, set()).add((int(r["player_id"]), str(r["player_name"])))
        dt_state["names"] = [(n, *next(iter(v))) for n, v in names.items() if len(v) == 1 and n]
        logger.info(f"D&T loaded for {date_str}: {len(epm_opt)} player rows, {len(box_preds)} box rows.")
        return True

    try:
        while True:
            cycle += 1
            t_cycle = time.time()
            funnel.start_cycle()
            ws_want = set()
            try:
                games = mg.NBALiveFeedClient.get_todays_games()
                statuses = (1, 2) if args.pregame else (2,)
                pre = [g for g in games if int(g.get("gameStatus", 1)) == 1]
                # D&T date = the NBA scoreboard's game date (correct even if started after midnight)
                codes = sorted({str(g.get("gameCode", "")).split("/")[0] for g in games} - {""})
                dt_date = datetime.strptime(codes[0], "%Y%m%d").strftime("%Y-%m-%d") if codes else TODAY_STR
                if "epm" not in dt_state or dt_state.get("date") != dt_date or (pre and time.time() - dt_state["ts"] > 900):
                    if refresh_dt(dt_date):   # injury / minutes updates before tip (ETag makes repeats cheap)
                        dt_state["date"] = dt_date
                        for g in pre:
                            calib_cache.pop(str(g.get("gameId")), None)
                if "epm" not in dt_state:
                    logger.warning("No D&T data yet; retrying next cycle.")
                    if args.once:
                        break
                    time.sleep(args.interval)
                    continue
                active = [g for g in games if int(g.get("gameStatus", 1)) in statuses]
                if not active:
                    book.reconcile({})
                    logger.info(f"Cycle #{cycle}: no tradeable games ({len(pre)} scheduled). Sleeping {args.interval}s.")
                    if args.once:
                        break
                    time.sleep(args.interval)
                    continue

                tokens = {game_date_token(str(g.get("gameCode", ""))) for g in active} - {""}
                markets = kalshi.fetch_open_props(dt_state["names"], tokens)
                by_ticker = {m.ticker: m for m in markets}
                _nm0 = len(book.markouts)
                book.reconcile(by_ticker, cancel_all=False)   # quotes persist; fills are collected; risk released on terminal proof only
                funnel.add("fills", max(0, len(book.markouts) - _nm0))
                book.update_markouts(by_ticker)
                if cycle == 1 or cycle % 20 == 0:
                    exchange_check(set(by_ticker))
                _dpnl_mid = daily_pnl(by_ticker, prev_mid)
                _dpnl = daily_pnl(by_ticker, prev_mid, kalshi.fee_cents, liquidation=True)   # risk switches use liquidation value
                if _dpnl <= -args.daily_loss * args.bankroll and not book.killed:
                    book.killed = f"daily loss ${_dpnl:.2f} <= -{args.daily_loss:.0%} of bankroll"
                    logger.error(f"KILL SWITCH: {book.killed}. Cancelling resting orders; no new entries tonight.")
                    with book.lock:
                        _gks = {b.get('game_key') for b in book.resting.values()}
                    for _gk in _gks:
                        book.cancel_game(_gk, "kill_switch")
                _tox = book.toxicity(args.tox_n)
                if _tox is not None and _tox <= args.tox_stop_c:
                    tox_pause_until = time.time() + 600
                thr_mult, thr_join, thr_no_add, thr_why = throttle(_dpnl, args.bankroll, _tox, args.tox_warn_c, args.tox_stop_c)
                if time.time() < tox_pause_until:
                    thr_mult, thr_why = 0.0, (thr_why + "; " if thr_why else "") + "toxicity pause"
                if thr_why:
                    logger.warning(f"RISK THROTTLE: {thr_why} (P&L at liquidation ${_dpnl:+.2f}, at mid ${_dpnl_mid:+.2f})")
                qp.gamma = base_gamma / max(thr_mult, 1e-9)        # Kelly fraction scaled by the throttle
                with book.lock:      # ONE consistent snapshot (the market clock moves orders between dicts)
                    reserved = book.reserved(include_resting=False)            # pending (cancel acked, unproven)
                    resting_res = book.reserved(only_resting=True)             # every live order: each can still fill
                    # quotes the optimizer may re-decide this cycle: only healthy RESTING / PARTIALLY_FILLED ones. A
                    # CANCEL_FAILED order is still live and can fill: it is ALWAYS counted (review fix)
                    resting_live = book.reserved(only_resting=True, states=(RESTING, PARTIALLY_FILLED))
                    resting_cf = book.reserved(only_resting=True, states=(CANCEL_FAILED, CANCEL_REQUESTED))
                    snap_resting = [dict(b) for b in book.resting.values()]
                    snap_pending = [dict(b) for b in book.pending.values()]
                cash = kalshi.get_sports_shard_balance() - sum(v["cost_usd"] for v in reserved.values()) \
                    - sum(v["cost_usd"] for v in resting_res.values())
                player_exp: Dict[str, float] = {}
                for _src in (positions, reserved, resting_res):
                    for _v in _src.values():
                        player_exp[_v.get("player", "")] = player_exp.get(_v.get("player", ""), 0.0) + max(0.0, _v.get("cost_usd", 0.0))
                # exposure of each key's OWN live/pending orders (a pilot's own resting quote is not "player already held")
                own_key_exp: Dict[Tuple[str, str], float] = {}
                for _src in (reserved, resting_res):
                    for _k, _v in _src.items():
                        own_key_exp[_k] = own_key_exp.get(_k, 0.0) + max(0.0, _v.get("cost_usd", 0.0))
                held = []
                elig_rows = []
                candidates, mkt_of = [], {}
                pred_rows, exits, pred_meta, shadow_todo = [], [], [], {}
                n_gate = {"feed_stale": 0, "late_close": 0, "provenance": 0, "books_disagree": 0, "live_ball": 0, "scope": 0, "small_gap": 0, "contradiction": 0, "fee": 0, "spread": 0, "unstable": 0, "disagree": 0, "untrusted": 0, "band": 0, "ev": 0}

                for g in active:
                  try:
                    gid = str(g.get("gameId"))
                    st = mg.NBALiveFeedClient.fetch_live_game_state(g)
                    if st.game_status == 2 and not st.live_box:
                        logger.warning(f"{st.matchup_title}: live box unavailable this cycle; skipping game.")
                        continue
                    if st.game_status == 2 and st.seconds_remaining_in_game < 45:
                        continue
                    token = game_date_token(st.game_code)
                    ok_prov, why = dt_provenance_ok(dt_state, st.home_team_id, st.away_team_id)
                    if not ok_prov:
                        if provenance_warned.get(gid) != why:
                            logger.error(f"{st.matchup_title}: D&T provenance check FAILED ({why}); skipping this game.")
                            provenance_warned[gid] = why
                        continue
                    hr, _ = mg.build_dt_rosters_and_fallback_lines(dt_state["epm"], dt_state["box"], st.home_team_id)
                    ar, _ = mg.build_dt_rosters_and_fallback_lines(dt_state["epm"], dt_state["box"], st.away_team_id)
                    team_of = {p.player_id: str(p.team_id) for p in list(hr) + list(ar)}
                    if len(hr) < 5 or len(ar) < 5:
                        logger.warning(f"{st.matchup_title}: D&T roster incomplete ({len(hr)}/{len(ar)}); skipping.")
                        continue
                    hd, ad = dt_state["team"].get(st.home_team_id, 0.0), dt_state["team"].get(st.away_team_id, 0.0)
                    if gid not in calib_cache:
                        calib_cache[gid] = mg.calibrate_to_dt(hr, ar, dt_state["box"], hd, ad, st.home_team_id, st.away_team_id)[:2]
                        bad = sum(1 for v in calib_cache[gid][1].values() if not v)
                        logger.info(f"{st.matchup_title}: calibrated to D&T box ({bad} player-stats flagged untradeable).")
                    calib, trusted = calib_cache[gid]

                    stars = {p.player_id for p in hr + ar
                             if p.p_mp >= args.min_minutes}
                    g_markets = [m for m in markets if m.player_id in stars and token and token in m.event_ticker.upper()]
                    if not g_markets:
                        continue

                    # Common random numbers: the same game state always gets the same simulated paths, so prices only
                    # move when the game does (a time-based seed made ~0.8c of pure noise look like new information).
                    skey = (st.game_id, st.pbp_context.last_action_number, int(st.seconds_remaining_in_game), st.period,
                            st.home_score, st.away_score, tuple(sorted(st.on_court_home)), tuple(sorted(st.on_court_away)), cal_id)
                    cache_hit = sim_cache.get(gid, (None,))[0] == skey
                    if cache_hit:
                        sim, sim_sh = sim_cache[gid][1], sim_cache[gid][2]
                    else:
                        seed = zlib.crc32(repr(skey).encode()) % 2_000_000_000
                        sim = mg.InPlayMarkovEngine(st, hr, ar, hd, ad, calib=calib).run_monte_carlo(n_sims=args.sims, seed=seed)
                        sim_sh = None
                        sim_cache[gid] = (skey, sim, None)
                        if args.shadow_sims > 0:
                            # A/B research engine (other matchup strength): run AFTER this cycle's orders are sent, so
                            # research never delays execution. Its result is cached for the same state.
                            shadow_todo[gid] = (st, hr, ar, hd, ad, seed, skey)
                    # Feed-latency breaker: many Kalshi prices jumped while our NBA state did not move -> the market
                    # knows something our feed has not shown yet. Pull quotes and sit this game out this cycle.
                    # Only liquid books count (wide-spread mids jump on their own), never during period breaks or other
                    # stoppages (the feed is legitimately frozen while Kalshi reprices), and it takes a broad move.
                    _tight = [_m for _m in g_markets if _m.yes_ask - _m.yes_bid <= args.max_spread and prev_mid.get(_m.ticker) is not None]
                    _moved = sum(1 for _m in _tight if abs(_m.mid - prev_mid[_m.ticker]) > args.max_mid_move)
                    _pc0 = st.seconds_remaining_in_game - max(0, 4 - st.period) * 720.0
                    _stoppage = st.game_status != 2 or _pc0 in (0.0, 720.0) or st.pbp_context.last_action_type in DEAD_BALL
                    if cache_hit and not _stoppage and _moved >= max(5, int(0.2 * len(_tight))):
                        _c = book.cancel_game(st.matchup_title, "feed_lag")
                        logger.warning(f"{st.matchup_title}: {_moved} Kalshi prices jumped with no new NBA event -> feed may be lagging; "
                                       f"skipping game this cycle (pulled {_c} quote(s)).")
                        n_gate["feed_stale"] += 1
                        continue
                    for (_t, _sd), _pos in positions.items():   # held exposures for the optimizer's covariance
                        _mm = next((x for x in g_markets if x.ticker == _t), None)
                        if _mm is None or _mm.player_id not in sim["id_to_idx"] or _pos["contracts"] < 1:
                            continue
                        _smp = sim[_mm.stat][:, sim["id_to_idx"][_mm.player_id]]
                        _ov = (_smp > _mm.line) if _sd == "yes" else (_smp < _mm.line)
                        held.append(dict(ticker=_t, side=_sd, game_key=st.matchup_title, outcome=_ov, stat=_mm.stat,
                                         team_id=team_of.get(_mm.player_id, _pos.get("team_id", "")), player_name=_mm.player_name,
                                         win_prob=float(np.clip(_ov.mean(), 0.01, 0.99)),
                                         price_cents=max(1.0, 100.0 * _pos["cost_usd"] / _pos["contracts"]), fee_cents=0.0,
                                         exposure_usd=max(0.0, _pos["cost_usd"])))
                    watch_titles[st.game_id] = st.matchup_title
                    cons = books.game(st.matchup_title) if books.enabled else {}
                    idx_of = sim["id_to_idx"]
                    late = st.period >= 4 and st.seconds_remaining_in_game <= 300.0
                    late_close = st.period >= 4 and st.seconds_remaining_in_game <= 120.0 and abs(st.home_score - st.away_score) <= 8
                    _pc = st.seconds_remaining_in_game - max(0, 4 - st.period) * 720.0
                    dead = st.game_status != 2 or _pc in (0.0, 720.0) or st.pbp_context.last_action_type in DEAD_BALL
                    ev_floor = args.min_ev + (0.0 if dead else args.live_ev_premium)
                    if st.pbp_context.last_action_number != watch_ids.get(st.game_id) or st.game_id not in act_seen_ts:
                        act_seen_ts[st.game_id] = time.time()
                    watch_ids[st.game_id] = max(watch_ids.get(st.game_id, 0), st.pbp_context.last_action_number)
                    _act = st.pbp_context.last_action_number
                    if not _act:   # state built without a play number: the feed's current one, or unknown (None = no orders)
                        _la0 = mg.NBALivePlayByPlayParser.latest_action(st.game_id, 0)
                        _act = _la0[0] if _la0 is not None else None
                    sim_act[st.matchup_title] = (st.game_id, _act)   # state version the prices came from

                    # ── pass 1: per market (gates, model, authority, logging) ─────────────────────────────────────
                    mk_ctx = []
                    n_cand0 = len(candidates)
                    phase_now = period_bucket(st.period, _pc)
                    stop_now = XM.stop_type_of(st.pbp_context.last_action_type)
                    t_act = max(0.0, time.time() - act_seen_ts.get(st.game_id, time.time()))
                    for m in g_markets:
                        funnel.add("markets_seen")
                        if m.player_id not in idx_of:
                            continue   # fouled out / inactive: not modelled
                        if m.ticker.split("-")[0] not in allowed_series:
                            n_gate["fee"] += 1
                            funnel.block("fee series")
                            continue
                        if not trusted.get((m.player_id, m.stat), False):
                            n_gate["untrusted"] += 1
                            funnel.block("untrusted player/stat")
                            continue
                        funnel.add("trusted")
                        if m.yes_ask - m.yes_bid > args.max_spread:
                            n_gate["spread"] += 1
                            funnel.block("spread")
                            continue
                        last_mid = prev_mid.get(m.ticker)
                        if (last_mid is None and not args.once) or (last_mid is not None and abs(m.mid - last_mid) > args.max_mid_move):
                            n_gate["unstable"] += 1
                            funnel.block("stale/unstable book")
                            continue
                        funnel.add("valid_book")
                        smp = sim[m.stat][:, idx_of[m.player_id]]
                        n = len(smp)
                        p_yes_raw = (float((smp > m.line).sum()) + 0.5) / (n + 1.0)
                        p_yes = calibrate(cal, m.stat, p_yes_raw)
                        _bk = prob_at_line(cons, mg.normalize_player_name(m.player_name), m.stat, m.line, smp) if cons else None
                        p_book_yes, n_books, book_dist = (_bk if _bk else (None, 0, None))
                        gate_book = (p_book_yes is not None and book_dist is not None and book_dist <= args.book_max_line_diff + 1e-9
                                     and n_books >= args.min_books)
                        mkt_yes = m.mid
                        in_scope = m.stat in trade_stats and st.period >= min_period
                        if in_scope and (dead or args.allow_live_ball) and args.micro and args.min_gap - 0.02 <= abs(p_yes - m.mid) <= args.max_disagree + 0.02:
                            _mp = kalshi.microprice(m.ticker)
                            if _mp is not None and m.yes_bid / 100.0 <= _mp <= m.yes_ask / 100.0:
                                mkt_yes = m.mid + max(-0.01, min(0.01, _mp - m.mid))   # depth may move the anchor at most 1c
                        gap_y = abs(p_yes - mkt_yes)
                        if mw_manual:
                            auth = dict(weight=float(np.clip(args.model_weight, 0.0, 0.35)), tier="PILOT" if args.pilot_live else "RESEARCH",
                                        confidence=1.0, cell_key="manual", source="manual --model-weight", w_sd=None)
                        elif authority:
                            auth = authority_for(m.stat, phase_now, gap_y, authority)
                        else:
                            auth = dict(weight=0.0, tier="RESEARCH", confidence=0.0, cell_key="none", source="no calibration", w_sd=None)
                        if args.pilot_live and auth["tier"] == "PRODUCTION":
                            auth = dict(auth, tier="PILOT")       # a pilot run is never production, whatever a cell earned
                        w_m = auth["weight"]
                        w_b = bw_for(m.stat, st.period, _pc)
                        p_bk_y = p_book_yes if gate_book else None
                        p_sh_yes = None
                        if sim_sh is not None and m.player_id in sim_sh["id_to_idx"]:
                            _ssm = sim_sh[m.stat][:, sim_sh["id_to_idx"][m.player_id]]
                            p_sh_yes = calibrate(cal, m.stat, (float((_ssm > m.line).sum()) + 0.5) / (len(_ssm) + 1.0))
                        pred_meta.append((gid, m.player_id, m.stat, m.line))
                        pred_rows.append((datetime.now().isoformat(timespec='seconds'), m.ticker, st.clock_display, m.player_name,
                                          m.stat, m.line, round(p_yes, 4), round(m.mid, 4), round(p_yes_raw, 4), cal_id,
                                          round(float(smp.mean()), 2), round(float(smp.std()), 2),
                                          round(float(sim['MIN_PROJ'].get(m.player_id, 0.0)), 1),
                                          sum(int(st.live_box.get(m.player_id, {}).get(k, 0)) for k in (('PTS', 'REB', 'AST') if m.stat == 'PRA' else (m.stat,))),
                                          mg.ENGINE_VERSION, None if p_book_yes is None else round(p_book_yes, 4), n_books,
                                          round(mkt_yes, 4), None if p_sh_yes is None else round(p_sh_yes, 4),
                                          m.yes_bid, m.yes_ask, st.period, bool(dead),
                                          phase_now, bool(st.live_box.get(m.player_id, {}).get("STARTER", False)),
                                          st.pbp_context.current_stint_minutes.get(m.player_id),
                                          int(st.live_box.get(m.player_id, {}).get("PF", 0)),
                                          int(st.pbp_context.completed_stints.get(m.player_id, 0)),
                                          round(float(sim.get("SIM_PACE", 0.0)), 1), round(float(sim.get("TARGET_PACE", 0.0) or 0.0), 1),
                                          auth["tier"], auth["cell_key"], round(w_m, 4), gap_bin_of_value(gap_y) or "gt10"))
                        if not in_scope:
                            n_gate["scope"] += 1
                            funnel.block("out of scope (stat / period)")
                        else:
                            funnel.add("in_scope")
                        if kalshi.ws is not None and in_scope:
                            ws_want.add(m.ticker)            # warm the WebSocket book early for every plausibly tradeable market
                        pu_yes = pool3(p_yes, mkt_yes, p_bk_y, w_m, w_b)
                        mk_ctx.append(dict(m=m, smp=smp, n=n, p_yes=p_yes, mkt_yes=mkt_yes, p_book_yes=p_book_yes, gate_book=gate_book,
                                           in_scope=in_scope, auth=auth, w_m=w_m, w_b=w_b, p_sh_yes=p_sh_yes, pu_yes=pu_yes,
                                           gap_y=gap_y))
                    # ── ladder coherence: weighted isotonic projection of the deployed YES probabilities ───────────
                    iso_delta: Dict[str, float] = {}
                    by_ps: Dict[tuple, list] = {}
                    for mc in mk_ctx:
                        by_ps.setdefault((mc["m"].player_id, mc["m"].stat), []).append(mc)
                    for _k, _lst in by_ps.items():
                        if len(_lst) < 2:
                            continue
                        _lst.sort(key=lambda x: x["m"].line)
                        _w = [1.0 / (max(1e-4, x["pu_yes"] * (1 - x["pu_yes"])) / max(1, x["n"]) + ((x["m"].yes_ask - x["m"].yes_bid) / 200.0) ** 2)
                              for x in _lst]
                        _proj = isotonic_decreasing([x["m"].line for x in _lst], [x["pu_yes"] for x in _lst], _w)
                        for x, pj in zip(_lst, _proj):
                            if abs(pj - x["pu_yes"]) > 1e-9:
                                iso_delta[x["m"].ticker] = pj - x["pu_yes"]
                                n_gate["ladder_projected"] = n_gate.get("ladder_projected", 0) + 1
                    # ── pass 2: per side ─────────────────────────────────────────────────────────────────────────
                    for mc in mk_ctx:
                        m, smp, n, p_yes, mkt_yes, auth = mc["m"], mc["smp"], mc["n"], mc["p_yes"], mc["mkt_yes"], mc["auth"]
                        w_m, w_b, p_sh_yes, in_scope = mc["w_m"], mc["w_b"], mc["p_sh_yes"], mc["in_scope"]
                        p_book_yes, gate_book = mc["p_book_yes"], mc["gate_book"]
                        dlt = iso_delta.get(m.ticker, 0.0)
                        for side, out_vec, p_mkt in (("yes", smp > m.line, mkt_yes), ("no", smp < m.line, 1.0 - mkt_yes)):
                            p_bk = None if not gate_book else (p_book_yes if side == "yes" else 1.0 - p_book_yes)
                            # EXIT: sell a held side when the bid (after fee) beats our fair value by the margin.
                            _held = positions.get((m.ticker, side))
                            if _held and not args.no_exits and _held["contracts"] >= 1 and \
                                    time.time() - _held.get("last_fill", 0.0) >= args.refill_cooldown:
                                _pm = p_yes if side == "yes" else 1.0 - p_yes
                                _bid = m.yes_bid if side == "yes" else m.no_bid
                                if abs(_pm - p_mkt) <= args.max_disagree and _bid >= 1:
                                    _fair = pool3(_pm, p_mkt, p_bk, w_m, w_b)
                                    _net = (_bid - kalshi.fee_cents(m.ticker, _bid, False)) / 100.0
                                    if _net >= _fair + args.exit_margin:
                                        exits.append((dict(ticker=m.ticker, game_key=st.matchup_title, player_id=m.player_id,
                                                           player_name=m.player_name, stat=m.stat, line=m.line, side=side,
                                                           price_cents=_bid, fee_cents=0.0, is_maker=False, p_model=_pm,
                                                           p_mkt=p_mkt, win_prob=_fair, roi=0.0, net=_net, team_id=team_of.get(m.player_id, "")),
                                                      m, int(_held["contracts"])))
                                        continue
                            held_other = (m.ticker, "no" if side == "yes" else "yes") in positions
                            _pos = positions.get((m.ticker, side))
                            if _pos and (time.time() - _pos.get("last_fill", 0.0) < args.refill_cooldown):
                                continue   # just filled here: wait for new information before adding
                            _ask_now = m.yes_ask if side == "yes" else m.no_ask
                            if _pos and _pos.get("last_px") and _ask_now < _pos["last_px"] - 3:
                                funnel.block("no averaging down")
                                continue   # market moved 3c+ against our last fill: never average down
                            if held_other or not in_scope:
                                continue
                            if late_close:
                                n_gate["late_close"] += 1
                                funnel.block("late close game (endgame not modelled)")
                                continue
                            if not dead and not args.allow_live_ball:
                                n_gate["live_ball"] += 1
                                funnel.block("live ball")
                                continue
                            p_model = p_yes if side == "yes" else 1.0 - p_yes
                            if abs(p_model - p_mkt) > args.max_disagree:
                                n_gate["disagree"] += 1
                                funnel.block("disagreement > cap")
                                continue
                            if abs(p_model - p_mkt) < args.min_gap:
                                n_gate["small_gap"] += 1
                                funnel.block("small gap")
                                continue
                            funnel.add("valid_gap")
                            if p_model <= p_mkt:
                                continue              # the model does not prefer this side
                            funnel.add("positive_model_signal")
                            if w_m <= 0.0:
                                n_gate["no_model_weight"] = n_gate.get("no_model_weight", 0) + 1
                                funnel.block(f"no local authority ({auth['source']})")
                                continue   # weight 0 = no authority for this cell: no fundamental directional entry
                            funnel.add("local_authority_nonzero")
                            tier = auth["tier"]
                            if tier == "RESEARCH" and run_is_live:
                                n_gate["research_tier"] = n_gate.get("research_tier", 0) + 1
                                funnel.block("RESEARCH tier (no live capital)")
                                continue
                            funnel.add("pilot_authority" if tier != "PRODUCTION" else "production_authority")
                            is_probe = False
                            if tier != "PRODUCTION" and side == "no":
                                n_gate["pilot_yes_only"] = n_gate.get("pilot_yes_only", 0) + 1
                                funnel.block("PILOT/RESEARCH: YES only")
                                continue
                            if not side_on[side]:
                                if side == "no" and probe_left > 0:
                                    is_probe = True      # one-contract, join-the-bid live probe (sized and capped below)
                                else:
                                    n_gate["side_off"] = n_gate.get("side_off", 0) + 1
                                    funnel.block(f"{side.upper()} side disabled")
                                    continue
                            if tier != "PRODUCTION" and player_exp.get(m.player_name, 0.0) - own_key_exp.get((m.ticker, side), 0.0) > 1e-9:
                                n_gate["pilot_player_held"] = n_gate.get("pilot_player_held", 0) + 1
                                funnel.block("PILOT: player already held")
                                continue
                            funnel.add("side_enabled")
                            # Sportsbook confirmation: never buy a side the books price BELOW Kalshi's own price.
                            if p_bk is not None and p_bk < p_mkt:
                                n_gate["books_disagree"] += 1
                                funnel.block("sportsbooks disagree")
                                continue
                            p_used = pool3(p_model, p_mkt, p_bk, w_m, w_b) + (dlt if side == "yes" else -dlt)
                            p_used = float(np.clip(p_used, 1e-4, 1 - 1e-4))
                            bid, ask = (m.yes_bid, m.yes_ask) if side == "yes" else (m.no_bid, m.no_ask)
                            p_sh = None if p_sh_yes is None or args.shadow_z <= 0 else (p_sh_yes if side == "yes" else 1.0 - p_sh_yes)
                            adm_sd = exec_prior[side][1] if (run_is_production or exec_prior_own[side] >= 10) else OLD_POLICY_PRIOR[side][1]
                            _tbw = kalshi.ws.fresh(m.ticker) if kalshi.ws is not None else None
                            _q_sb, _q_sa = ((_tbw[1], _tbw[3]) if side == "yes" else (_tbw[3], _tbw[1])) if _tbw else (0.0, 0.0)
                            ctx = dict(ticker=m.ticker, side=side, stat=m.stat, period=st.period, phase=phase_now, stop=stop_now,
                                       tier=tier, p_model=p_model, n=n, w_m=w_m, w_b=w_b, p_sh=p_sh, p_bk=p_bk,
                                       n_shadow=len(sim_sh[m.stat]) if sim_sh is not None else 0,
                                       w_sd=(auth.get("w_sd") if auth.get("w_sd") is not None else w_sd), adm_sd=adm_sd,
                                       need=ev_floor, late=late, is_probe=is_probe,
                                       join=thr_join or is_probe or (side == "no" and exec_prior[side][0] < -1.0),
                                       inv_x=inventory_skew_c(player_exp.get(m.player_name, 0.0), args.bankroll, args.player_cap, args.inv_skew_c),
                                       ttl=ttl_for(st.pbp_context.last_action_type, args.quote_ttl), t_since_action_s=t_act,
                                       qty_side_bid=_q_sb, qty_side_ask=_q_sa,
                                       micro=kalshi.ws.micro(m.ticker) if kalshi.ws is not None else {}, iso_delta=dlt)
                            if tier != "PRODUCTION":
                                ctx["need"] = 0.0
                            res, why_p = price_side(ctx, bid, ask, p_mkt, p_used, _tox)
                            if res is None or res.get("rejected"):
                                key_ = "exec_ev" if why_p == "execution_EV" else "ev"
                                n_gate[key_] = n_gate.get(key_, 0) + 1
                                funnel.block("execution EV <= 0" if why_p == "execution_EV" else "edge / LCB")
                                if why_p == "execution_EV" and res is not None:
                                    elig_rows.append(dict(time=datetime.now().isoformat(timespec="seconds"), ticker=m.ticker,
                                                          game=st.matchup_title, player=m.player_name, stat=m.stat, line=m.line,
                                                          side=side, quote_price=res["px"], p_model=round(p_model, 4), p_mkt=round(p_mkt, 4),
                                                          p_used=round(p_used, 4), period=st.period, phase=phase_now, spread=m.yes_ask - m.yes_bid,
                                                          dead_ball=bool(dead), engine=mg.ENGINE_VERSION, is_maker=True, tier=tier,
                                                          confidence=auth["confidence"], cell=auth["cell_key"], authority_weight=w_m,
                                                          gap_bin=gap_bin_of_value(mc["gap_y"]), passes_conservative=False,
                                                          exec_ev_c=round(res["qev"], 3), **POLICY))
                                continue
                            funnel.add("LCB_pass")
                            funnel.add("execution_EV_positive")
                            parts = res["parts"]
                            fill_prob = res["pf"] if res["pf"] is not None else 1.0
                            cand = {
                                "ticker": m.ticker, "game_key": st.matchup_title, "clock": st.clock_display,
                                "player_id": m.player_id, "player_name": m.player_name, "stat": m.stat, "line": m.line,
                                "side": side, "price_cents": res["px"], "fee_cents": res["fee"], "win_prob": p_used, "p_used": p_used,
                                "expected_edge_c": res["cond_edge_c"], "cond_edge_c": res["cond_edge_c"], "lcb_edge_c": res["lcb"],
                                "raw_edge_c": parts["nominal_c"] - parts["spread_c"], "w_m": w_m, "p_book": p_bk, "n_sims": n,
                                "p_shadow": p_sh, "exec_c": parts["exec_c"], "n_shadow": ctx["n_shadow"], "need": ctx["need"],
                                "exec_sd": parts["exec_sd_c"], "unit_c": res["px"] + res["fee"], "p_model": p_model, "p_mkt": p_mkt,
                                "roi": res["roi"], "is_maker": res["is_maker"], "outcome": out_vec, "spread_c": m.yes_ask - m.yes_bid,
                                "snap_bid": m.yes_bid, "snap_ask": m.yes_ask, "probe": is_probe, "quote_level": res["quote_level"],
                                "arm": res["arm"], "propensity": round(res["prop"], 4), "kelly_mean": res["kelly_mean"], "w_b": w_b,
                                "pi_b": json.dumps({str(k_): round(v_, 4) for k_, v_ in res["pi_b"].items()}), "greedy_px": res["greedy_px"],
                                "action_price": res["px"], "action_prob": round(res["prop"], 4),
                                "pfill": res["pf"], "fill_prob": fill_prob, "em_c": res["em"], "em_model_c": res["em_model"],
                                "em_prior_c": exec_prior[side][0], "qev_c": res["qev"], "qev_sd_c": res["qev_sd"], "alpha_c": 100.0 * (p_used - p_mkt),
                                "ttl": ctx["ttl"], "stop": stop_now, "phase": phase_now, "t_since_action_s": round(t_act, 1),
                                "team_id": team_of.get(m.player_id, ""), "period": st.period, "aggr": round(res["aggr"], 3),
                                "tier": tier, "confidence": (auth["confidence"] if tier == "PRODUCTION" else 1.0), "cell": auth["cell_key"],
                                "authority_weight": w_m, "gap_bin": gap_bin_of_value(mc["gap_y"]), "fee_c": res["fee"], "ctx": ctx,
                                "max_contracts": 1 if (res["arm"] != "base" or args.max_contracts == 1 or is_probe or tier != "PRODUCTION") else None}
                            cand["challenger_pf"], cand["challenger_qev"] = challenger_preds(ctx, bid, ask, res["px"], cand["alpha_c"], res["fee"])
                            candidates.append(cand)
                            mkt_of[(m.ticker, side)] = m
                            elig_rows.append(dict(time=datetime.now().isoformat(timespec="seconds"), ticker=m.ticker, game=st.matchup_title,
                                                  player=m.player_name, stat=m.stat, line=m.line, side=side, quote_price=res["px"],
                                                  fee_c=round(res["fee"], 3), p_model=round(p_model, 4), p_mkt=round(p_mkt, 4),
                                                  p_used=round(p_used, 4), p_book=None if p_bk is None else round(p_bk, 4),
                                                  net_edge_c=round(p_used * 100.0 - res["px"] - res["fee"], 3), period=st.period, phase=phase_now,
                                                  spread=m.yes_ask - m.yes_bid, dead_ball=bool(dead), engine=mg.ENGINE_VERSION,
                                                  cons_edge_c=round(res["lcb"], 3), spread_part_c=round(parts["spread_c"], 3),
                                                  model_unc_c=round(parts["model_c"], 3), p_shadow=None if p_sh is None else round(p_sh, 4),
                                                  is_maker=bool(res["is_maker"]), passes_conservative=True, tier=tier,
                                                  confidence=cand["confidence"], cell=auth["cell_key"], authority_weight=round(w_m, 4),
                                                  gap_bin=cand["gap_bin"], fill_prob=round(fill_prob, 4), cond_edge_c=round(res["cond_edge_c"], 3),
                                                  iso_delta=round(dlt, 4), **POLICY))
                    # Strike discipline: one player's strikes are one distribution. After the isotonic projection, offer the
                    # optimizer only the --max-strikes best strikes per (player, stat, side), ranked by expected realized
                    # alpha = P(fill) x conditional edge (not by LCB alone), so one view is not bought several times.
                    _mine = candidates[n_cand0:]
                    _groups: Dict[tuple, list] = {}
                    for _c in _mine:
                        _groups.setdefault((_c["player_id"], _c["stat"], _c["side"]), []).append(_c)
                    _keep = []
                    for _g in _groups.values():
                        _g.sort(key=lambda c_: -(c_["fill_prob"] * c_["cond_edge_c"]))
                        _keep += _g[:max(1, args.max_strikes)]
                        n_gate["ladder_dup"] = n_gate.get("ladder_dup", 0) + max(0, len(_g) - max(1, args.max_strikes))
                    _kept = {(c_["ticker"], c_["side"]) for c_ in _keep}
                    for _c in _mine:
                        if (_c["ticker"], _c["side"]) not in _kept:
                            for _er in elig_rows:
                                if _er.get("ticker") == _c["ticker"] and _er.get("side") == _c["side"] and _er.get("passes_conservative"):
                                    _er["passes_conservative"], _er["ladder_dropped"] = False, True
                    # Pilot discipline WITHIN the cycle (same rule the held-check applies across cycles): a pilot never shares a
                    # player with any other order. If a PRODUCTION candidate exists for the player the pilots yield; otherwise
                    # only the single best pilot for that player survives. Without this a pilot posted beside a same-player
                    # production quote would be cancelled next cycle as "player already held", losing its queue place.
                    _by_pl: dict = {}
                    for _c in _keep:
                        _by_pl.setdefault(_c["player_id"], []).append(_c)
                    _keep2 = []
                    for _pl in _by_pl.values():
                        _prod = [c_ for c_ in _pl if c_.get("tier") == "PRODUCTION"]
                        _pil = sorted([c_ for c_ in _pl if c_.get("tier") != "PRODUCTION"],
                                      key=lambda c_: -(c_["fill_prob"] * c_["cond_edge_c"]))
                        _keep2 += _prod + ([] if _prod else _pil[:1])
                        _ndrop = len(_pil) - (0 if _prod else min(1, len(_pil)))
                        if _ndrop > 0:
                            n_gate["pilot_player_held"] = n_gate.get("pilot_player_held", 0) + _ndrop
                    _kept2 = {(c_["ticker"], c_["side"]) for c_ in _keep2}
                    for _c in _keep:
                        if (_c["ticker"], _c["side"]) not in _kept2:
                            funnel.block("PILOT: player already held")
                            for _er in elig_rows:
                                if _er.get("ticker") == _c["ticker"] and _er.get("side") == _c["side"] and _er.get("passes_conservative"):
                                    _er["passes_conservative"], _er["pilot_player_dropped"] = False, True
                    candidates[n_cand0:] = _keep2
                  except Exception as e:
                    logger.exception(f"Game {g.get('gameId')}: error this cycle ({e}); skipping game.")

                for m in markets:
                    prev_mid[m.ticker] = m.mid

                for _b, _m, _cnt in exits:
                    logger.warning(f"[EXIT] {_b['game_key']} | {_b['player_name']} {_b['stat']} {_b['side'].upper()} {_b['line']} | "
                                   f"sell {_cnt}x @ {_b['price_cents']}¢ | worth {_b['win_prob'] * 100:.1f}¢ to us, "
                                   f"bid nets {_b['net'] * 100:.1f}¢ after fee")
                    cash += book.exit(_b, _m, _cnt)
                    time.sleep(0.25)
                if elig_rows:
                    csv_append(ELIGIBLE_CSV, [dict(r, run_id=RUN_ID) for r in elig_rows])
                candidates, n_gate["contradiction"] = drop_contradictions(candidates, positions)
                if n_gate["contradiction"]:
                    funnel.block("contradictory pair", n_gate["contradiction"])
                # Risk state = filled positions + pending + RESTING quotes. Resting quotes on a (ticker, side) the optimizer is
                # re-deciding this cycle are left out of that key's existing exposure: the optimizer returns the TOTAL target
                # for the key and sync_quotes reconciles the resting quantity to it (so a throttle can shrink it).
                cand_keys = {(c["ticker"], c["side"]) for c in candidates}
                risk_view = {k: dict(v) for k, v in positions.items()}
                for _src in (reserved, resting_cf, {k: v for k, v in resting_live.items() if k not in cand_keys}):
                    for k, v in _src.items():
                        risk_view.setdefault(k, {"contracts": 0.0, "cost_usd": 0.0, "game_key": v["game_key"], "player": v["player"],
                                                 "tier": v.get("tier", "PRODUCTION")})
                        risk_view[k]["cost_usd"] += v["cost_usd"]
                        risk_view[k]["contracts"] = risk_view[k].get("contracts", 0.0) + v.get("contracts", 0.0)
                if book.killed:
                    if candidates:
                        logger.warning(f"Kill switch active ({book.killed}): {len(candidates)} candidate(s) not traded.")
                        funnel.block(f"kill switch ({book.killed})", len(candidates))
                    candidates = []
                if thr_mult <= 0.0:
                    funnel.block("risk throttle: entries paused", len(candidates))
                    candidates = []
                if thr_no_add:
                    _held_players = {v.get("player") for v in positions.values()}
                    candidates = [c for c in candidates if c["player_name"] not in _held_players]

                def pilot_groups(cands):
                    """PILOT budget (v2026.37): every non-PRODUCTION candidate shares the pilot session budget
                    (--pilot-session-cap) and a per-player pilot budget (--pilot-player-cap), net of open pilot exposure
                    (filled + pending + resting), at full notional."""
                    open_p, open_pp = 0.0, {}
                    for _v in risk_view.values():
                        if _v.get("tier", "PRODUCTION") != "PRODUCTION":
                            open_p += max(0.0, _v.get("cost_usd", 0.0))
                            open_pp[_v.get("player", "")] = open_pp.get(_v.get("player", ""), 0.0) + max(0.0, _v.get("cost_usd", 0.0))
                    idx = [i for i, c in enumerate(cands) if c.get("tier") != "PRODUCTION"]
                    if not idx:
                        return []
                    out = [{"members": idx, "room": max(0.0, args.pilot_session_cap - open_p / args.bankroll)}]
                    by_pl: Dict[str, list] = {}
                    for i in idx:
                        by_pl.setdefault(cands[i]["player_name"], []).append(i)
                    for pl, ii in by_pl.items():
                        out.append({"members": ii, "room": max(0.0, args.pilot_player_cap - open_pp.get(pl, 0.0) / args.bankroll)})
                    return out

                orders = qp.solve_allocations(candidates, bankroll=args.bankroll, max_bet_usd=args.max_bet, positions=risk_view,
                                              held=held, group_caps=pilot_groups(candidates))
                funnel.add("QP_positive", qp.last_stats.get("positive", 0))
                funnel.add("below_one_contract", qp.last_stats.get("below_one", 0))
                funnel.add("integer_lifted", qp.last_stats.get("rounded_up", 0))
                if candidates and not orders:
                    funnel.block("QP / integer sizing / caps", len(candidates))
                funnel_d = dict(cand=len(candidates), qp_pos=qp.last_stats.get("positive", 0),
                                below_1=qp.last_stats.get("below_one", 0), lifted=qp.last_stats.get("rounded_up", 0),
                                sized=len(orders), pilot=sum(1 for o in orders if o.get("tier") != "PRODUCTION"))
                if candidates and qp.last_stats.get("below_one", 0) >= max(1, qp.last_stats.get("positive", 0)) // 2 + 1 and not qp.last_stats.get("rounded_up", 0):
                    logger.warning(f"BANKROLL SMALL FOR THIS EDGE REGIME: {qp.last_stats['below_one']}/{qp.last_stats['positive']} positive "
                                   f"allocations are under one contract (median wanted ${qp.last_stats.get('median_wanted_usd', 0):.2f} vs "
                                   f"median contract ${qp.last_stats.get('median_contract_usd', 0):.2f}); even one contract fails the risk test.")
                n_send = {"sent": 0, "state_moved": 0, "pbp_unavailable": 0, "book_moved": 0, "would_cross": 0, "outbid": 0,
                          "no_book": 0, "repriced_out": 0, "requoted": 0, "other": 0}
                # Phase A (v2026.37) - fresh book, FULL quote re-optimization, one re-solve. Every proposed order is re-read
                # against the live book (WebSocket when synchronized, else REST). If the mid moved more than
                # --revalidate-mid-c since pricing, or the book is one-sided/crossed, it is dropped (stale protection).
                # Otherwise the maker ladder is REBUILT on the fresh book and the quote re-chosen with the same rule as the
                # first pass (fresh P(fill), execution residual and QEV at every level), then the optimizer re-sizes.
                if orders and not args.no_revalidate:
                    refreshed = []
                    for o in orders:
                        tb = kalshi.top_of_book(o["ticker"])
                        if tb is None:
                            n_send["no_book"] += 1
                            continue
                        nb, na = int(round(tb[0])), int(round(tb[2]))
                        if not (1 <= nb < na <= 99):
                            n_send["other"] += 1
                            continue
                        if abs((nb + na) - (o["snap_bid"] + o["snap_ask"])) / 2.0 > args.revalidate_mid_c + 1e-9:
                            n_send["book_moved"] += 1
                            continue
                        fy = (nb + na) / 200.0
                        pm = fy if o["side"] == "yes" else 1.0 - fy
                        ctx = dict(o["ctx"])
                        ctx["force_arm"], ctx["force_prop"] = o["arm"], o["propensity"]
                        ctx["qty_side_bid"], ctx["qty_side_ask"] = (tb[1], tb[3]) if o["side"] == "yes" else (tb[3], tb[1])
                        ctx["book_age_s"] = 0.0
                        if kalshi.ws is not None:
                            ctx["micro"] = kalshi.ws.micro(o["ticker"])
                        pu = float(np.clip(pool3(o["p_model"], pm, o.get("p_book"), o["w_m"], o.get("w_b", book_w))
                                           + (ctx.get("iso_delta", 0.0) if o["side"] == "yes" else -ctx.get("iso_delta", 0.0)), 1e-4, 1 - 1e-4))
                        sbid, sask = side_book(nb, na, o["side"])
                        if o["is_maker"]:
                            res, _w = price_side(ctx, sbid, sask, pm, pu, _tox)
                        else:
                            res, _w = (price_side(ctx, sbid, sask, pm, pu, _tox) if sask == o["price_cents"] else (None, "ask changed"))
                        if res is None or res.get("rejected") or res.get("is_maker", True) != o["is_maker"]:
                            n_send["repriced_out"] += 1
                            continue
                        if res["px"] != o["price_cents"]:
                            n_send["requoted"] += 1
                        c = {k: v for k, v in o.items() if k not in ("contracts", "risk_usd", "ev_usd", "ev_if_filled_usd", "joint_kelly_fraction")}
                        c.update(p_mkt=pm, p_used=pu, win_prob=pu, lcb_edge_c=res["lcb"], expected_edge_c=res["cond_edge_c"],
                                 cond_edge_c=res["cond_edge_c"], price_cents=res["px"], fee_cents=res["fee"], fee_c=res["fee"],
                                 unit_c=res["px"] + res["fee"], roi=res["roi"], snap_bid=nb, snap_ask=na, qty_bid=tb[1], qty_ask=tb[3],
                                 book_ts=time.time(), pfill=res["pf"], fill_prob=res["pf"] if res["pf"] is not None else 1.0,
                                 em_c=res["em"], em_model_c=res["em_model"], qev_c=res["qev"], qev_sd_c=res["qev_sd"],
                                 quote_level=res["quote_level"], aggr=round(res["aggr"], 3), alpha_c=100.0 * (pu - pm),
                                 propensity=round(res["prop"], 4), action_prob=round(res["prop"], 4), action_price=res["px"],
                                 greedy_px=res["greedy_px"], pi_b=json.dumps({str(k_): round(v_, 4) for k_, v_ in res["pi_b"].items()}),
                                 side_bid=sbid, side_ask=sask, exec_c=res["parts"]["exec_c"], exec_sd=res["parts"]["exec_sd_c"],
                                 **{k: (ctx["micro"] or {}).get(k) for k in XM.MICRO_KEYS}, has_ws=bool((ctx["micro"] or {}).get("has_ws")))
                        c["challenger_pf"], c["challenger_qev"] = challenger_preds(ctx, sbid, sask, res["px"], c["alpha_c"], res["fee"])
                        refreshed.append(c)
                    orders = qp.solve_allocations(refreshed, bankroll=args.bankroll, max_bet_usd=args.max_bet,
                                                  positions=risk_view, held=held, group_caps=pilot_groups(refreshed)) if refreshed else []
                funnel.add("fresh_book_survivors", len(orders))
                # qp7 quantity-aware second-stage economics: the first QP solve proposes quantity; now quantity enters
                # the execution feature vector, expected fill FRACTION is re-scored, size authority is applied, and the
                # optimizer is solved again. Hard caps and CVaR below still use full requested notional.
                if orders:
                    _qrescored = []
                    for _o in orders:
                        _req = max(1, int(_o.get("contracts", 1)))
                        _allow = highest_authorized_qty(_req, size_authority, production=run_is_production)
                        if _allow < 1:
                            funnel.block("size bucket not authorized")
                            continue
                        _c = {k: v for k, v in _o.items() if k not in ("contracts", "risk_usd", "ev_usd", "ev_if_filled_usd", "joint_kelly_fraction")}
                        _c["requested_qty"] = int(_allow)
                        _c["size_bucket"] = size_bucket(_allow)
                        _c["max_contracts"] = min(_allow, int(_c["max_contracts"])) if _c.get("max_contracts") else _allow
                        _ctx = _c.get("ctx") or {}
                        if exec_model:
                            _sbid = int(_c.get("side_bid", _c.get("price_cents", 0)))
                            _sask = int(_c.get("side_ask", max(_sbid + 1, _c.get("price_cents", 0) + 1)))
                            _qa = float(_ctx.get("qty_side_bid", 0.0) if int(_c["price_cents"]) == _sbid else 0.0)
                            _xq = exec_features(_c["side"], _c["stat"], _c.get("period", 0), _sbid, _sask, int(_c["price_cents"]),
                                                float(_c.get("alpha_c", 0.0)), ttl_s=float(_c.get("ttl", 8.0)),
                                                stop=str(_c.get("stop", "other")), phase=_c.get("phase"),
                                                qty_bid=float(_ctx.get("qty_side_bid", 0.0)), qty_ask=float(_ctx.get("qty_side_ask", 0.0)),
                                                queue_ahead=_qa, t_since_action_s=float(_c.get("t_since_action_s", 0.0)),
                                                micro=_ctx.get("micro") or {}, order_qty=_allow,
                                                best_depth=float(_ctx.get("qty_side_bid", 0.0)))
                            _qv = XM.quote_value(exec_model, _xq, float(_c.get("alpha_c", 0.0)), float(_c.get("fee_cents", 0.0)))
                            _c["fill_prob"] = float(_qv["pf"])
                            _c["effective_fill_fraction"] = float(_qv.get("fill_fraction", _qv["pf"]))
                        else:
                            _c["effective_fill_fraction"] = float(_c.get("fill_prob", 1.0))
                        _depth = max(float(_ctx.get("qty_side_bid", 0.0)), 1.0)
                        _queue = max(float(_ctx.get("qty_side_bid", 0.0) if int(_c["price_cents"]) == int(_c.get("side_bid", _c["price_cents"])) else 0.0), 1.0)
                        _c["qty_over_best_depth"] = _allow / _depth
                        _c["qty_over_queue_ahead"] = _allow / _queue
                        _c["log_order_qty"] = math.log1p(_allow)
                        _c["pi_b_action"] = json.dumps({action_key(int(_c["price_cents"]), _allow, float(_c.get("ttl", 8.0))): float(_c.get("action_prob", _c.get("propensity", 1.0)))})
                        _qrescored.append(_c)
                    orders = qp.solve_allocations(_qrescored, bankroll=args.bankroll, max_bet_usd=args.max_bet,
                                                  positions=risk_view, held=held, group_caps=pilot_groups(_qrescored)) if _qrescored else []
                    for _o in orders:
                        _o["requested_qty"] = int(_o["contracts"])
                        _o["size_bucket"] = size_bucket(_o["contracts"])
                # The $ minimum exists to avoid paying a rounded-up fee on dust; it only applies where a fee is charged.
                orders = [o for o in orders if o["fee_cents"] <= 0
                          or o["contracts"] * (o["price_cents"] + o["fee_cents"]) / 100.0 >= args.min_order]
                # Tail-loss control: scenario CVaR95 of held + pending + resting (not replaced) + proposed <= limit, with
                # shared model-error shocks (global, stat, game, team, player) on top of the simulated basketball outcomes.
                if orders:
                    _okeys = {(o["ticker"], o["side"]) for o in orders}
                    _others = [dict(outcome=h["outcome"], contracts=h["exposure_usd"] / max(0.01, h["price_cents"] / 100.0),
                                    unit_c=h["price_cents"], stat=h.get("stat"), game_key=h.get("game_key"), team_id=h.get("team_id"),
                                    player_name=h.get("player_name")) for h in held]
                    for _b in snap_pending + [b for b in snap_resting if (b["ticker"], b["side"]) not in _okeys
                                              or b.get("state") not in (RESTING, PARTIALLY_FILLED)]:
                        _rem = max(0.0, float(_b.get("count", 0)) - float(_b.get("filled", 0.0)))
                        if _rem > 0:
                            _others.append(dict(outcome=outcome_of(_b), contracts=_rem, stat=_b.get("stat"), game_key=_b.get("game_key"),
                                                team_id=_b.get("team_id"), player_name=_b.get("player_name"),
                                                unit_c=float(_b.get("price_cents", 0)) + float(_b.get("fee_cents", 0))))
                    for o in orders:
                        o.setdefault("unit_c", o["price_cents"] + o["fee_cents"])
                    orders, _cv = cvar_scale(orders, _others, args.bankroll, args.cvar_limit, shock_sds=shock_sds)
                    if _cv.get("alpha", 1.0) < 1.0:
                        funnel.block("CVaR tail limit")
                        logger.warning(f"TAIL LIMIT: new sizes scaled x{_cv['alpha']:.2f} (CVaR95 ${_cv['cvar95']}, 99th pct ${_cv['p99']} "
                                       f"vs limit ${args.cvar_limit * args.bankroll:.2f}){' - ' + _cv['note'] if _cv.get('note') else ''}")
                funnel.add("CVaR_survivors", len(orders))
                # Kept quotes get the same state check as new sends: before any resting quote keeps its queue place, its
                # game must be verifiably unchanged since pricing (unreadable feed = unknown = cancel). Fail closed.
                if orders and not args.no_revalidate:
                    for gk in sorted({o["game_key"] for o in orders}):
                        _gid, _act = sim_act.get(gk, (None, None))
                        _la = mg.NBALivePlayByPlayParser.latest_action(_gid, _act) if (_gid is not None and _act is not None) else None
                        if _la is None or _la[0] > _act:
                            _why = "pbp_unavailable_presync" if _la is None else "state_moved_presync"
                            _c = book.cancel_game(gk, _why)
                            n_send[_why] = n_send.get(_why, 0) + sum(1 for o in orders if o["game_key"] == gk)
                            orders = [o for o in orders if o["game_key"] != gk]
                            funnel.block("NBA state moved / feed unreadable before send")
                            logger.info(f"[STALE] {gk}: {_why}; {_c} resting pulled, its orders dropped before sync")
                for o in orders:
                    if o.get("probe") or o.get("max_contracts") == 1:
                        # NO probes, exploration arms and every PILOT order: exactly one contract
                        o["contracts"] = 1; o["risk_usd"] = round((o["price_cents"] + o["fee_cents"]) / 100.0, 2)
                funnel_d["after_checks"] = len(orders)
                if kalshi.ws is not None:
                    ws_want |= {c["ticker"] for c in candidates} | {b["ticker"] for b in snap_resting}
                    kalshi.ws.track(ws_want)
                    with book.lock:
                        _live_tk = {b["ticker"] for b in list(book.resting.values()) + list(book.pending.values())}
                    _drop = set(kalshi.ws.want) - ws_want - _live_tk \
                        - {r["ticker"] for r in book.fresh_markouts()}
                    if _drop and cycle % 4 == 0:
                        kalshi.ws.untrack(_drop)       # closed markets, finished games, players no longer in scope
                orders, n_kept = book.sync_quotes(orders)
                open_risk = sum(v["cost_usd"] for v in positions.values()) + sum(v["cost_usd"] for v in book.reserved().values())
                logger.info(f"Cycle #{cycle}: {len(markets)} markets | {len(candidates)} +EV candidates | {len(orders)} new orders, {n_kept} kept resting | "
                            f"open risk ${open_risk:.2f} | funnel {funnel_d} | gates {n_gate} | {time.time() - t_cycle:.1f}s")

                # Phase B - the NBA state is re-checked immediately before EVERY order (never cached per game: a play can
                # land between two sends), and an unreadable feed counts as unknown state: no new exposure (fail closed).
                for bet in orders:
                    count = bet["contracts"]
                    unit = (bet["price_cents"] + bet["fee_cents"]) / 100.0
                    if count * unit > cash:
                        count = int(cash // unit)
                    if count < 1:
                        logger.info("Sports shard cash exhausted for this cycle.")
                        funnel.block("cash exhausted")
                        break
                    if bet["fee_cents"] > 0 and count * unit < args.min_order:
                        continue
                    if not args.no_revalidate:
                        gk = bet["game_key"]
                        _gid, _act = sim_act.get(gk, (None, None))
                        _la = mg.NBALivePlayByPlayParser.latest_action(_gid, _act) if (_gid is not None and _act is not None) else None
                        if _la is None:
                            n_send["pbp_unavailable"] += 1
                            funnel.block("NBA feed unreadable at send")
                            book.cancel_game(gk, "pbp_unavailable")
                            continue
                        if _la[0] > _act:
                            n_send["state_moved"] += 1
                            funnel.block("NBA state moved at send")
                            _c = book.cancel_game(gk, "state_moved_presend")
                            logger.info(f"[STALE] {gk}: new play since pricing (#{_act} -> #{_la[0]}); order skipped, {_c} resting pulled")
                            continue
                        bet["act_submit"] = _act
                    if bet.get("probe") and probe_left <= 0:
                        funnel.block("NO probe limit reached")
                        continue                 # --no-probe N is a per-run limit (review fix)
                    funnel.add("PBP_survivors")
                    if bet.get("probe"):
                        probe_left -= 1
                        logger.info(f"[NO PROBE] {probe_left} left this run")
                    logger.warning(
                        f"[+EV {bet.get('tier', '?')}] {bet['game_key']} {bet['clock']} | {bet['player_name']} {bet['stat']} {bet['side'].upper()} {bet['line']} | "
                        f"{'MAKER' if bet['is_maker'] else 'TAKER'} {count}x @ {bet['price_cents']}¢ (+{bet['fee_cents']:.2f}¢ fee) | "
                        f"model {bet['p_model']:.3f} mkt {bet['p_mkt']:.3f} used {bet['win_prob']:.3f} | cell {bet.get('cell')} conf {bet.get('confidence', 1):.2f} | "
                        f"edge if filled {bet['expected_edge_c']:+.2f}c x P(fill) {bet.get('fill_prob', 1.0):.2f} | "
                        f"Kelly {bet['joint_kelly_fraction'] * 100:.2f}% | EV ${bet['ev_usd']:.2f}")
                    # exchange-side expiry is the backstop a few seconds after the state-dependent life
                    _committed = book.place(bet, mkt_of[(bet["ticker"], bet["side"])], count, int(min(ttl, int(bet.get("ttl") or ttl) + 3)))
                    if _committed > 0:
                        n_send["sent"] += 1
                        funnel.add("quotes_posted")
                    cash -= _committed
                    time.sleep(0.25)
                funnel.posted(n_send["sent"])
                if orders or any(n_send.values()):
                    logger.info(f"Pre-send check: {n_send}")
                try:
                    csv_append(FUNNEL_CSV, [funnel.row(RUN_ID, cycle)])
                except Exception:
                    pass
                _starve = funnel.starvation()
                if _starve:
                    logger.warning(_starve)

                # Research after execution: the A/B engine at the other matchup strength, then the prediction log.
                t_sh = time.time()
                for _gid, (_st, _hr, _ar, _hd, _ad, _seed, _skey) in shadow_todo.items():
                    try:
                        mg.MATCHUP_STRENGTH = shadow_strength
                        if _gid not in calib_shadow:
                            calib_shadow[_gid] = mg.calibrate_to_dt(_hr, _ar, dt_state["box"], _hd, _ad, _st.home_team_id, _st.away_team_id)[0]
                        _sh = mg.InPlayMarkovEngine(_st, _hr, _ar, _hd, _ad, calib=calib_shadow[_gid]).run_monte_carlo(
                            n_sims=args.shadow_sims, seed=_seed)
                        if sim_cache.get(_gid, (None,))[0] == _skey:
                            sim_cache[_gid] = (_skey, sim_cache[_gid][1], _sh)
                    except Exception as e:
                        logger.warning(f"Shadow engine failed ({e}).")
                    finally:
                        mg.MATCHUP_STRENGTH = args.matchup_strength
                if pred_rows:
                    _ix = PRED_COLS.index("p_shadow")
                    for _k, (_gid, _pid, _stat, _line) in enumerate(pred_meta):
                        _sh = sim_cache.get(_gid, (None, None, None))[2]
                        if pred_rows[_k][_ix] is None and _sh is not None and _pid in _sh["id_to_idx"]:
                            _v = _sh[_stat][:, _sh["id_to_idx"][_pid]]
                            _row = list(pred_rows[_k])
                            _row[_ix] = round(calibrate(cal, _stat, (float((_v > _line).sum()) + 0.5) / (len(_v) + 1.0)), 4)
                            pred_rows[_k] = tuple(_row)
                    csv_append(PRED_LOG, pd.DataFrame(pred_rows, columns=PRED_COLS).to_dict("records"))   # keeps an existing header
                if shadow_todo:
                    logger.info(f"Research shadow (matchup strength {shadow_strength:g}) for {len(shadow_todo)} game(s) "
                                f"after orders: {time.time() - t_sh:.1f}s")

            except Exception as e:
                logger.exception(f"Cycle #{cycle} failed ({e}); continuing.")
            if args.once:
                break
            # Between cycles: poll play-by-play; pull resting orders on any player in a new play.
            _end = t_cycle + args.interval
            while time.time() < _end - 0.2:
                if (book.resting or book.fresh_markouts()) and watch_ids:
                    for _gid in list(watch_ids):
                        try:
                            _la = mg.NBALivePlayByPlayParser.latest_action(_gid, watch_ids[_gid])
                        except Exception:
                            _la = None
                        if _la is None:            # unreadable feed = unknown state: no resting exposure in this game
                            _c = book.cancel_game(watch_titles.get(_gid, ""), "pbp_unavailable")
                            if _c:
                                logger.warning(f"[PULL] play-by-play unreadable for {watch_titles.get(_gid, _gid)}: cancelled {_c} resting order(s)")
                            continue
                        _num = _la[0]
                        _new = _num > watch_ids[_gid]
                        if _new:
                            act_seen_ts[_gid] = time.time()
                        watch_ids[_gid] = _num
                        _c = book.cancel_game(watch_titles.get(_gid, "")) if _new else 0
                        if _c:
                            logger.info(f"[PULL] new play in {watch_titles.get(_gid, _gid)}: cancelled {_c} resting order(s)")
                    try:
                        book.retry_failed_cancels()
                        _fc = book.fast_checks()
                        if _fc:
                            logger.info(f"[PULL] book changed under resting quotes: {_fc}")
                    except Exception as e:
                        logger.warning(f"fast checks failed: {e}")
                    try:
                        book.mark_fresh()          # 5s / 15s / 30s markouts from the live book, with play-in-between flags
                    except Exception as e:
                        logger.debug(f"markout refresh failed: {e}")
                time.sleep(max(0.2, min(args.watch, _end - time.time())))

    except KeyboardInterrupt:
        logger.info("Shutting down: cancelling resting orders...")
    finally:
        try:
            if kalshi.ws is not None:
                kalshi.ws.on_update = None
            book.reconcile({})
        except Exception as e:
            logger.error(f"Final reconcile failed: {e} -- check open orders in the Kalshi UI.")
        write_clv_report(prev_mid)
        logger.info("Trader stopped.")


def write_clv_report(last_mid: Dict[str, float]):
    """Closing-line value: fill price vs the last market mid we observed for that ticker (side-adjusted)."""
    if not os.path.exists(LEDGER_CSV):
        return
    df = pd.read_csv(LEDGER_CSV)
    if df.empty:
        return
    df = df[df["contracts"] > 0].copy()   # CLV is measured on entries
    mid = df["ticker"].map(last_mid)
    df["last_mid_cents"] = np.where(df["side"] == "yes", mid, 1.0 - mid) * 100.0
    df["clv_cents"] = df["last_mid_cents"] - df["price_cents"]
    out = LEDGER_CSV.replace("kalshi_fills_", "kalshi_clv_")
    df.to_csv(out, index=False)
    ok = df.dropna(subset=["clv_cents"])
    if len(ok):
        w = ok["contracts"]
        logger.info(f"Last-mid drift (NOT execution CLV: basketball happened in between; use kalshi_markouts_*.csv): "
                    f"{len(ok)} fills, contract-weighted avg {np.average(ok['clv_cents'], weights=w):+.2f}c vs last mid, "
                    f"{(ok['clv_cents'] > 0).mean():.0%} of fills beat it, total fees ${df['fee_usd'].sum():.2f} -> {out}")


if __name__ == "__main__":
    main()