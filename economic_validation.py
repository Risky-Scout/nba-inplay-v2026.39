"""Offline paired probability/policy evaluation. Never submits orders or grants authority.

Inputs are immutable, as-of predictions and timestamped aggressor trades; see
VALIDATION.md for the schema, estimand, queue assumptions and evidence limits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from risk_metrics import expected_shortfall

TIME_COLUMNS = ("timestamp", "state_available_at", "label_available_at",
                "baseline_trained_through", "challenger_trained_through")
REQUIRED = (*TIME_COLUMNS, "game", "player", "stat", "ticker", "side", "y_yes",
            "p_baseline", "p_challenger", "price_cents", "fee_cents", "bid", "ask",
            "expected_fill_fraction", "requested_qty", "queue_ahead", "ttl_s",
            "book_age_s", "data_kind", "baseline_engine", "challenger_engine")


def prepare(frame):
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"missing opportunity fields: {sorted(missing)}")
    d = frame.copy()
    if d.empty:
        raise ValueError("no opportunity records")
    for col in TIME_COLUMNS:
        # Naive timestamps cannot establish the as-of boundary.
        if any(pd.Timestamp(v).tzinfo is None for v in d[col]):
            raise ValueError(f"timezone required for {col}")
        d[col] = pd.to_datetime(d[col], utc=True, errors="raise")
    for col in ("game", "player", "stat", "ticker", "baseline_engine", "challenger_engine"):
        if d[col].isna().any() or d[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing identity: {col}")
    if any(d[col].nunique() != 1 for col in ("baseline_engine", "challenger_engine")):
        raise ValueError("each evaluation arm requires one exact engine identity")
    if not d.side.isin(["yes", "no"]).all() or not d.y_yes.isin([0, 1]).all():
        raise ValueError("binary y_yes and yes/no sides required")
    if not d.data_kind.isin(["historical", "synthetic"]).all() or d.data_kind.nunique() != 1:
        raise ValueError("one explicit data_kind (historical or synthetic) required")
    nums = ["p_baseline", "p_challenger", "price_cents", "fee_cents", "bid", "ask",
            "expected_fill_fraction", "requested_qty", "queue_ahead", "ttl_s", "book_age_s"]
    d[nums] = d[nums].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(d[nums].to_numpy()).all():
        raise ValueError("non-finite opportunity feature")
    for col in ("p_baseline", "p_challenger", "expected_fill_fraction"):
        if not d[col].between(0, 1).all():
            raise ValueError(f"{col} outside [0, 1]")
    if not ((d.price_cents >= 1) & (d.price_cents < d.ask) & (d.price_cents <= 99) &
            (d.bid > 0) & (d.bid < d.ask) & (d.ask <= 100) & (d.fee_cents >= 0) &
            (d.requested_qty >= 1) & (d.queue_ahead >= 0) & (d.ttl_s > 0) & (d.book_age_s >= 0)).all():
        raise ValueError("invalid maker book, size, fee or TTL")
    for col in ("price_cents", "requested_qty"):
        if not (d[col] == np.floor(d[col])).all():
            raise ValueError(f"integer {col} required")
    if not ((d.state_available_at <= d.timestamp) & (d.baseline_trained_through < d.timestamp) &
            (d.challenger_trained_through < d.timestamp) &
            (d.label_available_at > d.timestamp + pd.to_timedelta(d.ttl_s, unit="s"))).all():
        raise ValueError("as-of or label-availability violation")
    # All snapshots/thresholds of one contract must share its outcome and release time.
    for col in ("game", "player", "stat", "y_yes", "label_available_at"):
        if (d.groupby("ticker")[col].nunique() > 1).any():
            raise ValueError(f"inconsistent contract {col}")
    # A game cannot straddle train/test dates merely because it passed midnight.
    d["date"] = d.timestamp.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    game_day = d.groupby("game").date.transform("min")
    d["date"] = game_day
    return d.sort_values(["timestamp", "ticker", "side"], kind="stable").reset_index(drop=True)


def independent_opportunities(d):
    """First state per game/player/stat; choose best logged action at that state ex ante.

    One family includes both sides and every strike. No outcome, realized fill or
    challenger prediction enters this selection, so paired comparisons share support.
    """
    d = d.copy()
    family = ["game", "player", "stat"]
    first = d.groupby(family).timestamp.transform("min")
    d = d[d.timestamp == first].copy()
    p = np.where(d.side.eq("yes"), d.p_baseline, 1-d.p_baseline)
    cost = (d.price_cents + np.ceil(d.fee_cents)) / 100
    d["opportunity_weight"] = np.maximum(0, p-cost-0.03) * d.expected_fill_fraction * d.requested_qty
    d["opportunity_weight"] /= np.maximum(cost, 0.01)
    d["opportunity_weight"] = d.opportunity_weight.clip(upper=1.0)
    d = d.sort_values(["opportunity_weight", "ticker", "side"], ascending=[False, True, True], kind="stable")
    return d.drop_duplicates(family).sort_values("timestamp", kind="stable").reset_index(drop=True)


def walk_forward(d, min_train_games=20):
    """Identity-shrunk logit calibration fitted only to earlier, already settled games."""
    d = d.copy()
    d["p_candidate"] = d.p_challenger
    d["calibration_train_games"] = 0
    folds = []
    for day in sorted(d.date.unique()):
        test = d.date == day
        cutoff = d.loc[test, "timestamp"].min()
        train = d[(d.date < day) & (d.label_available_at < cutoff) &
                  ~d.game.isin(d.loc[test, "game"])]
        ng = train.game.nunique()
        a, b = 0.0, 1.0
        if ng >= min_train_games:
            x = logit(train.p_challenger.clip(1e-6, 1-1e-6).to_numpy())
            y = train.y_yes.to_numpy(float)
            # Equal game mass; opportunity weighting cannot depend on realized returns.
            w = 1 / train.groupby("game").game.transform("size").to_numpy(float)
            def obj(theta):
                z = theta[0] + theta[1]*x
                return float(np.sum(w*(np.logaddexp(0, z)-y*z)) + 5*(theta[0]**2+(theta[1]-1)**2))
            fit = minimize(obj, [0, 1], bounds=[(-2, 2), (0.25, 2)], method="L-BFGS-B")
            if fit.success:
                a, b = map(float, fit.x)
        d.loc[test, "p_candidate"] = expit(a+b*logit(d.loc[test, "p_challenger"].clip(1e-6, 1-1e-6)))
        d.loc[test, "calibration_train_games"] = ng
        folds.append(dict(test_date=day, train_games=int(ng), intercept=a, slope=b,
                          max_label_available_at=None if train.empty else train.label_available_at.max().isoformat()))
    return d, folds


def prepare_trades(trades):
    required = {"trade_id", "timestamp", "ticker", "side", "price_cents", "quantity", "aggressor"}
    if not required.issubset(trades):
        raise ValueError(f"missing trade fields: {sorted(required-set(trades))}")
    t = trades.copy()
    if any(pd.Timestamp(v).tzinfo is None for v in t.timestamp):
        raise ValueError("timezone required for trade timestamps")
    t["timestamp"] = pd.to_datetime(t.timestamp, utc=True)
    if t.trade_id.isna().any() or t.trade_id.duplicated().any():
        raise ValueError("unique trade_id required (duplicates inflate fills)")
    if not t.side.isin(["yes", "no"]).all() or not t.aggressor.isin(["buy", "sell"]).all():
        raise ValueError("explicit side and aggressor required")
    for c in ("price_cents", "quantity"):
        t[c] = pd.to_numeric(t[c], errors="raise")
        if not np.isfinite(t[c]).all() or not (t[c] == np.floor(t[c])).all():
            raise ValueError("finite integer trade prices and sizes required")
    if not t.price_cents.between(1, 99).all() or not (t.quantity > 0).all():
        raise ValueError("invalid trade price/quantity")
    return t.sort_values(["timestamp", "trade_id"], kind="stable")


def replay_fill(row, quantity, trades, latency_ms=500):
    """Conservative FIFO replay: exact-price sell-aggressor prints after arrival.

    Cancellations never improve our queue; touches, better-price prints and prints
    before arrival/at expiry never fill us. Assumes small, non-impacting orders.
    """
    arrival = row.timestamp + pd.Timedelta(milliseconds=latency_ms)
    expiry = row.timestamp + pd.Timedelta(seconds=row.ttl_s)
    t = trades[(trades.ticker == row.ticker) & (trades.side == row.side) &
               (trades.aggressor == "sell") & (trades.price_cents == row.price_cents) &
               (trades.timestamp > arrival) & (trades.timestamp < expiry)]
    return int(min(quantity, max(0, math.floor(t.quantity.sum()-row.queue_ahead))))


def policy_replay(d, trades, p_col, bankroll=1000.0, latency_ms=500, fee_mult=1.0,
                  model_haircut=0.03, fill_scale=1.0):
    """Fixed 0.1-Kelly, 1% player, 3% game, 10% session, 3% drawdown stop.

    Full quote notional is reserved until expiry, actual fills then remain reserved
    until label availability. Fractions/future fills never increase initial capacity.
    """
    equity, peak = bankroll, bankroll
    pending, outcomes = [], []
    equity_path = [bankroll]
    reserved = {}
    daily = {}
    def advance(now):
        nonlocal equity, peak
        due = sorted([v for v in pending if v["at"] <= now], key=lambda v: v["at"])
        for event in due:
            key = event["key"]
            if event["kind"] == "expiry":
                reserved[key]["cost"] = event["filled_cost"]
            else:
                equity += event["pnl"]
                peak = max(peak, equity)
                equity_path.append(equity)
                daily[event["day"]] = daily.get(event["day"], 0.0)+event["pnl"]
                reserved.pop(key, None)
            pending.remove(event)
    for key, r in d.iterrows():
        advance(r.timestamp)
        q = 0
        p = float(r[p_col]) if r.side == "yes" else 1-float(r[p_col])
        fee_c = math.ceil(r.fee_cents*fee_mult)
        unit = (r.price_cents+fee_c)/100
        fraction = 0.1*max(0, (p-model_haircut-unit)/max(1-unit, 1e-6))
        open_total = sum(v["cost"] for v in reserved.values())
        game_open = sum(v["cost"] for v in reserved.values() if v["game"] == r.game)
        player_open = sum(v["cost"] for v in reserved.values() if v["player"] == r.player)
        room = max(0, min(.1*equity-open_total, .03*equity-game_open, .01*equity-player_open))
        if equity > 0 and equity >= .97*peak and r.book_age_s <= 2:
            q = min(int(r.requested_qty), max(0, math.floor(min(room, fraction*equity)/unit)))
        fill = int(math.floor(fill_scale*replay_fill(r, q, trades, latency_ms))) if q else 0
        won = r.y_yes if r.side == "yes" else 1-r.y_yes
        pnl = fill*(won-unit)
        outcomes.append(dict(date=r.date, game=r.game, requested=q, filled=fill, cost=fill*unit, pnl=pnl))
        if q:
            reserved[key] = dict(cost=q*unit, game=r.game, player=r.player)
            pending.append(dict(key=key, kind="expiry", at=r.timestamp+pd.Timedelta(seconds=r.ttl_s), filled_cost=fill*unit))
            day = r.label_available_at.tz_convert("America/New_York").strftime("%Y-%m-%d")
            pending.append(dict(key=key, kind="settle", at=r.label_available_at, pnl=pnl, day=day))
    advance(pd.Timestamp.max.tz_localize("UTC"))
    out = pd.DataFrame(outcomes)
    path = np.asarray(equity_path)
    daily_pnl = np.array([daily.get(k, 0.0) for k in sorted(set(d.date) | set(daily))])
    metrics = dict(net_pnl_usd=float(out.pnl.sum()), filled_contracts=int(out.filled.sum()),
                   requested_contracts=int(out.requested.sum()), invested_usd=float(out.cost.sum()),
                   roi=None if out.cost.sum() == 0 else float(out.pnl.sum()/out.cost.sum()),
                   max_settled_drawdown_usd=float(np.max(np.maximum.accumulate(path)-path)),
                   daily_cvar95_usd=None if not len(daily_pnl) else expected_shortfall(-daily_pnl))
    return out, metrics


def paired_interval(d, value, n_boot=2000, seed=41):
    """Whole-date bootstrap: games/players/strikes on a date move together."""
    by = d.assign(value=value).groupby("date").value.sum().to_numpy(float)
    if len(by) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(by, len(by), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(by.mean()), lo95=float(lo), hi95=float(hi), date_blocks=int(len(by)))


def paired_ratio_interval(d, numerator, denominator, n_boot=2000, seed=41):
    """Date-block ratio bootstrap, preserving ex-ante weights and paired support."""
    by = d.assign(numerator=numerator, denominator=denominator).groupby("date")[["numerator", "denominator"]].sum()
    if len(by) < 2 or by.denominator.sum() <= 0:
        return None
    sums = by.to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = sums[rng.integers(0, len(sums), (n_boot, len(sums)))].sum(axis=1)
    samples = samples[samples[:, 1] > 0]
    if len(samples) < n_boot // 2:
        return None
    lo, hi = np.quantile(samples[:, 0]/samples[:, 1], [.025, .975])
    return dict(mean=float(by.numerator.sum()/by.denominator.sum()), lo95=float(lo), hi95=float(hi), date_blocks=len(by))


def evaluate(frame, trade_frame, bankroll=1000.0):
    if not math.isfinite(bankroll) or bankroll <= 0:
        raise ValueError("positive finite bankroll required")
    raw = prepare(frame)
    d, folds = walk_forward(independent_opportunities(raw))
    trades = prepare_trades(trade_frame)
    base, bm = policy_replay(d, trades, "p_baseline", bankroll)
    cand, cm = policy_replay(d, trades, "p_candidate", bankroll)
    losses = {}
    for label, col in (("baseline", "p_baseline"), ("challenger", "p_candidate")):
        b = (d[col]-d.y_yes)**2
        w = d.opportunity_weight
        losses[label] = dict(brier=float(b.mean()), opportunity_weighted_brier=None if w.sum() == 0 else float(np.sum(w*b)/w.sum()))
    stress = {}
    for label, kw in (("latency_2s", dict(latency_ms=2000)), ("double_fees", dict(fee_mult=2)),
                      ("half_fills", dict(fill_scale=.5)), ("model_error_8pct", dict(model_haircut=.08))):
        _, stress[label] = policy_replay(d, trades, "p_candidate", bankroll, **kw)
    _, stress["all_selected_contracts_lose"] = policy_replay(
        d.assign(y_yes=np.where(d.side.eq("yes"), 0, 1)), trades, "p_candidate", bankroll)
    _, stress["double_queue_ahead"] = policy_replay(d.assign(queue_ahead=2*d.queue_ahead), trades, "p_candidate", bankroll)
    return dict(status="research_only", production_authorized=False, data_kind=str(d.data_kind.iloc[0]),
                input_rows=len(raw), independent_opportunities=len(d), games=int(d.game.nunique()), dates=int(d.date.nunique()),
                probability=losses, baseline=bm, challenger=cm,
                mean_daily_net_delta_usd=paired_interval(d, cand.pnl-base.pnl),
                opportunity_weighted_brier_gain=paired_ratio_interval(
                    d, d.opportunity_weight*((d.p_baseline-d.y_yes)**2-(d.p_candidate-d.y_yes)**2), d.opportunity_weight),
                net_delta_per_baseline_requested_contract_usd=paired_ratio_interval(d, cand.pnl-base.pnl, base.requested),
                folds=folds, stress=stress,
                limitations=["Replay assumes non-impacting FIFO orders and complete aggressor prints; no queue cancellations.",
                             "Single first-state player/stat family; opportunity weights frozen from baseline features.",
                             "Comparison is fixed offline policy, not a replay of the entire old trader.",
                             "Max drawdown uses settled P&L; open-position mark-to-market is unavailable.",
                             "Synthetic results cannot establish calibration or profitability on NBA/Kalshi data.",
                             "No automated model promotion; historical input provenance requires independent review."])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opportunities", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # Preserve lexical engine identities (e.g. 2026.30 vs 2026.3).
    frame = pd.read_csv(args.opportunities, dtype={"baseline_engine": str, "challenger_engine": str, "ticker": str})
    report = evaluate(frame, pd.read_csv(args.trades))
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Research report written to {args.out}; production authority is disabled.")


if __name__ == "__main__":
    main()
