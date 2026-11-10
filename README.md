# NBA In-Play Pricing & Market-Making System

**Author:** Joseph Shackelford, ASA, MAAA  
**Trader:** v2026.39 · **Probability engine:** 2026.29 · **Execution policy:** qp7

An end-to-end research system for **real-time NBA player-prop pricing, execution, portfolio construction, and model governance**.

The project is designed to answer a harder question than “what is the fair probability?”:

> Given uncertain live-game state, market prices, execution risk, correlated exposures, and model uncertainty, **should an order be quoted at all—and if so, at what price and size?**

## Live review in GitHub Codespaces — no reviewer accounts required

The portfolio branch includes a **LIVE REVIEW** path for current or upcoming NBA games.

It uses **no synthetic game, player, or market inputs**:

- current NBA scoreboard, box score, and play-by-play;
- current public Kalshi player-prop markets and REST order books;
- a current Dunks & Threes snapshot refreshed privately in GitHub Actions.

The reviewer never receives the repository owner's D&T credential. The Codespace consumes only the current read-only snapshot.

### Quick start

1. Open this branch in GitHub Codespaces.
2. Validate the code:

```bash
./scripts/validate.sh
```

3. Run the current-game live review:

```bash
./scripts/run_live_review.sh
```

The live-review command is **PAPER-only by construction**. It does not pass `--live`, `--pilot-live`, or authenticated WebSocket flags, and the trader refuses any attempt to combine `--review-dt-snapshot` with real-money mode.

The same probability engine, market parser, execution economics, portfolio optimizer, pre-send revalidation, and order-state machinery are used. Only the D&T authentication path is replaced by the read-only current snapshot.

> **Owner setup required once:** configure the repository's `DT_API_KEY` Actions secret and run the **Live Review Data Refresh** workflow. Confirm that the Dunks & Threes subscription terms permit sharing the captured/derived snapshot with the reviewer before enabling external access.

See [LIVE_REVIEW_DATA_POLICY.md](LIVE_REVIEW_DATA_POLICY.md) for provenance and safety details.

## What this project demonstrates

- live player-prop probability estimation
- generative Markov / Monte Carlo simulation
- path-dependent rotations and blowout behavior
- market-anchored probability calibration
- uncertainty-aware edge estimation
- maker execution and queue-state modeling
- fill-probability and partial-fill modeling
- correlated Kelly portfolio optimization
- CVaR and hard-notional risk controls
- WebSocket market monitoring with REST fallback
- stale-state / stale-book fail-closed behavior
- walk-forward validation and production governance

## System architecture

```text
Live NBA state + D&T priors
            ↓
Player / lineup state
            ↓
Generative Markov simulation
            ↓
Joint player-stat distributions
            ↓
Kalshi + sportsbook market state
            ↓
Market-anchored probability layer
            ↓
Calibration + uncertainty penalties
            ↓
Execution-adjusted expected value
            ↓
Correlated portfolio optimization
            ↓
Fresh PBP + fresh-book revalidation
            ↓
Maker quote / reject
            ↓
Fill, markout, P&L, and governance evidence
```

## Core design choices

### 1. Generative probability model

The model updates live player-stat distributions from possession-level game state. Calibration acts on underlying event intensities and role processes rather than cosmetically forcing final simulated totals toward a target.

Examples:

- PTS calibration alters rim / mid / FTA attempt intensities
- 3PM primarily alters 3PA intensity
- rebounds alter ORB / DRB capture weights
- assists alter assist routing
- shooting percentages are not changed merely to force a mean

### 2. Path-dependent rotations

Each Monte Carlo path can evolve differently. Simulated score margin, bench duration, stint length, and game phase affect substitution / return hazards and deep-bench probability.

### 3. Market anchoring

The system does **not** assume the model dominates the market.

Model influence must be earned through out-of-sample evidence. If that evidence is absent, the system can assign zero directional authority rather than manufacture a trade.

### 4. Execution-aware economics

The execution layer models:

- requested quantity
- quantity relative to displayed depth
- quantity relative to queue ahead
- queue depletion
- price-level resets
- time-to-live
- partial fills
- expected filled quantity
- post-fill markouts

The action identity for qp7 is **price × requested quantity × TTL bucket**.

### 5. Portfolio construction

Candidate orders are sized jointly rather than independently.

The optimizer includes:

- fractional Kelly
- player / game / position / session caps
- cross-market covariance
- shared model-risk shocks
- CVaR constraints
- existing exposure
- full-notional hard risk treatment

Fill probability may affect expected economic value, but it never reduces hard worst-case exposure.

### 6. Fail-closed execution

Immediately before submission, the trader re-checks both the live game state and the order book.

An order is rejected or repriced when, for example:

- play-by-play state changed
- market moved materially
- the book became stale
- the WebSocket sequence became unreliable
- the quote would now cross
- the opportunity no longer survives risk / edge checks

### 7. Model governance

**Code readiness is not economic readiness.**

Engine 2026.29 and quote policy qp7 do not inherit production authority from older versions. They must earn it from current evidence.

Production authorization requires evidence such as:

- walk-forward predictive validation
- confirmed authority on later unseen games
- live fill evidence for the current policy family
- live post-fill markouts
- an earned execution prior
- canonical model-status approval

A new version may therefore be technically correct and still be intentionally blocked from full production.

## Important files

| File | Purpose |
|---|---|
| `nba_markov_engine.py` | Live-state probability engine and Monte Carlo simulation |
| `nba_stint_engine.py` | Minutes, rotation, substitution, and return behavior |
| `kalshi_nba_inplay_trader.py` | Market parsing, pricing, execution, order lifecycle, and safeguards |
| `kalshi_portfolio_qp.py` | Correlated portfolio optimization and risk constraints |
| `kalshi_exec_model.py` | Execution probability / expected-filled-quantity interface |
| `fit_calibration.py` | Walk-forward probability calibration and authority |
| `fit_execution.py` | Execution / fill / markout model fitting |
| `grade_predictions.py` | Predictive, execution, and production-gate grading |
| `kalshi_ws.py` | WebSocket order-book state and queue tracking |
| `predictive_governance.py` | Version-specific predictive authority |
| `execution_size_authority.py` | Quantity-bucket production authority |
| `risk_report.py` | Drawdown / risk reporting |
| `pnl_attribution.py` | P&L attribution |
| `capacity_report.py` | Capacity diagnostics |
| `nightly_reports.py` | Nightly research / governance outputs |

## Validation

The current regression stack covers order integrity, probability-engine behavior, execution logic, portfolio risk, WebSocket recovery, policy-family isolation, and governance.

Run:

```bash
python3 test_v2039.py
python3 tests/test_v2037.py
python3 tests/test_v2038.py
```

Expected results:

```text
ALL v2026.39 CHECKS PASSED
V2026.37 CHECKS PASSED
ALL v2026.39 CHECKS PASSED
```

## Safe demo

The default trader mode is **PAPER**.

A representative research demonstration is:

```bash
python3 kalshi_nba_inplay_trader.py \
  --ws \
  --sims 8000 \
  --shadow-sims 2000 \
  --model-weight 0.35
```

The manual model weight is explicitly labeled a **RESEARCH OVERRIDE** and is not production authority.

For a hiring-manager review, use PAPER mode only.

## What to look at first

If you have 5 minutes:

1. Read this README.
2. Open `nba_markov_engine.py` for the probability architecture.
3. Open `kalshi_nba_inplay_trader.py` for the execution / fail-closed logic.
4. Open `kalshi_portfolio_qp.py` for correlated sizing and CVaR.
5. Open `fit_calibration.py` and `predictive_governance.py` for evidence-based authority.
6. Review `test_v2039.py` for the highest-priority regression checks.

For a guided technical review, see **[HIRING_MANAGER_GUIDE.md](HIRING_MANAGER_GUIDE.md)**.

## Current status

| Area | Status |
|---|---|
| Code / regression validation | PASS |
| Engine version | 2026.29 |
| Trader version | v2026.39 |
| Execution policy | qp7 |
| Full production authority | Not assumed; must be earned from current evidence |
| Recommended review mode | PAPER / research only |

## Scope

This repository is a quantitative-research and engineering portfolio project. It demonstrates probability modeling, market making, execution, portfolio risk, testing, and governance. It does **not** claim guaranteed profitability or positive returns on any particular day.
