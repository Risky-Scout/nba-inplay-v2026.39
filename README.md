# NBA In-Play Pricing & Market-Making System

**Author:** Joseph Shackelford, ASA, MAAA  
**Trader:** v2026.40 · **Probability engine:** 2026.31 · **Execution policy:** qp8

An end-to-end research system for real-time NBA player-prop pricing, execution, portfolio construction, and model governance.

## Current status

| Area | Status |
|---|---|
| Trader | v2026.40 |
| Probability engine | 2026.31 |
| Quote policy | qp8 |
| Code path | current |
| Full production authority | not assumed; must be earned from current evidence |
| Recommended review mode | PAPER / research only |

## What the system does

- prices live NBA PTS / REB / AST / 3PM / PRA markets;
- uses a generative Markov / Monte Carlo engine with live NBA state and Dunks & Threes priors;
- models path-dependent minutes, rotations, foul-out risk, pace, usage, shot mix, assists, and rebound opportunities;
- anchors probabilities to market information unless the model earns incremental authority;
- optimizes maker quote price, quantity, TTL, fill probability, and execution residual;
- sizes correlated positions with Kelly, hard caps, shared model-risk factors, and CVaR;
- revalidates play-by-play and book state immediately before sending;
- records prediction, decision, quote, fill, markout, P&L, and governance evidence.

## Probability engine 2026.31

Engine 2026.31 is intentionally a new probability identity and does not inherit 2026.30 calibration or authority.

Key changes relative to 2026.30:

- lexical engine/version identities are preserved at CSV ingestion;
- valid zero shot rates remain zero rather than receiving synthetic attempt mass;
- zero-foot shots remain rim attempts rather than falling through to a default distance;
- live rebound learning uses event-time lineup exposure reconstructed from substitutions when available;
- synthetic ORB/DRB minimum capture weights are removed;
- D&T generative calibration is iterative, damped, and verified on a fresh simulation seed rather than certified by one target/simulation ratio pass.

## Architecture

```text
Live NBA state + D&T priors
            ↓
Player / lineup state
            ↓
Generative Markov simulation
            ↓
Joint player-stat distributions
            ↓
Kalshi + optional sportsbook state
            ↓
Market-anchored calibration / authority
            ↓
Execution-adjusted expected value
            ↓
Correlated portfolio optimization + CVaR
            ↓
Fresh PBP + fresh-book revalidation
            ↓
Maker quote / reject
            ↓
Fill, markout, P&L and governance evidence
```

## Live review

For a hiring-manager demonstration:

```bash
./scripts/validate.sh
./scripts/run_live_review.sh
```

The live-review script is PAPER-only by construction. It cannot submit real-money orders.

After settlement:

```bash
./scripts/run_hiring_manager_audit.sh
```

## Important files

| File | Purpose |
|---|---|
| `nba_markov_engine.py` | Live-state probability engine and Monte Carlo simulation |
| `nba_stint_engine.py` | Minutes / rotation mechanics |
| `rotation_hazard.py` | Substitution / return and pathwise blowout helpers |
| `endgame_state.py` | Foul / bonus / intentional-foul state helpers |
| `kalshi_nba_inplay_trader.py` | Pricing, execution, order lifecycle and safeguards |
| `kalshi_portfolio_qp.py` | Correlated sizing and CVaR |
| `kalshi_exec_model.py` | Execution probability / filled-quantity interface |
| `kalshi_ws.py` | WebSocket order-book and queue state |
| `fit_calibration.py` | Walk-forward calibration and local authority |
| `fit_execution.py` | Execution challenger/champion fitting |
| `grade_predictions.py` | Predictive / execution / production-gate grading |
| `kalshi_results.py` | Settlement lookup and identity-safe evidence loading |
| `nightly_reports.py` | P&L / attribution / nightly reporting |
| `risk_report.py` | Drawdown and risk reporting |
| `capacity_report.py` | Observed-support capacity diagnostics |
| `execution_size_authority.py` | Quantity-bucket authority |
| `hiring_manager_audit.py` | Reviewer-facing audit package |

## Validation

Run:

```bash
./scripts/validate.sh
```

That compiles and runs the current regression stack. The expected final banner is:

```text
PORTFOLIO VALIDATION: PASS
Trader v2026.40 | Engine 2026.31 | Quote policy qp8
```

## Production governance

Code readiness is not economic readiness.

A real-money production start must earn the required predictive and execution evidence for the current engine / policy family. New probability-engine versions start with fresh predictive authority. Safety gates are not weakened to force activity.

## Known limitations

- Engine 2026.31 still needs fresh out-of-sample evidence before production authorization.
- qp8 still needs sufficient live execution evidence before a champion execution model can be promoted.
- close-game final-two-minute states remain blocked from new entries until the full foul / free-throw endgame process is integrated into the main event generator.
- default quoting remains one-sided unless the NO side earns authority.
- PAPER fill simulation is research evidence, not a substitute for real queue/fill evidence.

This repository is a quantitative-research and engineering portfolio project. It does not claim guaranteed profitability.
