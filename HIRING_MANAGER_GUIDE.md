# Hiring Manager Guide

## 5-minute review

This project is best understood as an **in-play pricing and market-making system**, not as a standalone prediction model.

### 1. Probability engine

Start with:

- `nba_markov_engine.py`
- `nba_stint_engine.py`

Key idea: the system models the **remaining distribution of player outcomes conditional on live game state**, not just a single point forecast.

Look for:

- possession-level event generation
- player role / usage state
- path-dependent rotations
- substitution and return hazards
- simulated-score-dependent blowout behavior
- generative D&T calibration
- joint player-stat simulation

### 2. Market and execution layer

Then review:

- `kalshi_nba_inplay_trader.py`
- `kalshi_exec_model.py`
- `kalshi_ws.py`
- `execution_hazard.py`

Key idea: a good probability is not automatically a good trade.

The trader explicitly separates:

- fair-value estimation
- uncertainty
- fill probability
- queue position
- adverse selection
- quote price
- quantity
- TTL
- fresh-book / fresh-state validation

### 3. Portfolio and capital risk

Review:

- `kalshi_portfolio_qp.py`
- `risk_report.py`
- `execution_size_authority.py`

Key idea: positions are sized jointly.

The optimizer accounts for correlated exposures and full-notional risk. Fill probability can improve expected-value ranking but cannot make hard exposure disappear.

### 4. Validation and governance

Review:

- `fit_calibration.py`
- `fit_execution.py`
- `grade_predictions.py`
- `predictive_governance.py`
- `test_v2039.py`
- `tests/test_v2037.py`
- `tests/test_v2038.py`

Key idea: **new code does not inherit economic authority**.

Predictive cells, execution policies, and quantity buckets must earn authority from the correct version / policy family and from later or live evidence.

## Design decisions worth discussing

### Why market anchoring?

Sports markets contain information the model may not have. The framework therefore treats the market as an information source and requires the model to earn incremental weight.

### Why generative calibration?

Adjusting event-generation rates preserves the internal structure of the simulation better than post-hoc count manipulation.

### Why path-dependent rotations?

Minutes and substitutions depend on how the game evolves. A fixed deterministic schedule understates uncertainty in player opportunity.

### Why a separate execution model?

A theoretical edge that rarely fills—or fills only when the market is moving against you—can be economically inferior to a smaller edge with better execution.

### Why fail closed?

In live markets, stale information is often more dangerous than no information. The system therefore rejects or cancels decisions when state quality degrades.

### Why policy-family governance?

Changing an economic rule changes the distribution of orders and fills. Evidence from another policy should not automatically certify the new one.

## Reproducible review

### Regression suite

```bash
python3 test_v2039.py
python3 tests/test_v2037.py
python3 tests/test_v2038.py
```

### Paper-mode live-feed research demo

```bash
python3 kalshi_nba_inplay_trader.py \
  --ws \
  --sims 8000 \
  --shadow-sims 2000 \
  --model-weight 0.35
```

This is intentionally PAPER / RESEARCH mode.

## Interview discussion prompts

Useful areas to challenge:

- where model error enters the probability stack
- how simulation variance is separated from structural uncertainty
- why exact action support includes quantity and TTL
- how queue resets are treated conservatively
- why hard risk is full-notional rather than fill-discounted
- how walk-forward authority avoids leakage
- what evidence would justify promoting a challenger execution model
- when market information should dominate the internal model
