# Research audit and validation

Baseline: latest `main` verified on November 14, 2026, commit
`0f20c6c3c29d8f8c215a11024efb7c734b99d74d`.
Branch: `research/calibration-execution-capital-preservation`.

**The owner explicitly authorized restoring live-trading capability. PAPER remains
the default; live modes require explicit flags and credentials. No orders were
submitted and nothing was deployed.**
This change does not establish improved NBA calibration or after-cost profitability.
The economic acceptance criterion remains unfulfilled pending historical evidence.

## Audit findings and changes

| Area | Baseline finding | Change / remaining constraint |
|---|---|---|
| Architecture | Shared generative Markov paths, D&T skills, lineup schedules, simulated pace/usage/assists/rebounds, market pooling, maker ladder, QP and CVaR already exist | Preserved this architecture and existing paper/order lifecycle paths |
| Minutes / injuries | Stored `p_active` did not change pregame simulated availability; scheduler could fill slots using zero allocations | Stratified Bernoulli availability per scenario, shared by every stat and teammates' redistributed minutes; known live eligibility supersedes pregame estimates; no zero-allocation lineup slots |
| Game state | Clutch compiled using the initial score difference | Compile normal/clutch late-game variants and choose using simulated margin at segment boundaries; OT starts clutch |
| Correlation | Common minutes, pace, possession events, rebound/assist routing and model-risk factors already present | Preserved joint paths and PRA identity; availability adds a shared roster shock rather than independent marginal thinning |
| Calibration | Iterative generative D&T mean matching with independent-seed verification already exists | Preserved; **mean matching is not binary probability calibration evidence**; changed engine identity to 2026.32 |
| Prior fitting | Outer-test losses selected the returned prior and `bool(params)` certified validation | Last inner-validation winner is frozen before the outer test; outer scores diagnostic; `validated=False` |
| Execution uncertainty | Mean quote EV used fill fraction correctly, but bootstrap kept fill fraction fixed | Paired game-bootstrap fraction/residual draws affect QEV uncertainty and maker-ladder lower bounds; uncertainty enters model identity; malformed or old-schema champions cannot load |
| Allocation | `kelly_fraction=0` selected legacy half Kelly; zero max-contract limit treated as absent | Explicit zero budgets allocate nothing; invalid Kelly fractions rejected |
| Tail risk | Mean of all losses at/above the quantile included excess mass at tied losses | Exact empirical worst-tail mass, fractional boundary atom; missing scenario vectors treated as losses; empty vectors reject new orders |
| Cross-game scenarios | Engine calls can reuse seed/row indices across independent games | Seeded per-game permutations preserve within-game player/strike dependence; shared model-error factors remain shared |
| Economic attribution | CVaR resizing dropped fill/confidence discounts from displayed EV | Resized EV retains these discounts and reports conditional-on-fill EV separately |
| Evidence | Existing audit has per-market/phase views and game bootstrap; candidate settlement diagnostics assume fills; existing execution fitter uses logging dates | Added strict offline as-of paired evaluation; legacy diagnostic/gate methodology has not been replaced or newly certified |
| Execution controls | Maker-only guards, stale-book/PBP gates, terminal-proof reconciliation, reserved pending risk and shutdown cancellation already exist | Retained; live capability restored on explicit owner authorization; direct mutations still require a non-paper authenticated client |

No code was deleted as obsolete: the audit did not establish a safe removal.

## Reproducible baseline comparison

Install `requirements.txt` in a virtual environment, then run:

```bash
python scripts/compare_research_baseline.py --out work/baseline-comparison.json
./scripts/validate.sh
```

The comparison loads the pinned baseline source from Git into temporary modules.
Inputs are **engineered synthetic failure cases**, not November 12 records and
not a trading backtest. Fixed simulation seed is 42, with 400 paths.

| Diagnostic | Baseline | Challenger | Interpretation |
|---|---:|---:|---|
| Requested contracts with explicit zero Kelly | 125 | 0 | Honor capital disablement |
| CVaR95: 99 zero losses and one $100 loss | $1 | $20 | Exact worst 5% tail mean is $20 |
| Quote EV SD when only fill fraction is uncertain | 0c | 5.0156c | Missing uncertainty is now charged |
| Mean future PTS for a pregame player with `p_active=0` | 1.615 | 0 | Availability respected across the joint path |
| Independent families from 3,000 repeated forecasts | — | 30 | Polling frequency is not evidence size |

The lower requested size or larger estimated risk is not a measured profit gain.
These diagnostics have exact invariants or fabricated inputs; a confidence interval
would not establish real-world generalization.

## Regression results

The unmodified baseline passed all three existing suites: v2039, v2037
(111 checks), and v2038 (49 checks). The improved branch retains these suites
with current-engine identities and mocked exchange methods. Live capability tests
verify explicit non-paper mode and both credentials are required; direct mutations
remain blocked in PAPER mode. Existing mocked trader-cycle tests cover the
production gate and pilot execution without exchange network requests.

The added unittest suite covers zero Kelly, exact CVaR, missing scenarios, paired
fill uncertainty, as-of violations, duplicate/overlapping evidence, future-label
invariance, queue/latency/TTL replay, duplicate trade rejection, single-date
interval refusal, independent cross-game CVaR coupling, prior holdout isolation, availability, feasible lineups and
simulated-score clutch selection. CI runs this suite and source compilation.
Current local results: 16 new unittest cases, 51 v2039 checks, 111 v2037 checks,
and 49 v2038 checks; no failures. Source compilation and the synthetic evaluator
CLI smoke test also pass (Python 3.12.14). Commands and CI status are recorded in
the pull request.

## Historical data availability

No historical NBA states, predictions, order books, trades, fills or settlements
are committed to the public repository. The owner identified November 12, 2026
logs in their Mac directory:

```text
/Users/josephshackelford/NBA-In-Play-Simulation/nba-inplay-live-current/
```

Those local files are not accessible from this cloud workspace. The reported 135
quote records are not 135 independent outcomes. One date cannot support a
multi-date chronological out-of-sample economic claim or date-bootstrap interval.
No private logs, credentials or D&T snapshots have been added to the branch.

Before economic validation, provide multiple dates of immutable prediction/graded
outcome, eligible/decision, quote/event/fill, queue and markout logs, plus as-of NBA
states/D&T snapshots and complete exchange trade/L2 events. Existing quote and
markout logs alone cannot reconstruct missing queue flow or simulate an unseen
maker-price action. Historical state snapshots are required to compare the changed
generator against the old generator without using hindsight injuries/rotations.

The new evaluator deliberately does not invent a converter from the uninspected
Mac files. Column meanings, timestamp zones, available-at dates, fees, exact engine
identities and settlement joins need verification first.

## Strict offline economic evaluator

```bash
python economic_validation.py \
  --opportunities /private/path/asof_opportunities.csv \
  --trades /private/path/aggressor_trades.csv \
  --out /private/path/economic_report.json
```

### Opportunity schema

One row per logged quote action/snapshot; bids/asks/probabilities are as of the
prediction, on the indicated contract side except `p_*` and `y_yes`, which refer
to YES. All timestamps must include a UTC offset.

| Fields | Meaning |
|---|---|
| `timestamp`, `state_available_at` | Prediction time, latest availability time of its input state |
| `baseline_trained_through`, `challenger_trained_through` | As-of model training/selection boundary, strictly earlier than prediction |
| `label_available_at`, `y_yes` | Time settlement became knowable, binary YES outcome |
| `game`, `player`, `stat`, `ticker`, `side` | Stable identities; side is `yes` or `no` |
| `baseline_engine`, `challenger_engine` | Exact lexical engine identities; do not mix versions within either arm |
| `p_baseline`, `p_challenger` | Frozen YES probabilities; generator comparison requires separate historical simulation of both |
| `bid`, `ask`, `price_cents`, `book_age_s` | Side book, logged maker price, state age |
| `requested_qty`, `queue_ahead`, `ttl_s` | Whole requested contracts, queue ahead at arrival (including latency additions), quote expiry horizon from decision time |
| `fee_cents` | Known per-contract fee estimate; evaluator rounds upward per contract conservatively |
| `expected_fill_fraction` | Ex-ante fraction forecast, never realized filled quantity |
| `data_kind` | Exactly `historical` or `synthetic`; mixed provenance rejected |

Each trade requires `trade_id`, `timestamp`, `ticker`, `side`, `price_cents`,
`quantity`, `aggressor` (`buy`/`sell` relative to that side). Trade IDs must be unique.
Input provenance and event completeness require independent review. A market's
trade stream must use one consistent side representation; do not double-count
YES/NO complements.

### Estimand and chronology

- Before fitting or evaluation, keep the first timestamp of each game/player/stat
  family. Choose one side/strike at that timestamp using only baseline ex-ante
  expected profit divided by capital at risk, expected fill fraction and requested
  quantity. Cap this opportunity weight at one. Repeated forecasts, both sides and
  overlapping strikes cannot add independent family observations.
- Paired predictive loss uses the same frozen opportunity weights for both arms;
  weights do not depend on realized returns or on challenger outcomes. Also report
  ordinary Brier so profit weighting cannot conceal general calibration failure.
- Fit an identity-regularized logit transform to challenger predictions only on
  earlier game dates with labels available before the test boundary. Weight training
  games equally. Require 20 training games; otherwise use the identity transform.
  A game's earliest New York prediction date controls its fold; midnight cannot split
  the game across train and test. No hyperparameter search is conducted on test rows.
- Replay only the shared logged price/first-state support. Exact-price sell-aggressor
  volume must consume queue ahead after 500 ms arrival and before TTL expiry. A bid
  touch, buy-aggressor trade or pre-arrival print cannot fill an order. Ignore queue
  cancellations; assume non-impacting FIFO orders and complete print coverage.
- Both arms use the **same fixed offline comparison policy**: 0.1 Kelly with a fixed
  three-percentage-point probability haircut, 1% player, 3% game and 10% total capital
  caps, and a 3% settled-equity drawdown stop. Reserve full requested cost until TTL,
  then keep filled exposure until settlement. This is not the entire old trader's
  QP policy, so it cannot by itself establish system-level uplift.
- Report after-cost net P&L, invested capital, ROI, actual/requested quantities,
  settled-equity drawdown and daily CVaR. Paired 95% intervals resample whole game-date
  blocks, preserving within-date correlation; single-date intervals are unavailable.
  Net-delta-per-requested-contract uses baseline requested quantity as its fixed
  denominator. Opportunity-weighted Brier gain uses a ratio bootstrap.
- Stress 2-second latency, doubled fees, half fills, doubled queue, an 8-point model
  haircut and all selected contracts losing. Half fills is a sensitivity, not a model
  of adverse selection. No scenario, report or passing test grants live authority.

## Outstanding limitations

Real calibration, after-cost returns and uncertainty intervals are **unmeasured**.
Availability sampling is conditional on pregame information, not a fitted injury
recovery model. Scenarios are finite/stratified and player availabilities are sampled
independently; correlated team news needs historical fitting. Rotation scheduling
still approximates minutes with discrete segments, and future foul caps do not
replace a possession-by-possession foul process. Clutch switches at boundaries,
not every possession. Close final-two-minute states remain blocked by the existing
entry gate until the full foul/free-throw process is integrated.

The existing execution fitter's legacy date-based rolling reports still need an
available-at audit for terminal/fill/markout labels, especially overnight orders;
the strict evaluator is not a certification of those legacy reports. Bootstrap
execution uncertainty holds the deployed scaler/drift fixed, so it does not capture
all nuisance-model or structural error. Existing shared model-risk factors and
conservative exposure controls remain necessary. No maker-price or Kelly tuning
has earned promotion from real historical evidence.
