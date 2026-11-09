# NBA In-Play Player-Prop Pricing & Execution System

**Author:** Joey Shackelford, ASA, MAAA  ·  v2026.38 (engine 2026.28, quote policy qp6)

This system prices NBA player props while the game is live (PTS, REB, AST, 3PM, PRA). It uses a possession-level absorbing Markov chain built on Dunks & Threes (D&T) player-skill estimates, and trades mispriced Kalshi contracts. Bets are sized with correlated fractional Kelly (0.10 during the pilot), and orders rest on the book only (maker-only) to keep fees down. Current settings are always archived per run in `run_config_*.json`. Sections 5c–5t describe how the system evolved; where an older section disagrees with them, the later section and the run config are authoritative.

## Run it in two minutes, with no API keys and no money

```bash
pip install -r requirements.txt     # Python 3.9+
python3 tests/test_engine.py         # engine validation against a known D&T projection
python3 tests/test_sizing.py         # correlated Kelly checks
python3 tests/demo_offline.py        # full trader against mocked NBA / D&T / Kalshi APIs
python3 tests/test_calibration_and_pairs.py   # calibration fit, contradictory-pair block, live gate, fill journal
python3 tests/test_market_sources.py          # sportsbook consensus, microprice, markouts, provenance
python3 tests/test_state_and_risk.py          # reservations, journal, exchange check, held covariance, kill switch, USG%
python3 tests/test_v2037.py                   # v2026.37: authority, order states, fill-aware economics, WS, full cycles
python3 tests/test_v2038.py                   # v2026.38: canonical authorization, family evidence, sign-flip FDR, confirmation, action support
python3 tests/test_v2029.py                   # v2026.29-30: maker EV, pre-send re-check, OT/engine fixes, regime weights, execution gate, full mocked cycle
```

The demo runs the real order path, including signed-payload construction, against a mock exchange:
- maker/taker routing;
- fill reconciliation;
- event-driven cancels when a new play touches a player;
- reduce-only exits;
- the ledger and the CLV report.

Expected final lines are `ENGINE CHECKS PASSED`, `SIZING CHECKS PASSED`, `END-TO-END DEMO PASSED` and `CALIBRATION + PAIR + GATE CHECKS PASSED`, and `V2026.29-33 CHECKS PASSED`.

| File | Role |
|---|---|
| `nba_markov_engine.py` | Live feeds (NBA CDN box score and play-by-play), the D&T v2 client, the Markov compiler, the Monte Carlo engine, and calibration to D&T |
| `nba_stint_engine.py` | Minutes model: projected share, in-game rotation evidence, foul-trouble windows, foul-outs, blowouts |
| `kalshi_portfolio_qp.py` | Kalshi fee model and the correlated fractional-Kelly quadratic program |
| `kalshi_nba_inplay_trader.py` | Market parsing, pricing gates, order lifecycle, fill reconciliation, exits, ledger, CLV report |
| `fit_calibration.py` | Fits per-stat probability calibration to settled markets and writes `calibration.json` |
| `grade_predictions.py` | Grades every logged prediction against settlement and writes the live gate, `model_status.json` |
| `kalshi_results.py` | Settlement lookups (cached in `settlements.csv`) and prediction-log loading |
| `evaluate_night.py` | Post-game scoring: Brier and log loss for model vs market, calibration buckets, per-stat bias, P&L, CLV |

## 1. Probability model

1. **Possession chain.** For each pair of 5-man units, the engine compiles two matrices from D&T per-100 skills:
   - Q: a 5×5 transient matrix (passes between players);
   - R: a 5×9 absorbing matrix (rim, mid and 3PT makes and misses, 2- and 3-shot fouls, turnovers).

   Both are adjusted for matchup defense (DEPM), rim protection, steal pressure, live usage and fatigue.
2. **Pace.** Expected touches per possession come from the fundamental matrix N = (I−Q)⁻¹, including offensive-rebound continuations. Touch durations (Gamma(2)) are then scaled so simulated possessions match D&T pace, blended with live pace as the game goes on.
3. **Minutes.** D&T expected minutes go through the stint engine and are water-filled so each team plays exactly 5 × time left. The engine builds a deterministic lineup schedule in 2-minute segments, starting from the lineup on the floor. Foul-outs and inactive players are removed.
4. **Rebounds and assists.**
   - Rebounds are split by D&T ORB/DRB rates.
   - The assisted-FG rate is Σ(on-court ast/100) ÷ the chain's made field goals per 100. The assister is drawn in proportion to ast/100.
5. **Calibration to D&T.**
   - Before each game, a full-game simulation is compared with D&T's projected box score, adjusted for each player's chance of playing.
   - Live increments are rescaled by per-player multipliers, using binomial thinning or thickening.
   - A player/stat whose raw ratio falls outside 0.67–1.5 is flagged and never traded.
   - Dispersion: the v2026.17 shrink (k = 0.88) was removed in v2026.21 (k = 1.0). Outcomes showed the spreads were too narrow; minutes, rotation and pace scenarios now widen them (section 5e).
6. **Output.** A joint distribution of every player's stats over 4,000 simulated completions of the game. Both the probabilities and the cross-market correlations come from it.

## 2. Validation (`tests/`, synthetic game with a known D&T projection)

| Check | Result |
|---|---|
| Simulated pace (target 99.0) | 98.9 possessions |
| Minutes vs D&T projection | mean error 0.5 min per player |
| Calibrated means vs D&T box (fresh seed, out of sample) | mean error 2.0%, max 6.8% across PTS / REB / AST / 3PM |
| Runtime | 2,000 full-game simulations with 24 minutes scenarios: see test_engine output |
| Single-bet sizing | optimizer matches closed-form fractional Kelly (the gap is the fee) |
| Correlated sizing | same-player ladder with ρ = 0.44 gets 14% less total stake than independent bets with the same edges |
| Existing exposure | new risk is capped by open positions ($140 held in a $150 game cap leaves $9.88) |

## 3. Pricing gates (in order)

1. The game is live, the box score has loaded, and more than 45 s remain.
2. The Kalshi event date token matches tonight's NBA game date.
3. The player-name match is unique, and the structured strike agrees with the market title.
4. The calibration trusts this player/stat.
5. The book is two-sided with a spread of 6¢ or less, and the mid has moved no more than 3¢ since the last cycle. A sudden jump usually means news the NBA feed hasn't shown yet.
6. |model − market| ≤ 0.15. A larger gap is treated as a data or model error, not an edge.
7. **Winner's-curse shrink:**
   - `p_used = w·model + (1−w)·market`, where w = 0.5 × (1 − ½·|gap| / 0.15).
   - The bigger the disagreement, the less weight the model gets.
8. Price is between 20¢ and 80¢. ROI on all-in cost (price + fee) must be at least 5% for a maker order, plus 3 points while the ball is live. A taker order needs 10%, except in the last 5 minutes of Q4/OT.
9. **No chasing:**
   - A 180 s cooldown before adding to a position.
   - Never average down if the market has moved 3¢ or more against the last fill.
   - No orders under $2.

## 4. Sizing: correlated fractional Kelly

The optimizer solves max μᵀx − (γ/2)xᵀΣx with γ = 2, where x = existing + new positions.
- μ is the expected return per dollar of all-in cost.
- Σ uses correlations measured from the simulated joint outcomes, and is block-diagonal across games.

| Cap (all include existing positions) | Default |
|---|---|
| Total open risk for the night (worst-case nightly loss) | 10% of bankroll (pilot) |
| Per game | 3% (pilot) |
| Per player (all of one player's markets) | 1.5% (pilot) |
| Per position | 1% (pilot), tapered on favorites: min(1, 2(1−price)) |
| Per order | $5 (pilot) |
| Kelly fraction | 0.10 (pilot) |

## 5. Execution, fees and in-game risk controls

- **Maker-first.**
  - Post-only orders rest one cent inside the bid, so they never pay the taker fee.
  - Each series' fee schedule is read from `GET /series` at startup. Prop series charge either no maker fee or 0.0175·P(1−P).
  - Fees are always included in EV.
- **Taker only when it pays for itself.** An IOC order at the ask is used only if ROI after the taker fee (0.07·P(1−P)) clears `--taker-min-ev`.
- **Event-driven cancels.** Between pricing cycles the bot polls the play-by-play every 2 s. A resting order is cancelled as soon as its player appears in a new play, so stale quotes aren't left for faster traders to pick off.
- **Exits.** When the bid, after the fee, is 3¢ or more above the model's blended fair value, the bot sells with an IOC `reduce_only` order. That order can only shrink a position, never reverse it.
- **Order lifecycle.**
  - Every resting order has an expiry and is cancelled each cycle.
  - Fills are reconciled from `GET /portfolio/fills`.
  - The ledger records actual fills and the fees actually charged.
  - Positions are rebuilt from the ledger on restart and checked against Kalshi's positions for open markets (section 5i).

## 5c. Calibration, contradictory pairs and the live gate (v2026.19)

The first two live nights showed that the engine overrated unlikely outcomes. It bought cheap contracts on extremes, and held contradictory pairs such as NO 10+ and YES 25+ points on the same player. v2026.19 adds three controls.

- **Per-stat calibration fitted to outcomes** (`fit_calibration.py`).
  - Form: p = sigmoid(a + b·logit(p_raw)), fitted per stat. b > 1 means the raw tails were too fat for that stat.
  - Each market gets equal weight, and a ridge penalty pulls the fit toward no change.
  - A stat needs at least 40 settled markets, and its fit is written only if it improves Brier score out of sample (5 folds split by player).
  - In testing, a fit on about 250 markets (one night) was dominated by noise, and the out-of-sample check correctly refused to write it. Expect several nights of data before a fit is written.
  - The fit corrects whatever the current engine outputs, and is only valid for the engine version it was fitted on (section 5f).
- **Contradictory-pair block.** The bot never holds an UNDER (NO) at one line together with an OVER (YES) at an equal or higher line on the same player and stat. Both legs lose whenever the result lands between the lines. Where candidates conflict, the higher-ROI one is kept. Middles (OVER at a lower line plus UNDER at a higher one) are still allowed.
- **Live gate.**
  - `grade_predictions.py` writes `model_status.json`, and `--live` refuses to start unless it shows PASS.
  - Only predictions logged by v2026.19 or later count. These are out of sample by construction, since any calibration active at the time was fitted on earlier nights.
  - PASS requires at least 300 settled markets, a 90% interval for the edge entirely above 0 (resampled by player), and a mean edge of at least 2¢ per contract at the mid, which roughly covers the cost of getting filled.
  - `--override-gate` exists, but trading live without a pass is not recommended.

**Nightly workflow:**
1. Paper trade: `python3 kalshi_nba_inplay_trader.py`.
2. Next day, grade: `python3 grade_predictions.py`.
3. Refit: `python3 fit_calibration.py`.
4. Go live only once the gate reports PASS.

## 5d. What the first graded night showed, and the v2026.20 response

Graded on Nov 5: 447 settled markets, 64,738 predictions, 2 games.

| Finding | Evidence | v2026.20 change |
|---|---|---|
| The market was sharper overall | Brier score: model 0.1234 vs market 0.1170 | The price is now market-anchored: p = sigmoid((1−w)·logit(market) + w·logit(model)), w = 0.20, refit daily by `fit_calibration.py` (Nov 5 fit: 0.31) |
| The model is overconfident | Per-stat calibration slope b = 0.71–0.97 (all below 1). When the model said 60–95%, outcomes came in 10–15 points lower | Market anchoring shrinks the model's extremes |
| Large disagreements carry no edge | Edge at the mid: +2.2¢ (gap ≤ 3¢), +4.3¢ (3–6¢), +3.3¢ (6–10¢), +0.5¢ (10–15¢), 0.0¢ (> 15¢) | Trade only gaps of 3–10¢. On the fills side, 69 of 76 settled fills had gaps over 10¢ |
| Crossing the spread lost the money | Taker fills −$74.62; maker fills −$1.06 | Maker only (`--taker` to re-enable) |
| The model added nothing on 3PM, and PRA is biased high | Fitted model weight: 3PM 0.07. PRA mean 0.505 vs actual 0.405 | Trade only PTS, REB and AST |
| First half negative, second half positive | Q1–Q2 −1.0¢ (one game); Q3 +1.6¢ / +6.6¢; Q4 +5.2¢ | Trade from Q3 (`--min-period`). The model's edge is fast state updating, and the market prices pregame information well |
| Minimum edge | | 2¢ absolute after fees, plus 3% ROI |

**Caveat:** this is one night and two games, and the games disagreed. Fitted on one game and tested on the other, the market-anchored price beat the market on ORL@PHX and roughly tied it on MIA@GSW. These rules cut the measured losses, but they do not establish a profitable edge yet. The live gate (section 5c) is the measure of that.

## 5e. Engine fix (v2026.21): the simulation itself was overconfident

**Diagnosis from the Nov 5 outcomes** (14,379 sets of predictions at several lines for one player and stat):
- Each set implies the engine's mean and spread for that player's final total. These were compared with the market's implied distribution and with the actual results.
- The engine's spreads were about 20% narrower than the market's on every stat (PTS 0.78×, REB 0.78×, AST 0.85×, 3PM 0.88×, PRA 0.71×). Outcomes say the spreads needed to be wider (PTS ×1.17, AST ×1.17, PRA ×1.41).
- The engine's mean projections, measured against actual finals, were as accurate as the market's on PTS, REB, AST and 3PM. PRA ran high.

**Root causes in the chain:**
1. Minutes were identical in every simulated game. Allocation and lineups were computed once and reused for all 4,000 simulations, so foul trouble, blowouts and coaching variance never happened. A player's PTS, REB and AST rise and fall together with his minutes, which is why PRA (their sum) was the narrowest of all.
2. A 0.88 dispersion shrink (added in v2026.17 from bugged-era fills) narrowed every spread by a further 12%.

**Fix:**
- 24 minutes scenarios per run. Each draws every player's remaining minutes (lognormal, coefficient of variation 0.18), refills each team to exactly 5 × time left within the foul and blowout caps, and builds its own lineup schedule.
- The shrink is removed (k = 1.0).
- Measured spread change on the synthetic game: PTS ×1.18–1.20, AST ×1.11, REB ×1.16, 3PM ×1.08–1.10, PRA ×1.22. Mean projections are essentially unchanged.

**Effect on Nov 5 outcomes** (the measured spread change applied to the logged distributions, not fitted to outcomes), Brier score on PTS/REB/AST:

| | Market | Old engine | Fixed engine |
|---|---|---|---|
| ORL@PHX | 0.1205 | 0.1142 | 0.1141 |
| MIA@GSW | 0.1241 | 0.1313 | 0.1298 |
| Both games | 0.1225 | 0.1237 | 0.1228 |

**Still open:**
- PRA is still too narrow (×1.22 against a target of ×1.41). It likely also needs night-to-night role and usage variation shared across a player's stats, so PRA is not traded.
- MIA@GSW still lost to the market. Five players (Giannis, Portis, Podziemski, Melton, Wiggins) account for most of it. The prediction log now records projected mean, spread, projected minutes and the current stat, so the next grading can separate minutes errors from rate errors.

## 5f. Independent audit fixes (v2026.22)

| Audit finding | Fix |
|---|---|
| Calibrated probability not used in the trade path | Entries and exits now use the calibrated probability (identity for a stat with no fit). The pool weight is fitted on calibrated probabilities |
| No fitted calibration for PTS/REB/AST; a fit on one engine could be applied to another | Every prediction is tagged with `ENGINE_VERSION`. `fit_calibration.py` fits only on rows from the current engine, and the trader ignores a calibration fitted on a different engine (it would double-correct) |
| Delayed fills could be forgotten | Cancelled orders stay in a pending journal for 4 cycles and keep being matched against `/portfolio/fills`. Tested: a fill reported after the cancel is still booked |
| Stale quotes after other players' plays | Any new play in a game cancels every resting order in that game |
| No overtime | A tie at the horn simulates up to 3 five-minute OT periods on the closing lineups. Tested: tied with 20 s left, a star's P(5+ more points) rises from 0.2% to 20.5%; an 8-point game is unchanged |
| Monte Carlo noise mistaken for information | Common random numbers: the seed is a hash of the game state, and an unchanged state reuses its simulation. Paths per state raised to 8,000 |
| No uncertainty margin on edge | The edge bar adds 1.28 Monte Carlo SE (scaled by the model weight) plus a 0.5¢ adverse-selection buffer |
| New orders during live ball, without residual-possession modelling | New orders only during dead balls (`--allow-live-ball` to override) |
| Independent minutes per player | Hierarchical minutes: player shock (CV 0.15) plus a team rotation shock (CV 0.10) that moves starters and bench in opposite directions (close game vs blowout). Spreads still match the Nov 5 targets (PTS ×1.17–1.19, AST ×1.10–1.13) |
| Fringe players dropped from the simulation | The simulation roster keeps anyone active with projected minutes, plus anyone on the court or with live minutes. Trade eligibility is still limited by `--min-minutes` |
| Player-blocked calibration folds leaked within-game information | Folds are blocked by game. The pool weight is written only if pooling beat the market on held-out games; otherwise it is 0 |
| Live gate not clustered | The gate now requires at least 15 games, and resamples whole games once there are 10 or more |
| Half-Kelly too aggressive for an unproven edge | Quarter-Kelly by default (`--kelly`) |

**Not done yet** (needs data or infrastructure, documented honestly):
- Removing possibly double-counted matchup adjustments on game-optimized D&T inputs. This needs a historical test so it doesn't break calibration.
- An end-game regime for intentional fouls and the clock.
- Rebuilding the FT process (and-ones, technicals, FT rebounds).
- Residual-possession state.
- Using the collected play-by-play features (assist pairs, live shot mix).
- Kalshi WebSocket, order-book depth and microprice, two-sided quoting with inventory skew.
- Reconciliation against the exchange positions endpoint.
- D&T freshness and game_id provenance checks.
- Walk-forward recalibration across nights, once enough nights exist.

## 5g. New information sources and execution data (v2026.23)

Implemented from the second audit, ordered by expected value per change:

1. **Sportsbook consensus** (`sportsbook_consensus.py`, The Odds API; set `ODDS_API_KEY` in the Keychain or the environment).
   - Each book's Over/Under pair is de-vigged multiplicatively, and books are combined by the median of logits.
   - When a book hangs a different line than Kalshi, the book probability is moved to the Kalshi line using the simulation's shape at the book's location.
   - **Now:** a confirmation gate (never buy a side the books price below Kalshi) plus logging (`p_book`, `n_books`).
   - **Later:** `fit_calibration.py` fits a sportsbook weight for a three-way logit pool, written only if it beats market + model on held-out games.
   - Cost: 3 credits per game per refresh (`--books-interval`, default 90 s).
2. **Order-book microprice.** For markets that pass the cheap gates in a dead ball, the size-weighted top of book, (ask·Q_bid + bid·Q_ask)/(Q_bid + Q_ask), replaces the midpoint as the Kalshi price. It falls back to the mid on any doubt and is logged as `p_micro`.
3. **Adverse-selection markouts** (`kalshi_markouts_<date>.csv`): every entry fill is re-marked 15 s, 30 s and 120 s later. This is the training data for the fill-probability and adverse-selection models.
4. **Shared pace scenarios.** Each minutes scenario also draws a game-pace shock (CV 0.04) shared by every player. Spreads still match the Nov 5 targets.
5. **Per-stat model weights**, shrunk toward the global weight and written only when the global weight was earned out of sample.
6. **Whole-distribution diagnostic** in `grade_predictions.py`: the model's vs the market's implied spread and mean per stat from each set of lines, plus sportsbook accuracy on the same markets.

Already in v2026.22: overtime, and the two-clock design (the simulation reruns only when the game state changes; execution checks run between plays).

**Deferred, with reasons:**
- Kalshi WebSocket order book: new infrastructure and a persistent connection.
- A learned competing-hazards possession model, a minutes/rotation hazard model, an end-game regime, and the removal of matchup layers by ablation: all need a historical play-by-play dataset.
- The fill-probability and adverse-selection models: they need the markout data this version starts collecting.
- Regime weights by period, spread or liquidity: they need many more graded nights.

## 5h. Pilot safeguards (v2026.24): the engine is frozen, execution is narrowed

The Markov engine is unchanged (`ENGINE_VERSION` 2026.23). Changes are limited to evidence integrity and risk.

| Change | Why |
|---|---|
| The live gate is engine-specific: `model_status.json` records the engine, the grader counts only predictions from the current engine, and the trader rejects a PASS written for another engine | Older engines must not authorize a new one; v2026.23 starts with zero evidence |
| The sportsbook gate uses exact Kalshi lines only (`--book-max-line-diff 0`). Translated lines are logged, not traded | Translating a book line uses the model's distribution shape, so it is not independent confirmation |
| No new positions in the last 2:00 of Q4/OT within 8 points | Intentional fouls, free throws and end-game strategy are not modelled yet; avoiding the regime beats a rushed model |
| Microprice: at least 5 contracts on each side of the top of book, and it may move the Kalshi anchor by at most ±1¢ | An unvalidated microstructure feature must not dominate the pricing; the markout file will test it |
| D&T provenance fails closed: both teams in /epm and /game-predictions-box, and where returned, matching date, game id, opponent and an update within 36 h. A mismatch skips the game | One bad data join can create enormous apparent edges |
| Pilot defaults: 0.10-Kelly, session 10%, game 3%, player 1.5%, position 1%, $5 per order | Exposure sized for an engine with no graded outcomes. After an edge is established, raise the Kelly multiplier first |
| Calibration activates only with 3+ games from the current engine (diagnostics are always printed) | One night must not steer the engine |
| The grader reports model-minus-market Brier per stat with game-bootstrap 90% intervals | One night yields an interval, often "inconclusive", not a verdict |

Note on sizing: the per-position cap times the bankroll must exceed the $2 minimum order. A 1% cap needs `--bankroll` of $200 or more.

## 5i. State, risk and evidence integrity (v2026.25)

| Change | Why |
|---|---|
| Reserved risk: every resting order, and every cancelled-but-unconfirmed order, counts at full possible cost against cash and every cap until the exchange proves otherwise. Late partial fills shrink the reservation | Cancellation does not mean zero exposure |
| Durable order journal (`kalshi_orders_journal.json`): unresolved orders survive a crash and are reconciled at restart | In-memory state disappears on a crash |
| Fill IDs already in today's ledger are loaded at startup | A restart must never book the same fill twice |
| Exchange reconciliation (`GET /portfolio/positions`) on cycle 1 and every 20 cycles, for open NBA markets. A mismatch trips the kill switch (no new entries; exits still run) | The exchange is the source of truth. Settled markets are excluded, since they show 0 on the exchange while the ledger keeps the fill |
| Held positions enter the Kelly covariance as fixed exposures (the cross term γ·f′Σ_NH·h) | A new bet correlated with what is held (same player, minutes, OT) carries more risk than its own variance shows. Tested: stake 0.050 → 0.033 with a correlated holding |
| Kill switches: a daily loss of 3% of bankroll (realized + mark-to-market), an exchange position mismatch, or 3+ cancel failures. Each cancels resting orders and blocks new entries | The best model still loses money if its state is wrong |
| Strategy-subset gate: every trade-eligible candidate is logged (`kalshi_eligible_*.csv`) and settled at the price that would have been paid, after fees. The live gate now also needs ≥150 such candidates with net profit's 90% interval (game bootstrap) above 0 | What matters is E[profit given the trade policy], not average accuracy |
| Sportsbook gate needs ≥2 books, each updated within 90 s | One stale book must not veto or confirm a trade |
| Live usage uses standard USG% (actions × team minutes/5 ÷ (minutes × team actions)), shrunk toward D&T with weight min(0.25, team possessions/80). Engine version 2026.24 | Replaces the rough actions-per-minute × 0.48 conversion; targets the player-role misses |
| Shadow engine: the same state simulated with the extra matchup layer off (DEPM, rim protection, steal adjustments), never traded, logged as `p_shadow` | An A/B test settles whether the layer double-counts D&T's game-optimized inputs |
| Quote log (`kalshi_quotes_*.csv`): every posted order with the book around it | Training data for P(fill given quote and state) and quote placement |
| Feed-latency breaker: if 3+ Kalshi prices in a game jump with no new NBA event, quotes are pulled and the game skipped that cycle | The market may know something the feed has not shown yet |
| Calibration activates only with ≥5 games and ≥100 markets per stat | A few games carry little independent information |
| Run config archive (`run_config_*.json`): version, engine, calibration ID, weights, limits and a code hash for every run | Stale documentation causes operational mistakes |

**Still deferred:** fitted adverse-selection buffer and quote-level optimization (need the markout and quote logs this version collects); end-game and FT regime (needs historical play-by-play); WebSocket order book.

## 5j. Model weight default (v2026.26)

The default logit-pool weight is now **0.35** (model 35%, Kalshi 65%), up from 0.20, at the operator's choice. It sits at the top of the Nov 5 fitted range (0.31–0.34). Held out game by game, a weight near 0.34 beat the market on ORL@PHX and was slightly worse on MIA@GSW. A higher weight means larger fair-value moves for the same model/market gap, so more candidates clear the edge bar, with more exposure to model error. The simulation-error buffer scales with the weight, and every other gate is unchanged. Once `fit_calibration.py` writes a weight for the current engine (5+ games), that fitted weight replaces the default unless `--model-weight` is passed.

## 5k. Persistent maker quotes (v2026.27–28)

- **Before:** every resting order was cancelled and re-posted every 15-second cycle, even when unchanged. On Nov 6 the bot posted about 300 orders in 70 minutes and got roughly 1 fill. Each re-post lost queue priority, and the cancelled orders (reserved until their fills were confirmed) filled the risk cap with phantom exposure.
- **Now:** a resting order the strategy still wants at the same ticker, side and price stays in the queue. It is cancelled only when:
  - the desired price changes or the order is no longer wanted;
  - any new play happens in its game (unchanged);
  - it is older than 75 s (it is then refreshed; the exchange expiry is 90 s);
  - the kill switch trips.

  Only cancelled-but-unconfirmed orders count as reserved risk for sizing. The cycle line reports `new orders, kept resting`.
- **Feed-lag breaker (v2026.27):** counts only liquid markets (spread of 6¢ or less), is skipped during period breaks and stoppages, and needs at least 5 markets and 20% of them to move.

## 5l. Why it loses at the fill, and the v2026.29 response

Where 5m differs (spread credit default, edge bar, shadow haircut, live gate, engine version), 5m is authoritative.

**Diagnosis** (from the code and the Nov 5–6 record; no Nov 6 logs were available, so these are mechanisms, not measured losses):

| # | Defect | Mechanism | Fix |
|---|---|---|---|
| 1 | Edge was mostly the spread | A resting bid at bid+1 counted (market fair − our price) as edge. On a 6¢ book a 3¢ model disagreement passed the bar only because 2¢ of the "edge" was spread capture, which is exactly what a faster trader takes back by hitting a stale quote. The bot therefore preferred wide, thin books, where fills are most adversely selected | `--as-spread-frac 0.5`: half the spread capture is assumed given back on fills (prior). The grader estimates the real share from event-free markouts |
| 2 | Orders priced on a stale snapshot | All games are simulated, then all orders sent, so a book read at cycle start can be many seconds old (the "post only cross" rejects) | Pre-send re-check: no orders for a game with a new play since its simulation; every order re-read against the live book (no crossing, no outbid, mid moved ≤ 1¢) |
| 3 | Quotes outlived the dead ball | Quotes are posted in dead balls but rested up to 75 s, so they sat into live play before the lagging NBA feed showed the first new play | `--quote-ttl 30` exchange expiry, refreshed at 27 s |
| 4 | Model uncertainty ignored | 8,000 paths shrink Monte Carlo error, not specification error | Haircut = w × (main − shadow disagreement beyond one MC s.e.); `--shadow-z` |
| 5 | Sizing on nominal edge | The optimizer sized on p_used | It now sizes on the conservative probability (price + fee + kept edge) |
| 6 | Cross-game risk = 0 | A biased model is wrong in every game at once | Low-rank model-risk factors in Σ (global, stat, game; 1¢ each). Small at pilot sizes: the haircut on the mean is the main defence |
| 7 | Paper mode locked itself out | Paper cancels stayed in `pending` forever and were reserved at full cost against every cap | Paper pending cleared at reconcile |
| 8 | Paper never saw the order book | Book requests were always signed; with no key they failed, so paper never used the microprice | The public book is read unsigned in paper |
| 9 | $2 minimum killed small edges | At 0.10-Kelly on $250, a 3¢ edge sizes to about $1.50 and was always dropped, even on fee-free series | The minimum applies only where a fee is charged |
| 10 | Markouts mixed basketball with pick-offs | Marks every 15 s, with no record of plays in between | 5/15/30/120 s marks with a play-in-between flag; 5–30 s marks from the live book between cycles |
| 11 | Gate tested eligibility, not fills | E[profit given eligible] can be positive while E[profit given filled] is negative | New fill gate: ≥ 100 settled entry fills from this engine, net profit 90% CI above 0 (game bootstrap) |
| 12 | Weights fitted on the wrong population | Pool weight fitted on every prediction, not where the bot trades | Fitted on the trade regime (Q3+, 20–80¢, gap 3–10¢, spread ≤ 6¢, dead ball); falls back to all rows with a label when there are fewer than 100 markets |
| 13 | Per-stat weights untested | Written whenever the global weight was earned | Written only if they beat the global weight on held-out games for that stat |
| 14 | Three-way pool off the simplex | Kalshi weight could be fitted negative while the trader clamped it to 0 | Fitted with w_model + w_book ≤ 1 |
| 15 | `calibration.json` | The uploaded file has no `_meta.engine`, so the trader ignores it: PTS/REB/AST trade raw engine probabilities | No change needed; the next `fit_calibration.py` run on 5+ games of engine 2026.24 writes a valid one |

**Volume.** The stricter edge rule removes candidates whose edge was mostly spread capture. On the mocked test state
(every market 5¢ from the model, 4¢ spread), candidates fell from 55 to 22, almost all from the give-back prior. Whether
that is too strict is measured, not assumed: every candidate the old rule would take and the new one drops is logged
(`passes_conservative = False` in `kalshi_eligible_*.csv`) and graded separately (section 9b of `grade_predictions.py`).
Section 11 prints the measured give-back from event-free markouts; pass it as `--as-spread-frac` when it has 20+ marks.
Lowering the $2 minimum on fee-free series raises the number of orders that are actually sent.

**What is still not done:** WebSocket order book and fills (polling remains; the re-check narrows the stale window but does
not close it), a fitted fill-probability and adverse-selection model (needs the markouts and quote logs this version
collects), queue-aware quote placement, joint ladder (CDF) fitting, rotation hazards and the end-game regime.

**Nightly workflow (unchanged):** paper trade → `grade_predictions.py` → `fit_calibration.py`. Live only after the
model, strategy and fill gates all pass.

## 5m. Execution-aware economics and engine fixes (v2026.30, engine 2026.25)

**Maker EV.** A resting order is worth *directional alpha + execution alpha − fee*:
directional alpha = 100·(p_used − p_mkt), what the model adds over the market's own fair price;
execution alpha = the part of the bid-to-fair spread that survives a fill. Until event-free markouts measure it,
execution alpha is 0 (`--as-spread-frac 1.0`), so a wide spread can no longer manufacture maker EV.
`grade_predictions.py` (section 11) prints the measured value to use instead.

**Robust edge into Kelly, instead of more gates.** The kept edge also subtracts Monte Carlo error, a 0.5¢ buffer, and
calibration uncertainty: 0.75 × 100·p(1−p)·|logit p_model − logit p_mkt|·w_sd, where w_sd is the game-bootstrap s.d. of
the pool weight (`fit_calibration.py`; 0.15 until fitted). The optimizer now takes this as `expected_edge_c`
(μ = edge / unit cost) and keeps `win_prob` for the outcome covariance; reported EV uses the kept edge.

**The edge bar moved from 2¢ (nominal) to 0.5¢ (robust), and min ROI from 3% to 1%.** The 2¢ bar was a proxy for
the cost of getting filled, and that cost is now subtracted explicitly. Keeping both would charge it twice, and with no
spread credit the old bar admits nothing: at a 0.35 weight and the 10¢ disagreement cap, robust edge peaks near 1.9¢. On
the mocked state (markets 3/5/7/9¢ from the model): v2026.28 rule 52 candidates, no spread credit + 2¢ bar 0, v2026.30
default 46. Small robust edges are sized small by Kelly rather than banned. `--min-edge-c 2 --min-ev 3` restores the old bar.

**Latency.** The research shadow engine runs after the cycle's orders are sent (never on the execution path), and order-book
depth is fetched only for final orders in the pre-send check (`--micro` re-enables per-market microprice in pricing).

**Engine 2026.25 (probability bugs):**
| Bug | Effect | Fix |
|---|---|---|
| OT minute cap used regulation ceilings | A star past ~1.15× projected minutes got 0 remaining minutes in OT (test: 44 min played, 4:00 left in OT1 → 0) | Cap is period-aware; in OT never below minutes played + time left |
| OT elapsed time assumed 48 minutes | Rotation evidence and live pace used the wrong elapsed time in OT | `NBAStintEngine.elapsed_minutes(period, sec_rem)` |
| Missing possession defaulted to HOME | Biased first possession | UNKNOWN → 50/50 draw |
| OT inherited the final regulation possession | | OT opens with a jump-ball draw |
| Pace scenario multiplier exp(−σz + σ²/2) | Mean exp(σ²) > 1 (slightly slow simulated pace) | exp(−σz − σ²/2), mean one |
| Hard bench usage cap | Missed genuine role promotions | Removed; live usage is already shrunk to D&T (weight ≤ 0.25) |
| Extra matchup layer on game-optimized D&T inputs | Possible double counting, unvalidated, live | `MATCHUP_STRENGTH` ∈ [0,1], production default 0 (`--matchup-strength`); the shadow runs at 1 and section 12 of the grader reports the A/B |

`ENGINE_VERSION` is 2026.25, so calibrations and gates start fresh for this engine.

**Calibration deployment:**
- The sportsbook weight is fitted with the deployed model weight held fixed (w_book ≤ 1 − w_model).
- Regime weights (any stat × period, then stat × period) replace the hard Q3 rule once fitted. Until calibration.json has an H1 weight, trading starts in Q3; after that, from Q1.
- A stat or regime weight that overrides a coarser one needs 5+ held-out games and a game-bootstrap 97.5% lower bound of held-out gain above 0. Tested: no false cells in 6 null runs, but low power (a strong real Q4 effect was found in 1 of 6 runs of 10 games). Regime weights will take many graded games to earn.
- The pre-tip loader slices positionally.

**Live authorization is execution evidence:** the model edge gate AND ≥ 100 settled entry fills with net profit
90% CI above 0 AND ≥ 30 event-free 30-s maker markouts with mean ≥ 0. Eligible-candidate P&L (section 9) is research only.

**Logging for the fill model:**
- `kalshi_quote_events_*.csv` records every cancel (reason: new_play, feed_lag, state_moved_presend, expired_refresh, no_longer_wanted, kill_switch) and every fill, with quote age.
- Quote rows carry the play number at submit, top-of-book sizes and the expected kept edge.
- Markouts carry order_id, fill_id and the engine.

**Still not done (next):**
- substitution-hazard rotations with lineup persistence;
- a future foul-out hazard;
- Bayesian live shot-mix and assist routing;
- emergency fallback profiles for unprojected players;
- an end-game/intentional-foul regime;
- a fitted P(fill) and conditional markout model, with quote choice among bid, bid+1 and ask−1;
- CVaR portfolio optimization;
- a WebSocket order book.

## 5n. What the Nov 6 files show, and the v2026.31 response

Data: 59 live maker fills (484 contracts), 38 with 15/30/120 s markouts, 2,022 quotes, 6,753 eligible candidates,
195,918 predictions over 9 games. Settlement was approximated by each market's final logged price (resolves 366 markets
and 25 fills); `grade_predictions.py` uses Kalshi's settlements.

| Question | Nov 6 answer |
|---|---|
| Was the −13.61¢ "CLV" adverse selection? | No: it compares each fill with the last observed mid, so it is mostly game outcome. Realized −16.1¢/contract on 25 resolved fills, noise s.e. ≈ 13¢: one night cannot separate luck from edge |
| Were fills picked off? | Yes, by side. 30 s markout: NO (under) fills −3.85¢ (30 fills, all 7 pick-offs of −9 to −22¢); YES fills +0.10¢ (8). Overall −2.79¢, larger than the +1.13¢ of spread the bot thought it earned |
| Selection on fills? | Resolved under candidates: +8.6¢ when not filled (26) vs −10.2¢ when filled (5). Small n, but the pattern a passive maker fears |
| Is the model sharper than the market? | No. Brier 0.0510 model vs 0.0432 market (366 markets); edge at mid +0.82¢ (90% CI −0.46 to +2.48); 6–10¢ gaps −1.9¢ |
| Right model weight? | Fitted on outcomes, held out by game: 0.10 overall, 0.00 in the trade regime (pooling lost to the market in 9/9 held-out games). Nov 5: 0.31 in sample. The bot used 0.35 |
| Matchup layer? | No difference: shadow (layer off) Brier 0.0511 vs main 0.0510 |
| Duplicate fills? | None: every repeated order_id line has a distinct fill_id (partial fills) |
| Fill rate / period | 2.6% of quotes filled; two thirds of filled contracts came in Q1–Q2 (the run used --min-period 1) |

**Changes:**
- **Measured execution alpha by side** (`--exec-alpha-no-c −2.3`, `--exec-alpha-yes-c 0`): Nov 6 markouts shrunk by n/(n+20), added to the maker's kept edge. `grade_predictions.py` prints the updated priors each night.
- **Execution-health breaker:** if the last 8 fills' contract-weighted markout is ≤ −3¢, sizes are halved and the bot joins the bid instead of improving it. At ≤ −6¢, entries pause for 10 minutes. Marks with a play in between are included, because the Nov 6 pick-offs were exactly fills just before a play the feed showed seconds later. The live markout gate uses all 30 s marks for the same reason.
- **Drawdown throttle:** sizing is 100% / 65% / 35% at 0 / −1% / −2% of bankroll, with no new exposure to players already held past −2%. The −3% stop is unchanged.
- **Default model weight 0.35 → 0.20.**

**Consequence, stated plainly:** replayed on Nov 6's eligible candidates, the evidence defaults keep 6 of 606 market-sides
(all YES), against 317 at weight 0.35, and no under ever clears the measured pick-off cost. On current evidence the
model's disagreements are too small to pay for execution. Raising volume needs a sharper model or better execution (a live
order-book feed, a fitted fill and toxicity model), not looser settings. For paper nights, the data-collection settings
`--model-weight 0.35 --exec-alpha-no-c 0` keep producing the fills and markouts that measure this; live trading stays
behind the model, fill and markout gates.

## 5o. Execution integrity, liquidation risk and minutes resolution (v2026.32, engine 2026.26)

**Execution (P0):**
- **Fresh-book repricing, then one re-solve.** Every proposed order is re-read against the live book after the optimizer. It must still be mechanically valid (no cross, not outbid, mid within 1¢) and still clear the edge bar when its kept edge is recomputed at the fresh mid. Survivors are re-sized by one more optimizer pass. (v2026.31 tolerated a 1¢ adverse move against a 0.5¢ edge bar.)
- **Play-by-play checked before every order, fail closed.** No per-game caching, because a play can land between two sends. `latest_action()` returns None when the feed cannot be read, and that means unknown state, so there is no new exposure: the order is skipped and the game's resting quotes are pulled. The between-cycle watcher does the same. (`new_actions()` returned "nothing happened" on a failed read.)
- **NO bids join the bid** while the NO execution alpha is below −1¢; they never stand alone in front of the queue.

**Evidence (P0):**
- **Fill gate:** LIVE fills only (paper fills are simulated and cannot certify passive execution), counted per order rather than per fill message, needing ≥ 100 orders, ≥ 10 games and ≥ 250 contracts, with a game-bootstrap 90% CI above 0.
- **Markout gate:** ≥ 50 LIVE maker orders over ≥ 10 games, with a contract-weighted 30 s markout ≥ 0.

**Risk and accounting (P1):**
- **Liquidation-value P&L** (bid minus taker fee) drives the drawdown throttle and the −3% stop. Mid-marked P&L is logged alongside.
- **Markout start = play number at submit,** because REST fills are reported late.
- **No double count:** `--exec-alpha-*-c` is the raw 30 s markout, which already contains any retained spread, so combining it with `--as-spread-frac < 1` is refused at startup.

**Engine 2026.26 (P1):**
- **Lineup segments of 45/30/15 s** (more than 12:00 left / to 6:00 / last 6:00) instead of 120 s, with lineup persistence: an on-court player stays until his allocation is used, or until a bench player has 90 s more remaining. Mock, 4:00 left in Q4:
  - simulated minutes vs allocation error 0.33 → 0.04 min;
  - star simulated 2.42 → 2.83 of his 2.83 allocated minutes;
  - P(+5 points) 4.9% → 6.1%.
  - Run time per Q3 state 1.1 → 1.7 s.
- **Future foul-outs:** each minutes scenario draws future fouls ~ Poisson(rate × allocated minutes). The rate is a position prior (0.050 / 0.075 per minute) shrunk toward tonight's PF per minute with 12 minutes of prior strength. A sixth foul caps the player at alloc × Beta(k, N − k + 1), and teammates are re-filled.

**Calibration (P2):** the global pool weight and the sportsbook weight now need the same held-out bootstrap test as stat and period weights. The model weight is capped at 0.20 until there are 10+ games of the current engine. Cell-weight fallbacks are refitted inside each held-out fold (no leakage).

**Not done yet** (each changes the event model and needs its own validation):
- calibration applied inside event rates instead of post-simulation thickening (the PMF issue for M > 1);
- FT / bonus / and-one structure;
- Bayesian live shot-mix and assist routing;
- emergency profiles for unprojected players;
- a CVaR constraint on the QP;
- separate quantity / entry / liquidation fields for held positions.

## 5p. Inventory, evidence and research/production separation (v2026.33, engine 2026.27)

**Production optimizes money; research optimizes information.**

**Production defaults:**
- **Model weight 0** unless calibration.json earns one, and **weight 0 means no directional entry**.
- **NO side off** until exec_priors.json shows it passing on live orders.
- **Q3+ only;** Q1–Q2 never switch on automatically.
- **Edge rule:** the 90% LCB must be ≥ 0.5¢ plus inventory skew.
- **Tail limit:** CVaR95 ≤ 2% of bankroll.
- **Execution priors:** live-only and quote-policy matched.

**Research mode:** `--model-weight`, `--allow-no` or `--exec-alpha-*-c` force PAPER mode and print a banner. `--live --model-weight` is refused unless `--override-model-weight-risk` is given.

| Area | Change |
|---|---|
| Resting quotes | `sync_quotes` is quantity-aware per (ticker, side, price): equal keeps; larger target keeps the resting part and posts the difference; smaller target cancels and re-posts (risk beats queue); remaining quantity is used for partial fills. Resting quotes reserve cash and enter the risk state for every cap (keys the optimizer is re-deciding are re-sized to a total target, so throttles shrink resting size) |
| State freshness | Before `sync_quotes` keeps anything, each game in the desired set is play-by-play checked; unreadable or moved → its resting quotes are cancelled and its orders dropped. The per-order check before each POST stays |
| Positions | Average-cost accounting: exits remove average entry cost; realized P&L is tracked separately; ledger reload replays the same rule. Caps and the optimizer see the remaining cost basis, not net cash |
| Logs | Quote log: fresh pre-send bid/ask, depth, play number at submit, book age, top-up size, run id, quote policy. Markouts start at Kalshi's fill timestamp (`ts_source`, `reporting_delay_ms`; local fallback flagged). Every log row carries `run_id` → `run_config_<run_id>.json`; appends keep each file's existing header |
| Edge | LCB90 = Edge_mean − buffer − 1.282·√(σ_MC² + σ_cal² + σ_exec²) − H_shadow, each charged once. H_shadow (main-vs-shadow disagreement beyond MC noise) applies automatically whenever the shadow is cached for the exact state; never negative or NaN. Inventory skew requires +1¢·(player exposure / cap)² |
| Execution priors | grade_predictions.py writes exec_priors.json from LIVE maker orders only, merged per order, game-clustered, tagged with engine and quote policy (`qp3`). Staged status per side: <10 diagnostic, 10–29 provisional, 30–49 interval only, 50+ over 10+ games pass/fail on the 90% lower bound of the ALL-fill 30 s markout (an event-free view is reported alongside). A failing side is disabled. S.e. floor 0.25¢. CLI default s.e. corrected to the s.e. of the mean (Nov 6: YES 0.71¢, NO 1.92¢) |
| Tail risk | After sizing: scenario P&L of held + pending + resting (not replaced) + proposed orders. Binary search for the largest scale with CVaR95 ≤ 2% of bankroll, re-evaluated after rounding to whole contracts; the 99th percentile is logged |
| Engine 2026.27 | Foul-out caps are hard in `_water_fill` (the relaxation pass can relax soft rotation caps only). Foul-rate clamp 0.015–0.12/min, with every bind recorded in `FOUL_CLAMP_LOG` |
| Calibration | Per-stat (a, b) needs leave-one-game-out gains with a bootstrap 97.5% lower bound > 0 (5+ games). Without trade-regime evidence the production pool weight is 0 (all-market fit printed for research only) |

**Research command for paper data collection:**
`--model-weight 0.35 --allow-no --exec-alpha-no-c 0` (forced paper; NOT validated for live capital).

## 5q. First-half information architecture, phase 1 (v2026.34, engine 2026.28)

From the Markov and high-speed-trading specialists' first-half review. Principle: the model does not need more
confidence in Q1-Q2; it needs genuinely new live information, with strong priors.

| Change | Before | Now |
|---|---|---|
| Starter status | `p_mp >= 26` proxy | tonight's actual opening five (NBA box `starter`), else D&T `p_pct_start`, else the proxy |
| Opening stints | starters had no stint until first sub-in, so no fatigue term | on since tip; halftime/OT resets the stint |
| Rotation evidence | weight `elapsed/48` (0.125 at 6:00) | `exp_so_far / (exp_so_far + 10)`, 1.5x evidence once a full stint has been seen |
| Shot mix | collected, unused; rim = <=6 ft or any paint shot | Dirichlet posterior (prior = 30 attempts) on the zone split; total and FG% stay D&T; rim = <=4 ft |
| Assist routing | D&T ast/100 only | Dirichlet posterior (prior = 12 assists) with tonight's assister->scorer pairs |
| Pace | `0.35 x elapsed/48` | `N_poss / (N_poss + 180)`: same order of magnitude by sampling math (Q1 end 12%, half 22%) |
| Unknown on-court player | dropped (four teammates absorbed his minutes) | replacement-level emergency profile |
| Role multipliers | 1.12 / 1.16 hand-coded | 1.00 until earned; fatigue now named parameters |
| Minutes CV | constants | per-phase tables (values unchanged until fitted on logged final minutes) |
| Regimes | H1 / Q3 / Q4 | Q1 / Q2 / HT / Q3 / Q4 in trader, calibration and grader |
| H1 weight | could fall back to the global weight | a first-half cell must beat the MARKET on held-out games; otherwise its weight is 0 |
| Grading | period means | 5c: per-phase Brier/log loss (model, blend, market) and edge by gap 0-3/3-6/6-10c with 90% game bootstrap; first-half error by starter status |

Not done (need data we do not have yet): substitution/return hazards by coach and player, event-rate (in-chain) D&T
calibration, score-dependent simulated rotations, fitted phase CVs. First-half trading still requires --min-period 1
and, for earned weights, a first-half cell that beat the market out of sample.

## 5r. From candidates to fills (v2026.35, quote policy qp4)

Nov 7 showed two separate bottlenecks: candidates that never became orders, and orders that never filled.
- **Whole-contract sizing.** The optimizer's continuous answer was floored to whole contracts, so at 0.10-Kelly on a
  small bankroll nearly every stake rounded to 0. Now it floors, then adds one contract at a time where that still
  raises the same EV-minus-risk objective (covariance, caps and held positions included). It never adds a contract
  the risk model calls a loss: on $100, a 0.6c edge on a 50c contract still does not trade; 1.5c now does.
- **Quote ladder.** The bid goes as high as the mid (always below the ask) while the 90% edge bar still holds. The
  edge is measured against the mid, so bidding above bid+1 costs nothing the rule counts; it only gives up spread
  capture, which Nov 6 showed was not real. `--quote-ladder bid1` restores bid+1.
- **Funnel.** Each cycle logs candidates -> positive optimizer stakes -> under one contract -> lifted -> sized ->
  after pre-send checks, and warns when the bankroll is too small for the edge regime.
- **Gap floor 1c** (was 3c): the LCB already charges for noise, so the hard floor double-filtered.
- **Tails:** 10-20c and 80-90c are allowed with +1.0c on the edge bar (20-80c unchanged).
- **NO probes:** `--no-probe N` sends up to N one-contract, join-the-bid NO orders per run, so the NO side can re-earn
  live status from real markouts (grade_predictions section 9). Off by default.
- Not yet (needs this data): fitted fill probability and markout by quote level; taker routing on an LCB.

## 5s. Production governance and execution-conditioned alpha (v2026.36, quote policy qp5)

The objective every layer now serves: **realized alpha = P(fill) x [predictive alpha + execution alpha - costs]**,
maximized under CVaR, concentration, inventory and drawdown limits.

**Run modes.** `production` requires ALL of: a passed live gate for this engine, a calibration fitted on this engine,
an earned (not manual) model weight, and >= 30 live YES orders under qp5. `--override-gate` and
`--override-model-weight-risk` no longer authorize production. `--pilot-live` is the only way to trade real money
without that evidence, and it forces 1 contract per order, 1% session risk, 0.25% player risk, YES only, no taker,
Kelly <= 0.10, under a PILOT banner. Paper and research modes are unchanged. A validated production start can save
`production_profile.json` (`--write-production-profile`); later production runs refuse to start if any key setting
differs. Every start prints a STARTUP MANIFEST (trader, engine, code hash, run mode, calibration, weight source,
execution prior source, quote policy, gate status, caps, periods, NO status, CVaR, daily stop, book feed).

**Execution priors are per quote policy.** qp5 starts at 15% of the qp3 mean with a wide s.d. (YES 2.0c, NO 3.0c)
and is updated by its own live base-arm marks by precision weighting (`exec_priors.json`, policy-matched; the
prediction engine version no longer resets execution evidence). Production admission charges the qp5 posterior;
pilot and paper admit on the measured qp3 s.e. until qp5 has 10 orders per side, so evidence can be collected.

**Quoting (qp5).** Admission is computed at every maker level from the bid up to min(ask - 1, mid). The quote is
then: the argmax of P(fill | q) x (alpha + E[M30 | fill, q] - fee) when a validated `exec_model.json` exists (no
quote at all if every level is negative); otherwise bid+1 (join when toxic, NO joins). Controlled exploration: +/-1c
arms at 10% each, one contract, propensity logged, decaying as the execution model's sample grows. The v2026.35
midpoint walk is research only.

**Execution model.** `python3 fit_execution.py` fits, from LIVE qp5 quotes only, a logistic P(fill) (inverse-
propensity weighted) and a linear E[30s markout | fill] on event-free marks, with features side, spread, quote level,
distance from mid, price, model-market gap, Q4, stat. The markout model is not allowed to load on the model-market
gap (the directional alpha is added separately; loading it would count the absorbed part of the edge twice).
Validation is strictly walk-forward (all dates but the last ->
the last); it is used only if both parts beat constant baselines with >= 300 quotes, 40 fills, 30 marks, 10 held-out
fills.

**Sizing.** Integer marginal ascent replaces flooring: from the floor, repeatedly add the single contract (0->1,
1->2, ...) or swap (i -> j) that most improves the same objective under every cap; CVaR is re-checked after. Model-
risk factors now include team rotation and player role. `--kelly-mean posterior` (auto once the execution model is
fitted) feeds the optimizer the posterior mean while the 90% LCB stays the admission gate.

**Quote lifetime and fast cancels.** Life depends on the stoppage: timeout 12s, substitution 8s, free throw 5s,
period break 45s, instant replay 15s, other 8s; the exchange expiry is 3s later. Every `--watch` tick checks each
resting quote against the book with OUR OWN resting size removed: cancel if the ask side moved >= 1c, another trader
now bids above us, a joined quote's other-trader depth fell below 25%, or the book was unreadable 3 checks in a row
(one failed read is no information). Pilot runs hold positions to settlement (exits would be taker sells).

**Kalshi WebSocket (`--ws`).** A local L2 book per tracked ticker (orderbook_snapshot + sequenced orderbook_delta);
any sequence gap marks the book stale and re-snapshots; stale or quiet books fall back to REST. It feeds the pre-send
check, fast cancels and markouts. Orders still go over REST. Verify on your connection first:
`python3 kalshi_nba_inplay_trader.py --ws-selftest` (needs `pip3 install websocket-client`).

**Model authority and books.** Calibration cells are judged strictly walk-forward (fit on every game before a date,
score the games on that date; no leave-one-out across time), now including (stat, phase, gap-bin) cells, so the
model weight can depend on where its disagreement with the market has paid. Sportsbook weights are fitted per
(stat, phase) cell on the simplex, also walk-forward.

**Ladder discipline.** A player's strikes are one distribution: if the pooled YES probabilities are not
non-increasing in the line, that player-stat is skipped for the cycle; otherwise only the single best strike per
(player, stat, side) goes to the optimizer.

**Settled positions** (market closed, exchange flat) are removed from open risk at each exchange check.

**Not in this release (engine rebuild; both specialists placed them after the five execution items):**
substitution/return hazards by coach and player, score-dependent rotations inside each simulated path, D&T
calibration inside the event intensities, full foul/free-throw/end-game process, and a joint ladder CDF fit.

## 5t. Local authority, order integrity and fill-aware economics (v2026.37, quote policy qp6, engine unchanged)

The economic objective everywhere is now **expected realized P&L = P(fill) × E[profit | fill]**, with
E[profit | fill] = directional alpha + execution residual − fee. Hard risk limits still assume every resting order fills.

### LOCAL AUTHORITY AND THREE-TIER CAPITAL POLICY

**A global model weight of 0 does NOT mean zero trading, if a local cell independently earns authority.**

`fit_calibration.py` fits a model weight for every cell of the hierarchy
`*` → `STAT` → `*|PHASE` → `STAT|PHASE` → `*|PHASE|GAP` → `STAT|PHASE|GAP`
(phases Q1, Q2, HT, Q3, Q4; gap bins g0-3, g3-6, g6-10 cents; a gap above 10c is never a cell). Each cell:

- is fitted against **market-only** (w = 0) on its own rows. It never needs the global weight to be earned;
- is shrunk toward its parent: a ridge fit toward the parent weight, then `w = λ·w_fit + (1−λ)·w_parent` with
  λ = n_eff / (n_eff + 150), n_eff = min(markets, 15 × games). Where a cell has two parents (STAT|PHASE: STAT and *|PHASE),
  the smaller parent weight is used;
- is judged on **strictly chronological walk-forward evidence**. For each game date D, the per-stat transforms and every
  hierarchical weight are refitted on games strictly before D, and each game on D is scored. The directional edge at the
  mid is sign(p_cal − p_mkt) × (y − p_mkt) × 100, and the Brier gain is measured at the past-only weight. A future
  game cannot change any earlier decision (tested).

The same strictly chronological standard now authorizes the per-stat transform (a, b) and the sportsbook weights.
Leave-one-game-out is printed as a diagnostic only.

| Tier | Requirements (named constants, printed by `fit_calibration.py`) | Capital |
|---|---|---|
| **PRODUCTION** | ≥ 10 games, ≥ 150 markets, ≥ 3 dates; walk-forward edge ≥ 1.5c; 90% game-bootstrap lower bound > 0; Brier gain > 0; **and** survives Benjamini–Hochberg FDR 5% across every cell tested | normal correlated Kelly × confidence (0.5–1.0) |
| **PILOT** | ≥ 5 games, ≥ 75 markets, ≥ 2 dates (one night never promotes); edge ≥ 1.0c; 80% lower bound ≥ 0; not clearly worse in Brier | **1 contract**, maker only, YES only, no taker, no averaging down, no second pilot on a held player, and within one cycle a pilot yields to any same-player production candidate (one pilot per player); separate budget `--pilot-session-cap 0.01` and `--pilot-player-cap 0.0025` |
| **RESEARCH** | everything else | no live capital (logged; paper only) |

A cell that is RESEARCH only for **lack of data** falls through to its next coarser cell. A cell with enough data that
**failed** blocks its coarser parents, because it is evidence against trading there. The trader's lookup order is
stat|phase|gap, stat|phase, *|phase|gap, *|phase, stat, global.

**Why the FDR control (added beyond the spec):** the hierarchy tests about 50 cells on the same nights. I measured this on
a true null (a calibrated market and an uninformative model, 4 nights × 4 games, 6 seeds):

- About **20% of tested cells pass PILOT by chance**. That is what an 80% bound means, and it is accepted because PILOT is
  the one-contract evidence tier.
- Without the FDR control, the spec's 90% bound alone put several noise cells into PRODUCTION on every refit. With FDR
  0.10, a noise cell still reached PRODUCTION in 1 of 6 refits. Hence FDR 0.05 (`FDR_PROD`).

`calibration.json` now carries `_pool.authority` with weight, weight_fit, parent_weight, λ, games, markets, dates, the
walk-forward edge, its 80% / 90% bounds, Brier gain, stability, p-value, FDR survival, tier and confidence for every cell.
`w`, `stat_w` and `cell_w` remain for compatibility. A pre-v2026.37 calibration is read as PILOT cells only.

`--model-weight` now defaults to 0, so the calibration's authority decides. A manual weight is RESEARCH (paper) in a paper
run, and PILOT in a `--pilot-live` run; a production run refuses it. With authority loaded, the default first period is
Q1, because every phase must earn its own cell.

**Production authorization is cell-aware.** The broad model gate is now **engine sanity**: ≥ 300 graded markets over ≥ 15
games, and the edge's 90% upper bound > 0 (not demonstrably worse than the market). A production run also needs:

- at least one PRODUCTION cell;
- the fill gate passed for **this policy family**;
- the YES markout gate passed;
- the execution prior earned.

### Order-state machine (P0)
Orders are RESTING / PARTIALLY_FILLED / CANCEL_REQUESTED / CANCEL_FAILED / PENDING_TERMINAL, then terminal FILLED /
CANCELLED_CONFIRMED / EXPIRED_CONFIRMED.

- **A failed DELETE** stays live (CANCEL_FAILED): it keeps its full reservation, keeps being fast-monitored, and is retried.
- **A successful DELETE** moves to PENDING_TERMINAL, still reserved.
- **Risk is released only on exchange proof.** Either `GET /portfolio/orders/{id}` reports canceled, executed or expired
  with its fill count fully booked, or, when the status cannot be read, the order's exchange-enforced `expiration_time`
  has passed by 30 s and a successful fills read has happened after that. Elapsed polling cycles are never proof.
- **Fills always book.** A late fill after a cancel request still books. An order the exchange still reports as resting
  goes back to CANCEL_FAILED.
- **The journal survives crashes.** It persists every non-terminal order with its state. After a restart, pending orders
  wait for proof, and unknown-state orders are cancelled again and stay reserved meanwhile.

`reserved()` carries contracts as well as cost, and the game, player, stat, line, team and tier tags.

### Policy identity
- **`policy_id`** is a SHA-256 of every economically relevant argument (only operational flags such as --once, --interval,
  --watch and --live are excluded), the trader version, a hash of the quote-policy source, the TTL table, the
  champion execution model's id and the execution prior's id.
- **`policy_family_id`** is the same minus the execution prior id. **This is a deliberate departure from the spec.**
  `exec_priors.json` is re-estimated every night, so a gate pooled by the exact `policy_id` could never accumulate 100
  orders over 10 games.

The production fill gate counts only: LIVE fills, the current engine, quote policy qp6, the **base arm**, and this policy
family. Exploration arms and older policies are reported separately and never certify production. Both ids are logged
on every quote, quote event, fill, markout, eligible candidate, run config and model status.

### Execution model v2 (`kalshi_exec_model.py`, `fit_execution.py`)
- **Labels are censored.** A quote counts as 0 only after a terminal unfilled event. A quote still resting or waiting for
  cancel proof is excluded.
- **Features** (standardized with training-only scalers, saved in the model):
  - TTL and the stoppage type;
  - quote aggressiveness (price − bid) / (ask − bid);
  - depth (log), imbalance and queue-ahead;
  - book age and time since the last NBA action;
  - 250 ms / 1 s mid velocity, spread and imbalance velocity, bid depletion and ask replenishment (from the WebSocket
    history);
  - phase, stat, side.
- **Markouts are residualized.** Every quote's mid is re-marked 30 s after submit (`kalshi_quote_marks_*.csv`), and a
  *drift* model fitted on terminal-unfilled quotes gives the counterfactual market move without our fill. The production
  markout model (`mark_all`, ALL fills) predicts M30 − drift, so convergence toward the model is not counted on top of
  directional alpha. An event-free model (`mark_ef`) is kept as a diagnostic.
- **Validation** uses rolling expanding windows (dates 1..k → k+1) on every later date:
  - fill model: log loss, Brier, calibration intercept and slope, reliability bins;
  - markout model: MSE, MAE, bias;
  - the **doubly-robust value of the model's quoting policy vs the bid+1 and join-the-bid baselines**.
  A model is validated only if its policy beats both baselines.
- **Champion and challenger.** The newest fit is always `challenger_exec_model.json` and is scored in shadow on every
  quote. It replaces `champion_exec_model.json` only if it is validated and beats the champion's policy value on dates the
  champion never trained on. Only the champion trades. (The v2026.36 `exec_model.json` is no longer read.)
- **Uncertainty-aware quote choice.** Game-bootstrap coefficient draws give QEV mean and s.d. per level. PRODUCTION picks
  argmax (QEV − 1.282·sd) and needs QEV_LCB > 0; PILOT needs mean QEV > 0.
- **Exploration** decays with the policy-value uncertainty (`explore_scale`) and is capped at 2% per arm in PRODUCTION.
  Each arm is priced with its own P(fill), residual and QEV.

### Fill-aware sizing
- **The quote's execution mean** is the champion's quote-specific residual shrunk toward the side prior:
  weight n_fills / (n_fills + 30) on the model. The same number drives the edge bound, the quote choice and the QP.
- **The QP's expected return** is μ = confidence × P(fill) × conditional edge / cost.
- **The covariance** is sqrt(φᵢφⱼ)·Cov with diagonal φ·Var + φ(1−φ)μ²; held exposure counts as filled.
- **Caps stay at full notional.** Confidence and P(fill) never shrink the session, game, player, position or pilot caps,
  CVaR or the cash reservation.
- **The pilot budget** is a group constraint in both the continuous and the integer stage.
- **Without a champion model, P(fill) = 1**, the v2026.36 behaviour.

### Fresh book, market clock, CVaR, ladder
- **Fresh-book re-optimization.** After the first QP, every order is re-read on the fresh book. A book that moved more
  than `--revalidate-mid-c`, or is crossed, is dropped. Otherwise the maker ladder is rebuilt and the quote re-chosen by
  the same `price_side()` rule, then re-sized, then the per-order play-by-play check runs before the send.
- **WebSocket market clock (`--ws`).** Every synchronized book update for a market with a live quote runs its fast checks
  immediately on a market-clock thread, instead of waiting for the `--watch` loop. All book state is under one lock.
  Subscriptions are warmed for every in-scope market and pruned with `untrack()`. A sequence gap still falls back to REST.
- **CVaR95 includes shared model-error shocks** (global, stat, game, team, player) on top of the simulated basketball
  outcomes. Fill probability never reduces worst-case exposure. Held, pending and resting items carry their team and
  player tags.
- **Ladder coherence.** One player's strikes are projected onto a non-increasing curve by weighted isotonic regression
  (no longer skipped). The `--max-strikes` (default 1) best strikes per player, stat and side are then ranked by
  P(fill) × conditional edge.

### Opportunity funnel and starvation diagnostic
Every cycle writes `kalshi_funnel_<date>.csv`, which counts each stage:

markets_seen → trusted → valid_book → in_scope → valid_gap → positive_model_signal → local_authority_nonzero →
pilot / production authority → side_enabled → LCB_pass → execution_EV_positive → QP_positive → below_one_contract →
integer_lifted → fresh_book_survivors → CVaR_survivors → PBP_survivors → quotes_posted → fills

`grade_predictions.py` prints the totals with percentages. After 30 minutes without a posted quote the trader prints
**OPPORTUNITY STARVATION** with the blocking reasons by share. This is diagnostic only: **no threshold is ever lowered
automatically.**

`grade_predictions.py` also adds:
- an authority-cell table (Brier for model, market and pool, with intervals);
- a **walk-forward promotion and demotion table** (never automatic; one night always says WAIT);
- results by tier: candidates, quotes, fills, settled P&L, markout and fill rate.

### Pre-release review fixes
Regression tests in `tests/test_v2037.py` cover the WebSocket gap fix (R1), pilot queue place (R2, R2b), CANCEL_FAILED risk (1, 6), fills (R6a–c), backoff (R7) and journal repair. The NO probe limit at send time and the locked snapshot are covered only by code review, not by a dedicated test.
- **WebSocket false gaps:** sequence bookkeeping now runs for every message, including untracked tickers, so a delta on a
  ticker we no longer watch no longer looks like a gap and forces a resubscribe.
- **Pilot quotes kept their queue place:** a pilot's own resting order is not counted as "player already held". Within
  a cycle a pilot is also dropped if the same player has a production candidate or a better pilot, so it is never posted
  and then cancelled on the next cycle.
- **CANCEL_FAILED / CANCEL_REQUESTED orders count as live risk** in caps, cash and CVaR until the exchange proves they
  are terminal.
- **The NO probe limit is enforced** again at the send stage.
- **The main thread reads order state under the lock**, from a snapshot taken once per cycle.
- **Fills are never lost.** A failed fills read (`None`) is not proof of zero fills. Partial fills are booked before an
  order is released. Fills that arrive up to 10 minutes after an order was released are still read and booked.
- **Cancel retries back off** (1, 2, 4 … 30 s). The kill switch trips at 3 distinct orders or 6 attempts on one order,
  so a single slow cancel cannot cause a retry storm.
- **Journal recovery** fills in a missing `expiration_ts` (submitted + TTL + 3 s) and repairs `filled` from the ledger.
- **Network calls happen outside the lock.**

### Not done in this version (spec items 26–32)
These are not implemented, and their tests are reported as SKIP:
- substitution and return hazards;
- score-dependent future rotations;
- fitted phase minute CVs;
- generative D&T calibration inside the event rates;
- richer role posteriors;
- the foul / FT / endgame process;
- the matchup-strength grid.

Each changes simulation probabilities, so each needs `ENGINE_VERSION` 2026.29, which resets every earned weight. The
close-game late-Q4 block stays.

**Nightly workflow:**
1. Run the trader.
2. `python3 grade_predictions.py`
3. `python3 fit_calibration.py`
4. `python3 fit_execution.py`
5. Read the promotion table.

## 5u. Statistical integrity of promotion and evaluation (v2026.38, engine unchanged)
No engine probability changes, so `ENGINE_VERSION` stays 2026.28 and every engine-2026.28 prediction remains evidence.

### One production authorization
`model_status.json["passed"]` is exactly:

**engine sanity** AND **≥ 1 PRODUCTION cell** (calibration.json fitted for this engine) AND **fill gate** (this policy
family) AND **markout gate** (this policy family) AND **execution prior earned** (≥ 30 LIVE base-arm YES orders, this
policy family).

The grader prints each term. The trader re-checks every term against its own run and also requires `passed`. The old
rule (global edge 90% CI low > 0 and mean ≥ 2c) is kept as `global_model_dominance`, a diagnostic that authorizes
nothing.

### Execution evidence belongs to the policy family
- **The markout gate and `exec_priors.json` use only** LIVE, qp6, base-arm marks of the **current `policy_family_id`**.
  Marks from other families are printed separately.
- **`exec_priors.json` records its family.** The trader ignores a prior from another family and falls back to the weak
  hierarchical prior.
- **Fix: sizing settings left the family.** `--pilot-live` forces Kelly, session/player caps, `max_contracts` and
  `--no-exits`, and v2026.37 hashed all of them into the family. So pilot evidence could never authorize production, and
  production could never collect its own evidence. Production could therefore never be authorized.
  - Size and risk-budget settings (`SIZING_ARGS`) now enter `policy_id` but not `policy_family_id`.
  - Everything that decides which quotes are posted, and at what price, is still in the family: edges, gates, TTL,
    quote code, champion execution model, taker/NO switches.
  - Caveat: pilot orders are 1 contract, so their fill selection may differ somewhat from larger production orders.

### A proper null test for FDR
The p-value given to Benjamini–Hochberg is now a **game-clustered sign-flip randomization test** of H0: E[edge] = 0.
- **How it works:** each held-out game's edge contribution is a cluster, and its sign is flipped. It is exact up to 16
  games and Monte Carlo above that.
- **Why:** the previous value, the share of bootstrap draws ≤ 0, was centred on the observed sample and was optimistic.
  The bootstrap is kept for the reported intervals (`p_boot` is logged for comparison).
- **Measured size:** about 4–5% at the 5% level with 8–30 games, slightly conservative with few games.
- **Assumption:** under H0 a game's contribution is symmetric about zero. A strongly left-skewed null could make the test
  somewhat anti-conservative.

### Discovery, then confirmation (`authority_registry.json`)
- **Discovery:** a cell that passes every PRODUCTION rule on today's data (including FDR across all tested cells) is
  frozen as a **CANDIDATE** through today's last date. It is **not** PRODUCTION yet, though it can trade as PILOT on its
  own evidence.
- **Confirmation:** it is decided once, on a **fixed** sample: the first ≥ 6 games over ≥ 2 dates after the freeze,
  which the discovery search never saw. The rule is edge > 0 and sign-flip p ≤ 0.10.
  - Outcome: **CONFIRMED** or **REJECTED**. Later games never reopen the decision.
  - A REJECTED cell can become a fresh candidate only after 2 more dates.
- **Only CONFIRMED cells can be PRODUCTION**, and they must still meet every PRODUCTION rule on the full data each night.
- **Resets:** the registry is per engine, so a new engine starts empty. Delete `authority_registry.json` only if you
  deliberately want to restart confirmation.

### Action support in off-policy evaluation
- **Logging:** every quote now records the behaviour policy over its whole ladder: `pi_b` (JSON {price: probability}),
  `greedy_px`, `action_price` and `action_prob`. The greedy level gets 1 − p_explore × (#arms), each available ±1c arm
  gets p_explore, and every other level gets 0. On the fresh-book pass the probabilities are those of the final book.
- **Evaluation:** doubly robust values count a target action only where π_b(target | X) ≥ 0.05.
  - Values are reported on supported decisions, with the overlap share.
  - Older rows without `pi_b` have known support only at the action actually taken.
- **Validation** of an execution model needs model support ≥ 60%. It also needs a positive **game-clustered 90% lower
  bound** on V(model) − V(bid+1) **and** V(model) − V(join), each on jointly supported decisions with ≥ 60% overlap.
- **Supported quote offsets:** `fit_execution.py` records the quote offsets (price − bid) with ≥ 20 logged LIVE actions.
  A PRODUCTION quote may only use those offsets. Anything else is extrapolation, left to PILOT exploration.

### Champion promotion needs a lower bound
A validated challenger replaces the champion only if, on dates the champion never saw, every one of these holds:
- ≥ 2 dates, ≥ 8 games and ≥ 40 jointly supported decisions;
- action overlap ≥ 60%;
- V(challenger) − V(champion) has mean > 0 and a game-clustered 90% lower bound > 0;
- fill log loss is no worse (tolerance 0.002);
- markout prediction MSE is no worse (tolerance 5%).

The checks and the delta are printed and stored under `promotion` in the challenger file.

### Not done in this version
- The P1 items: queue-progress tracking, fill hazard, continuation-value cancels, residual-alpha model, error
  attribution, deployed-probability grading.
- The residual-markout production gate.
- The engine items (rotation and substitution hazards, score-dependent rotations, phase CVs, D&T calibration inside the
  event rates, extra role posteriors, foul/FT state, fitted live priors). These change probabilities and need engine
  2026.29.

## 6. Measurement

Each night produces four files:
- `kalshi_fills_<date>.csv`: every fill, with model, market and blended probabilities, ROI, Kelly fraction and fees.
- `kalshi_predictions_<date>.csv`: the model and market probability for every priced market, every cycle, including markets not traded.
- `kalshi_clv_<date>.csv`: closing-line value of each entry against the last observed mid.
- `kalshi_eval_<date>.csv`: written by `python3 evaluate_night.py` after settlement. The script reports Brier score and log loss (model vs market vs blend), calibration buckets, per-stat bias, P&L including exits, and CLV.

The edge question is answered by these files, not by a single night's P&L. Two things need to hold over about 200 or more settled bets:
- CLV is consistently positive;
- the model's Brier score is at least as good as the market's on the bets taken.

## 7. Development history: first live night (Nov 5, 2026)

The system went live on two games (ORL@PHX, MIA@GSW) with a $250 bankroll. Live monitoring surfaced several issues that the synthetic tests had not caught. Each was fixed the same night:

| Issue found live | Fix |
|---|---|
| Player-name matching worked for only 19 of 498 markets | Normalized-token, longest-unique-match name parser (488 markets after the fix) |
| A pre-tip clock read as late Q1, so the model thought the game was nearly over | Pre-game state is detected from an empty box score and pinned to 48:00 |
| Fills clustered in 10–20¢ tails | Dispersion shrink (k = 0.88), 20–80¢ price band, 0.15 disagreement gate |
| CLV turned negative as resting maker orders were picked off after news | Winner's-curse shrink, refill cooldown, no averaging down, event-driven cancels, live-ball EV premium |
| No way to take profit | Reduce-only exits |

The final settled results of the first night are produced by `evaluate_night.py`. They are a one-night sample and are not evidence of edge either way.

## 8. Setup for live use

| Credential | Read from |
|---|---|
| D&T API key | macOS Keychain service `DT_API_KEY`, or environment variable `DT_API_KEY` |
| Kalshi API key ID | Keychain service `KALSHI_API_KEY_ID`, or environment variable `KALSHI_API_KEY_ID` |
| Kalshi private key (PEM) | path in environment variable `KALSHI_PEM_PATH` |
| The Odds API key (optional, sportsbook consensus) | Keychain service `ODDS_API_KEY`, or environment variable `ODDS_API_KEY` |

Kalshi holds sports cash in a separate balance (shard 3, Sports/Basketball), and API orders can't move cash into it. Fund shard 3 before trading.

```bash
python3 kalshi_nba_inplay_trader.py --once          # one cycle, then exit
python3 kalshi_nba_inplay_trader.py                 # paper trading (no orders sent)
python3 kalshi_nba_inplay_trader.py --pilot-live    # real money, evidence collection, minimum size (unvalidated config)
python3 kalshi_nba_inplay_trader.py --live          # production: refused unless every authorization check passes
python3 kalshi_nba_inplay_trader.py --ws-selftest   # verify the WebSocket book against REST, then exit
python3 grade_predictions.py && python3 fit_calibration.py && python3 fit_execution.py   # after each night
```

Every limit is a command-line flag (`--help`). Ctrl-C cancels resting orders and writes the CLV report. Run one instance per account.

## 9. Known limitations and next steps

- **No validated edge yet.** Engine 2026.28 has no graded out-of-sample record; production stays locked until it does.
- **Execution model needs data.** Until `fit_execution.py` validates, quotes use bid+1 with exploration arms.
- **WebSocket is opt-in** until `--ws-selftest` passes on your connection; REST polling is the default.
- **Rotation is not hazard-based** and simulated minutes do not react to the simulated score (engine rebuild items).
- **D&T calibration acts on simulated counts**, not inside event rates, so tails can be off while means match.
- **Final two minutes of close games are not traded** (no full foul / free-throw end-game process).
- **One-sided quoting.** The bot bids only; there is no two-sided market making.