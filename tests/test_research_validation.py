"""Offline invariants: as-of boundaries, execution replay and capital preservation."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import economic_validation as EV
import kalshi_exec_model as XM
import kalshi_nba_inplay_trader as T
import nba_markov_engine as MG
from kalshi_portfolio_qp import CorrelatedKellyQP
from risk_metrics import expected_shortfall
from fit_engine_priors import nested_select


def fixture(n=30):
    """Fabricated inputs exclusively for regression tests, never market evidence."""
    rows, trades = [], []
    for i in range(n):
        stamp = pd.Timestamp("2026-01-01T20:00:00Z")+pd.Timedelta(days=i)
        rows.append(dict(timestamp=stamp.isoformat(), state_available_at=(stamp-pd.Timedelta(seconds=1)).isoformat(),
                         label_available_at=(stamp+pd.Timedelta(hours=3)).isoformat(),
                         baseline_trained_through="2025-01-01T00:00:00Z", challenger_trained_through="2025-01-01T00:00:00Z",
                         game=f"g{i}", player="p", stat="PTS", ticker=f"t{i}", side="yes", y_yes=i % 2,
                         p_baseline=.9, p_challenger=.6, price_cents=40, fee_cents=1, bid=39, ask=43,
                         expected_fill_fraction=.5, requested_qty=10, queue_ahead=3, ttl_s=8,
                         book_age_s=.5, data_kind="synthetic", baseline_engine="2026.31", challenger_engine="2026.32"))
        trades.append(dict(trade_id=f"e{i}", timestamp=(stamp+pd.Timedelta(seconds=2)).isoformat(),
                           ticker=f"t{i}", side="yes", price_cents=40, quantity=8, aggressor="sell"))
    return pd.DataFrame(rows), pd.DataFrame(trades)


def roster(team, first=1, n=8):
    return [MG.DTPlayerSkillProfile(first+k, f"p{first+k}", team, 3, 78, 210, 100, 30, .2,
            6, 4, 7, 5, 5, 2, 2, 7, 1, 1, .64, .42, .36, .78, 1, 0, 1) for k in range(n)]


def state():
    return MG.LiveInPlayState("g", "A @ H", "5:00", 4, 300, 1, 2, "H", "A", 80, 80,
                             "HOME", [], [], {}, game_status=1)


class ValidationTests(unittest.TestCase):
    def test_zero_kelly_and_invalid_fraction(self):
        c = dict(ticker="x", side="yes", win_prob=.9, price_cents=40, fee_cents=0,
                 game_key="g", player_name="p", outcome=np.array([1, 0]*50))
        self.assertEqual(CorrelatedKellyQP(kelly_fraction=0).solve_allocations([c], 1000, 100, {}), [])
        c["max_contracts"] = 0
        self.assertEqual(CorrelatedKellyQP(kelly_fraction=.1).solve_allocations([c], 1000, 100, {}), [])
        for k in (-.1, float("nan"), float("inf"), 1.1):
            with self.assertRaises(ValueError):
                CorrelatedKellyQP(kelly_fraction=k)

    def test_discrete_tail_mass(self):
        losses = np.r_[np.zeros(99), 100.]
        self.assertAlmostEqual(expected_shortfall(losses), 20.)
        self.assertAlmostEqual(expected_shortfall([1, 1, 1]), 1.)
        self.assertAlmostEqual(expected_shortfall([1, 2], 0), 1.5)

    def test_cvar_missing_and_tied_scenarios(self):
        orders = [dict(contracts=100, unit_c=100, expected_edge_c=5, outcome=None)]
        out, diag = T.cvar_scale(orders, [], 1000, .02)
        self.assertLessEqual(sum(o["contracts"] for o in out), 20)
        self.assertLessEqual(diag["cvar95"], 20)
        orders[0]["outcome"] = np.array([])
        self.assertEqual(T.cvar_scale(orders, [], 1000)[0], [])
        orders[0]["outcome"] = np.r_[np.ones(99), 0.]
        out, diag = T.cvar_scale(orders, [], 1000, .001)
        self.assertLessEqual(diag["cvar95"], 1.)

    def test_cvar_couples_games_independently(self):
        # Complementary player outcomes hedge only if they belong to the same game.
        y = np.array([0, 1]*200)
        a = dict(contracts=100, unit_c=50, game_key="g1", outcome=y)
        b = dict(contracts=100, unit_c=50, game_key="g1", outcome=1-y)
        _, same = T.cvar_scale([a, b], [], 1000, 1)
        _, different = T.cvar_scale([a, dict(b, game_key="g2")], [], 1000, 1)
        self.assertEqual(same["cvar95"], 0.)
        self.assertGreater(different["cvar95"], 50.)

    def test_joint_fraction_uncertainty(self):
        x = XM.exec_features("yes", "PTS", order_qty=10)
        n = len(x)
        coef = lambda v: [v]+[0.]*(n-1)
        m = dict(scaler=dict(mean=[0.]*n, sd=[1.]*n), fill_any=coef(2), fill_fraction=coef(0),
                 mark_all=coef(0), boot_fill=[coef(2)]*4, boot_mark=[coef(0)]*4,
                 boot_fill_fraction=[coef(v) for v in [-4, -2, 2, 4]])
        old = copy.deepcopy(m)
        old.pop("boot_fill_fraction")
        self.assertEqual(XM.quote_value(old, x, 10, 0)["qev_sd"], 0.)
        new = XM.quote_value(m, x, 10, 0, z=1.282)
        self.assertGreater(new["qev_sd"], 3.)
        self.assertLess(new["qev_lcb"], new["qev"])
        self.assertAlmostEqual(new["qev_total"], 10*new["qev"])
        self.assertNotEqual(XM.model_id(old), XM.model_id(m))
        deployable = dict(m, validated=True, quote_policy="qp8", features=XM.EXEC_FEATURES,
                          uncertainty_schema="joint-fill-fraction-v1")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"champion.json"
            p.write_text(json.dumps(deployable))
            self.assertIsNotNone(XM.load_model(str(p), "qp8"))
            deployable.pop("uncertainty_schema")
            p.write_text(json.dumps(deployable))
            self.assertIsNone(XM.load_model(str(p), "qp8"))

    def test_live_requires_explicit_mode_and_credentials(self):
        client = T.KalshiClient("fake", "")  # PAPER remains the default, even with credentials.
        client._private_key = object()
        self.assertFalse(client.can_trade)
        client._session.post = lambda *a, **k: self.fail("paper POST attempted")
        client._session.delete = lambda *a, **k: self.fail("paper DELETE attempted")
        self.assertEqual(client.send_order(None, "yes", 40, 1, True, 8), (None, 0.0))
        self.assertEqual(client.send_exit(None, "yes", 40, 1), (None, 0.0))
        self.assertFalse(client.cancel_order("o", "t"))
        client.paper = False
        self.assertTrue(client.can_trade)  # Capability only; never submit a real order in tests.
        client.key_id = ""
        self.assertFalse(client.can_trade)
        client.key_id = "fake"
        client._private_key = None
        self.assertFalse(client.can_trade)

    def test_asof_and_identity_validation(self):
        d, _ = fixture()
        for col in ("state_available_at", "baseline_trained_through", "challenger_trained_through"):
            bad = d.copy()
            bad.loc[0, col] = "2030-01-01T00:00:00Z"
            with self.assertRaises(ValueError):
                EV.prepare(bad)
        bad = d.copy()
        bad.loc[0, "timestamp"] = "2026-01-01T20:00:00"
        with self.assertRaises(ValueError):
            EV.prepare(bad)
        bad = pd.concat([d, d.iloc[[0]].assign(y_yes=1)])
        with self.assertRaises(ValueError):
            EV.prepare(bad)

    def test_repeated_forecasts_and_overlapping_strikes(self):
        d, _ = fixture()
        repeats = pd.concat([d]*100, ignore_index=True)
        d2 = EV.independent_opportunities(EV.prepare(repeats))
        self.assertEqual(len(d2), len(d))
        other = d.iloc[[0]].assign(ticker="other-strike", price_cents=41)
        self.assertEqual(len(EV.independent_opportunities(EV.prepare(pd.concat([d, other])))), len(d))

    def test_future_labels_never_change_earlier_predictions(self):
        d, _ = fixture()
        a, folds = EV.walk_forward(EV.prepare(d))
        d.loc[25:, "y_yes"] = 1-d.loc[25:, "y_yes"]
        b, _ = EV.walk_forward(EV.prepare(d))
        np.testing.assert_array_equal(a.p_candidate[:26], b.p_candidate[:26])
        late = EV.prepare(d)
        late.loc[:24, "label_available_at"] = pd.Timestamp("2030-01-01T00:00:00Z")
        _, folds = EV.walk_forward(late)
        self.assertEqual(folds[25]["train_games"], 0)

    def test_queue_latency_ttl_and_duplicate_trades(self):
        d, t = fixture(1)
        r = EV.prepare(d).iloc[0]
        t = EV.prepare_trades(t)
        self.assertEqual(EV.replay_fill(r, 10, t), 5)
        self.assertEqual(EV.replay_fill(r, 10, t, latency_ms=2000), 0)
        self.assertEqual(EV.replay_fill(r, 10, t.assign(aggressor="buy")), 0)
        self.assertEqual(EV.replay_fill(r, 10, t.assign(price_cents=39)), 0)
        self.assertEqual(EV.replay_fill(r, 10, t.assign(timestamp=r.timestamp+pd.Timedelta(seconds=8))), 0)
        with self.assertRaises(ValueError):
            EV.prepare_trades(pd.concat([t, t]))

    def test_pending_capital_and_drawdown_stop(self):
        d, t = fixture(30)
        d = EV.prepare(d)
        # Three same-time opportunities share player risk, including unfilled reserves.
        first = pd.concat([d.iloc[[0]].assign(ticker=f"x{i}", game=f"g{i}") for i in range(3)], ignore_index=True)
        out, _ = EV.policy_replay(first, EV.prepare_trades(t.iloc[:0]), "p_baseline")
        self.assertLessEqual((out.requested*.41).sum(), 10.)
        lost = d.assign(y_yes=0, queue_ahead=0)
        full = EV.prepare_trades(t.assign(quantity=100))
        out, metrics = EV.policy_replay(lost, full, "p_baseline")
        self.assertEqual(out.iloc[-1].requested, 0)
        self.assertLessEqual(metrics["max_settled_drawdown_usd"], 35.)

    def test_report_and_single_cluster(self):
        d, t = fixture()
        report = EV.evaluate(d, t)
        json.dumps(report, allow_nan=False)
        self.assertFalse(report["production_authorized"])
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertEqual(report["independent_opportunities"], 30)
        self.assertIsNotNone(report["mean_daily_net_delta_usd"])
        self.assertEqual(set(report["stress"]), {"latency_2s", "double_fees", "half_fills", "model_error_8pct", "all_selected_contracts_lose", "double_queue_ahead"})
        self.assertIsNone(EV.evaluate(d.iloc[:1], t.iloc[:1])["mean_daily_net_delta_usd"])

    def test_availability_is_joint_and_respects_floor_time(self):
        h, a = roster(1), roster(2, 101)
        h[0].p_active = 0
        s = state()
        engine = MG.InPlayMarkovEngine(s, h, a, 0, 0)
        sim = engine.run_monte_carlo(100, seed=42)
        idx = sim["id_to_idx"][h[0].player_id]
        self.assertEqual(sim["PTS"][:, idx].sum(), 0)
        self.assertEqual(sim["AST"][:, idx].sum(), 0)
        self.assertEqual(sim["MIN_PROJ"].get(h[0].player_id, 0), 0)
        self.assertAlmostEqual(sum(sim["MIN_PROJ"].get(p.player_id, 0) for p in h), 25)
        np.testing.assert_array_equal(sim["PRA"], sim["PTS"]+sim["REB"]+sim["AST"])
        np.testing.assert_array_equal(sim["PTS"], engine.run_monte_carlo(100, seed=42)["PTS"])
        # Positive live minutes supersede a stale pregame zero-availability estimate.
        s.live_box = {h[0].player_id: dict(MIN=10, PTS=5)}
        s.game_status = 2
        sim = MG.InPlayMarkovEngine(s, h, a, 0, 0).run_monte_carlo(100)
        self.assertGreater(sim["MIN_PROJ"][h[0].player_id], 10)

    def test_outer_holdout_never_selects_priors(self):
        def score(name, value, train, test):
            return [(value-10)**2 if test != "d4" else value]
        def altered(name, value, train, test):
            return [(value-10)**2 if test != "d4" else -value*1000]
        a = nested_select(score, ["d1", "d2", "d3", "d4"])
        b = nested_select(altered, ["d1", "d2", "d3", "d4"])
        self.assertEqual(a["params"], b["params"])
        self.assertFalse(a["validated"])
        self.assertEqual(a["selected_through"], "d3")

    def test_clutch_selection_uses_simulated_margin(self):
        h, a = roster(1), roster(2, 101)
        st = state()
        st.home_score = 91  # Starts outside clutch; simulated games can become close.
        engine = MG.InPlayMarkovEngine(st, h, a, 0, 0)
        original, used = engine._compile_pair, set()
        class TracedRows(list):
            def __init__(self, rows, flag):
                super().__init__(rows)
                self.flag = flag
            def __getitem__(self, key):
                used.add(self.flag)
                return super().__getitem__(key)
        def compile_pair(h, a, clutch, *args):
            pair = original(h, a, clutch, *args)
            for side in pair:
                side.P = TracedRows(side.P, clutch)
            return pair
        engine._compile_pair = compile_pair
        engine.run_monte_carlo(100, seed=42)
        self.assertEqual(used, {False, True})

    def test_rotation_never_inserts_zero_availability(self):
        r = roster(1, n=6)
        sched = MG.InPlayMarkovEngine._schedule(r, np.array([0, 5, 5, 5, 5, 5.]), [], 300, False)
        self.assertTrue(all(len(s) == 5 and 0 not in s for s in sched))
        with self.assertRaises(ValueError):
            MG.InPlayMarkovEngine._schedule(r, np.array([0, 0, 5, 5, 5, 5.]), [], 300, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
