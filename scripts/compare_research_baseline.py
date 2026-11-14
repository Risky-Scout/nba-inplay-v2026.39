#!/usr/bin/env python3
"""Reproduce engineered failure cases against a pinned Git revision. Not a backtest."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import economic_validation as EV
import kalshi_exec_model as XM
import nba_markov_engine as MG
from kalshi_portfolio_qp import CorrelatedKellyQP
from risk_metrics import expected_shortfall
from test_research_validation import fixture, roster, state


def load_old(revision, filename, directory):
    content = subprocess.check_output(["git", "show", f"{revision}:{filename}"], cwd=ROOT)
    path = Path(directory)/filename
    path.write_bytes(content)
    name = "baseline_"+path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare(revision):
    with tempfile.TemporaryDirectory() as tmp:
        old_qp = load_old(revision, "kalshi_portfolio_qp.py", tmp)
        old_xm = load_old(revision, "kalshi_exec_model.py", tmp)
        old_mg = load_old(revision, "nba_markov_engine.py", tmp)
        c = dict(ticker="T", side="yes", win_prob=.9, price_cents=40, fee_cents=0,
                 game_key="G", player_name="P", outcome=np.array([0, 1]*50))
        old_size = old_qp.CorrelatedKellyQP(kelly_fraction=0).solve_allocations([c], 1000, 100, {})
        new_size = CorrelatedKellyQP(kelly_fraction=0).solve_allocations([c], 1000, 100, {})
        losses = np.r_[np.zeros(99), 100.]
        old_tail = float(losses[losses >= np.quantile(losses, .95)].mean())
        x = XM.exec_features("yes", "PTS", order_qty=10)
        n = len(x)
        coef = lambda v: [v]+[0.]*(n-1)
        model = dict(scaler=dict(mean=[0.]*n, sd=[1.]*n), fill_any=coef(2), fill_fraction=coef(0),
                     mark_all=coef(0), boot_fill=[coef(2)]*4, boot_mark=[coef(0)]*4,
                     boot_fill_fraction=[coef(v) for v in [-4, -2, 2, 4]])
        h, a = roster(1), roster(2, 101)
        h[0].p_active = 0
        old_sim = old_mg.InPlayMarkovEngine(state(), h, a, 0, 0).run_monte_carlo(400, seed=42)
        new_sim = MG.InPlayMarkovEngine(state(), h, a, 0, 0).run_monte_carlo(400, seed=42)
        idx = new_sim["id_to_idx"][h[0].player_id]
        d, _ = fixture()
        repeats = EV.prepare(__import__("pandas").concat([d]*100, ignore_index=True))
        return dict(baseline_revision=revision, data_kind="synthetic_engineered_failure_cases",
                    economic_out_of_sample_evidence=False, profitability_claim=False,
                    checks={
                        "contracts_with_zero_kelly": dict(baseline=sum(o["contracts"] for o in old_size), challenger=sum(o["contracts"] for o in new_size)),
                        "cvar95_99_zeros_one_100_loss": dict(baseline=old_tail, challenger=expected_shortfall(losses), exact=20.),
                        "qev_sd_with_uncertain_fill_fraction_cents": dict(baseline=old_xm.quote_value(model, x, 10, 0)["qev_sd"], challenger=XM.quote_value(model, x, 10, 0)["qev_sd"]),
                        "zero_availability_player_mean_future_points": dict(baseline=float(old_sim["PTS"][:,idx].mean()), challenger=float(new_sim["PTS"][:,idx].mean()), seed=42, simulations=400),
                        "repeated_forecasts": dict(input_rows=len(repeats), independent_families=len(EV.independent_opportunities(repeats)))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="0f20c6c3c29d8f8c215a11024efb7c734b99d74d")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).write_text(json.dumps(compare(args.baseline), indent=2, allow_nan=False)+"\n")
