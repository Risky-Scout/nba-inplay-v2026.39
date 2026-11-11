#!/usr/bin/env python3
"""Hiring-manager audit for the NBA in-play pricing / market-making system.

Turns timestamped research logs into an auditable reviewer package:
1) predictive calibration vs the market (Brier + log loss),
2) breakdowns by stat, proposed OVER/UNDER directive, and game phase,
3) every settled trade-policy-eligible opportunity,
4) counterfactual P&L for opportunities the system did and did not quote,
5) gate attribution without cherry-picking only profitable misses.

Anti-hindsight rule:
- The raw decision audit keeps every timestamped eligible observation.
- Headline economic summaries deduplicate to the FIRST eligible observation for
  each (ticker, side), so a fast polling loop cannot multiply one idea.
- "Missed" means no quote was posted within the matching window after that
  timestamp-valid opportunity. Profitable AND losing misses are reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_results import load_predictions, settlements, phase_of_rows
from nba_markov_engine import ENGINE_VERSION

EPS = 1e-12
HERE = Path(__file__).resolve().parent


def _read(pattern: str) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(HERE / pattern))):
        try:
            d = pd.read_csv(p)
            d["source_file"] = os.path.basename(p)
            frames.append(d)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bool(s: pd.Series, default=False) -> pd.Series:
    return s.fillna(default).astype(str).str.lower().isin(["true", "1", "yes"])


def _num(d: pd.DataFrame, c: str, default=np.nan) -> pd.Series:
    if c not in d.columns:
        return pd.Series(default, index=d.index, dtype=float)
    return pd.to_numeric(d[c], errors="coerce")


def _logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _brier(p, y) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def _ece(p, y, bins=10) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    if not len(p):
        return float("nan")
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        ix = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if ix.any():
            out += ix.mean() * abs(float(p[ix].mean()) - float(y[ix].mean()))
    return float(out)


def _metric_row(d: pd.DataFrame) -> dict:
    if d.empty:
        return {}
    pm = _num(d, "p_dir")
    pk = _num(d, "p_mkt_dir")
    y = _num(d, "y_dir")
    ok = pm.notna() & pk.notna() & y.notna()
    pm, pk, y = pm[ok], pk[ok], y[ok]
    if not len(pm):
        return {}
    mb, kb = _brier(pm, y), _brier(pk, y)
    ml, kl = _logloss(pm, y), _logloss(pk, y)
    return {
        "N": int(len(pm)),
        "Model_Brier": mb,
        "Market_Brier": kb,
        "Brier_Improvement_vs_Market": kb - mb,
        "Model_LogLoss": ml,
        "Market_LogLoss": kl,
        "LogLoss_Improvement_vs_Market": kl - ml,
        "Model_ECE_10bin": _ece(pm, y),
        "Avg_Model_Prob": float(pm.mean()),
        "Avg_Market_Prob": float(pk.mean()),
        "Actual_Hit_Rate": float(y.mean()),
    }


def _group_metrics(d: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    if d.empty:
        return pd.DataFrame()
    for key, g in d.groupby(cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = {c: v for c, v in zip(cols, key)}
        row.update(_metric_row(g))
        row["View"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def predictive_audit(res: dict[str, str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    try:
        d = load_predictions()
    except SystemExit:
        return pd.DataFrame(), {}
    d = d[d["engine"].astype(str) == ENGINE_VERSION].copy()
    if d.empty:
        return d, {}

    d["settlement"] = d["ticker"].map(res)
    d["y_yes"] = d["settlement"].map({"yes": 1.0, "no": 0.0})
    d = d.dropna(subset=["y_yes"]).copy()
    if d.empty:
        return d, {}

    d["p_model_yes"] = _num(d, "p_model_yes")
    d["p_mkt_yes"] = _num(d, "p_mkt_yes")
    d = d.dropna(subset=["p_model_yes", "p_mkt_yes"]).copy()
    d["directive"] = np.where(d["p_model_yes"] >= d["p_mkt_yes"], "OVER", "UNDER")
    d["p_dir"] = np.where(d["directive"] == "OVER", d["p_model_yes"], 1.0 - d["p_model_yes"])
    d["p_mkt_dir"] = np.where(d["directive"] == "OVER", d["p_mkt_yes"], 1.0 - d["p_mkt_yes"])
    d["y_dir"] = np.where(d["directive"] == "OVER", d["y_yes"], 1.0 - d["y_yes"])
    d["phase"] = phase_of_rows(d)
    d["model_minus_market"] = d["p_dir"] - d["p_mkt_dir"]

    tables = {
        "Calibration Overall": pd.DataFrame([{"View": "Overall", **_metric_row(d)}]),
        "Calibration by Stat": _group_metrics(d, ["stat"], "Stat"),
        "Calibration by Direction": _group_metrics(d, ["directive"], "Directive"),
        "Calibration Stat x Dir": _group_metrics(d, ["stat", "directive"], "Stat x Directive"),
        "Calibration by Phase": _group_metrics(d, ["phase"], "Phase"),
        "Calibration Stat x Phase": _group_metrics(d, ["stat", "phase"], "Stat x Phase"),
    }
    return d, tables


def _nearest_posted(eligible: pd.DataFrame, quotes: pd.DataFrame, window_s: float) -> pd.DataFrame:
    e = eligible.copy()
    e["_t"] = pd.to_datetime(e.get("time"), errors="coerce")
    e["posted_within_window"] = False
    e["matched_order_id"] = None
    e["posted_delay_s"] = np.nan
    if quotes.empty or "time" not in quotes.columns:
        return e

    q = quotes.copy()
    q["_tq"] = pd.to_datetime(q["time"], errors="coerce")
    q = q.dropna(subset=["_tq"])
    if q.empty:
        return e

    groups = {(str(k[0]), str(k[1])): g.sort_values("_tq")
              for k, g in q.groupby(["ticker", "side"], dropna=False)}
    for ix, r in e.iterrows():
        if pd.isna(r["_t"]):
            continue
        g = groups.get((str(r.get("ticker")), str(r.get("side"))))
        if g is None or g.empty:
            continue
        dt = (g["_tq"] - r["_t"]).dt.total_seconds()
        cand = g[(dt >= -2.0) & (dt <= window_s)]
        if cand.empty:
            continue
        j = (cand["_tq"] - r["_t"]).dt.total_seconds().abs().idxmin()
        delay = float((q.loc[j, "_tq"] - r["_t"]).total_seconds())
        e.at[ix, "posted_within_window"] = True
        e.at[ix, "matched_order_id"] = q.loc[j].get("order_id")
        e.at[ix, "posted_delay_s"] = delay
    return e


def opportunity_audit(res: dict[str, str], match_window_s: float = 60.0):
    e = _read("kalshi_eligible_*.csv")
    if e.empty:
        return e, pd.DataFrame(), pd.DataFrame()
    if "engine" in e.columns:
        e = e[e["engine"].astype(str) == ENGINE_VERSION].copy()
    if e.empty:
        return e, pd.DataFrame(), pd.DataFrame()

    e["settlement"] = e["ticker"].map(res)
    e["y_yes"] = e["settlement"].map({"yes": 1.0, "no": 0.0})
    e["quote_price"] = _num(e, "quote_price")
    e["fee_c"] = _num(e, "fee_c", 0.0).fillna(0.0)
    e["p_used"] = _num(e, "p_used")
    e["p_model"] = _num(e, "p_model")
    e["p_mkt"] = _num(e, "p_mkt")
    e["side"] = e["side"].astype(str).str.lower()
    e["directive"] = np.where(e["side"].eq("yes"), "OVER", "UNDER")
    e["settled"] = e["y_yes"].notna()
    e["won"] = np.where(e["settled"], np.where(e["side"].eq("yes"), e["y_yes"], 1.0 - e["y_yes"]), np.nan)
    e["counterfactual_net_c"] = np.where(
        e["settled"],
        e["won"] * 100.0 - e["quote_price"] - e["fee_c"],
        np.nan,
    )
    e["counterfactual_pnl_$"] = e["counterfactual_net_c"] / 100.0
    if "passes_conservative" in e.columns:
        e["passes_conservative"] = _bool(e["passes_conservative"])
    else:
        e["passes_conservative"] = True

    q = _read("kalshi_quotes_*.csv")
    e = _nearest_posted(e, q, match_window_s)

    def reason(r):
        if bool(r.get("posted_within_window", False)):
            return "POSTED"
        if not bool(r.get("passes_conservative", True)):
            return "EXECUTION_EV_FILTER"
        return "NOT_POSTED_AFTER_ELIGIBILITY"

    e["decision_reason"] = e.apply(reason, axis=1)
    e["counterfactual_class"] = np.select(
        [
            ~e["settled"],
            e["posted_within_window"],
            e["counterfactual_net_c"] > 0,
            e["counterfactual_net_c"] < 0,
        ],
        ["UNSETTLED", "POSTED", "MISSED_PROFITABLE", "MISSED_LOSING"],
        default="MISSED_FLAT",
    )

    e["_t"] = pd.to_datetime(e.get("time"), errors="coerce")
    first = e.sort_values(["_t", "ticker", "side"], kind="mergesort").drop_duplicates(["ticker", "side"], keep="first").copy()
    settled_first = first[first["settled"]].copy()

    economic = []
    segments = [
        ("ALL_FIRST_OPPORTUNITIES", settled_first),
        ("POSTED", settled_first[settled_first["posted_within_window"]]),
        ("MISSED", settled_first[~settled_first["posted_within_window"]]),
        ("MISSED_PROFITABLE", settled_first[settled_first["counterfactual_class"] == "MISSED_PROFITABLE"]),
        ("MISSED_LOSING", settled_first[settled_first["counterfactual_class"] == "MISSED_LOSING"]),
    ]
    for name, g in segments:
        if g.empty:
            economic.append({"Segment": name, "N": 0})
            continue
        economic.append({
            "Segment": name,
            "N": int(len(g)),
            "Win_Rate": float(g["won"].mean()),
            "Mean_Net_c_per_Contract": float(g["counterfactual_net_c"].mean()),
            "Total_1Contract_PnL_$": float(g["counterfactual_pnl_$"].sum()),
            "Avg_Model_Prob_Used": float(g["p_used"].mean()) if g["p_used"].notna().any() else np.nan,
            "Avg_Quoted_Price_c": float(g["quote_price"].mean()),
        })
    economic = pd.DataFrame(economic)

    missed = settled_first[~settled_first["posted_within_window"]]
    if missed.empty:
        gate = pd.DataFrame()
    else:
        gate = missed.groupby("decision_reason", dropna=False).agg(
            Rejected=("ticker", "size"),
            Profitable_Misses=("counterfactual_net_c", lambda x: int((x > 0).sum())),
            Losing_Misses=("counterfactual_net_c", lambda x: int((x < 0).sum())),
            Mean_Counterfactual_Net_c=("counterfactual_net_c", "mean"),
            Counterfactual_1Contract_PnL_USD=("counterfactual_pnl_$", "sum"),
        ).reset_index()
        gate = gate.rename(columns={"Counterfactual_1Contract_PnL_USD": "Counterfactual_1Contract_PnL_$"})

    return e, economic, gate


def decision_ledger_audit(res: dict[str, str]):
    d = _read("kalshi_decisions_*.csv")
    if d.empty:
        return d, pd.DataFrame()
    if "engine" in d.columns:
        d = d[d["engine"].astype(str) == ENGINE_VERSION].copy()
    if d.empty:
        return d, pd.DataFrame()
    d["settlement"] = d["ticker"].map(res) if "ticker" in d.columns else None
    d["y_yes"] = d["settlement"].map({"yes": 1.0, "no": 0.0}) if "settlement" in d.columns else np.nan
    d["side"] = d.get("side", pd.Series("", index=d.index)).astype(str).str.lower()
    d["directive"] = np.where(d["side"].eq("yes"), "OVER", "UNDER")
    d["quote_price"] = _num(d, "quote_price")
    d["fee_c"] = _num(d, "fee_c", 0.0).fillna(0.0)
    d["settled"] = d["y_yes"].notna()
    d["won"] = np.where(d["settled"], np.where(d["side"].eq("yes"), d["y_yes"], 1.0 - d["y_yes"]), np.nan)
    priced = d["settled"] & d["quote_price"].notna()
    d["counterfactual_net_c"] = np.where(priced, d["won"] * 100.0 - d["quote_price"] - d["fee_c"], np.nan)
    d["counterfactual_pnl_$"] = d["counterfactual_net_c"] / 100.0
    if "reason" not in d.columns:
        d["reason"] = "UNKNOWN"
    if "stage" not in d.columns:
        d["stage"] = "UNKNOWN"
    attr = d.groupby(["stage","reason"], dropna=False).agg(
        Decisions=("ticker","size"),
        Settled=("settled","sum"),
        Priced_Settled=("counterfactual_net_c", lambda x: int(x.notna().sum())),
        Profitable=("counterfactual_net_c", lambda x: int((x > 0).sum())),
        Losing=("counterfactual_net_c", lambda x: int((x < 0).sum())),
        Mean_Counterfactual_Net_c=("counterfactual_net_c","mean"),
        Counterfactual_1Contract_PnL_USD=("counterfactual_pnl_$","sum"),
    ).reset_index()
    attr = attr.rename(columns={"Counterfactual_1Contract_PnL_USD": "Counterfactual_1Contract_PnL_$"})
    return d, attr


def funnel_audit() -> pd.DataFrame:
    return _read("kalshi_funnel_*.csv")


def _style_workbook(path: Path):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="D9EAF7")
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col in range(1, ws.max_column + 1):
            vals = [str(ws.cell(r, col).value or "") for r in range(1, min(ws.max_row, 200) + 1)]
            width = min(45, max(10, max((len(v) for v in vals), default=10) + 2))
            ws.column_dimensions[get_column_letter(col)].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top")
    wb.save(path)


def _write_markdown(path: Path, tables, economic, gate, decision_attr, pred_rows, opp_rows, settled_opp, decision_rows):
    def md_table(d: pd.DataFrame, n=30):
        if d is None or d.empty:
            return "_No settled evidence yet._"
        try:
            return d.head(n).to_markdown(index=False, floatfmt=".4f")
        except Exception:
            return d.head(n).to_string(index=False)

    overall = tables.get("Calibration Overall", pd.DataFrame())
    lines = [
        "# Hiring Manager Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Engine: {ENGINE_VERSION}",
        "",
        "## Interpretation",
        "",
        "This report separates predictive skill, market-relative skill, and economic decision quality.",
        "Counterfactual missed opportunities include profitable and losing misses; they are not cherry-picked.",
        "Headline opportunity economics use the first timestamp-valid observation per ticker/side.",
        "",
        "## Coverage",
        "",
        f"- Settled prediction rows scored: {pred_rows:,}",
        f"- Eligible decision observations audited: {opp_rows:,}",
        f"- Settled eligible observations: {settled_opp:,}",
        f"- Side-level decision ledger rows: {decision_rows:,}",
        "",
        "## Overall probability quality",
        "",
        md_table(overall),
        "",
        "## Economic decision summary",
        "",
        md_table(economic),
        "",
        "## Missed-opportunity / gate attribution",
        "",
        md_table(gate),
        "",
        "## Full decision-gate attribution",
        "",
        md_table(decision_attr),
        "",
        "## Calibration by stat",
        "",
        md_table(tables.get("Calibration by Stat")),
        "",
        "## Calibration by OVER / UNDER directive",
        "",
        md_table(tables.get("Calibration by Direction")),
        "",
        "## Calibration by stat x directive",
        "",
        md_table(tables.get("Calibration Stat x Dir")),
        "",
        "## Calibration by game phase",
        "",
        md_table(tables.get("Calibration by Phase")),
        "",
        "## Reviewer guidance",
        "",
        "- Positive Brier_Improvement_vs_Market means the model had lower Brier score than the market.",
        "- Positive LogLoss_Improvement_vs_Market means the model had lower log loss than the market.",
        "- MISSED_PROFITABLE and MISSED_LOSING are both shown. A useful gate should improve accepted decisions, not merely remove losers in hindsight.",
        "- Counterfactual P&L assumes one contract at the logged quote price and fee. It is a decision-quality diagnostic, not evidence that the order would have filled.",
        "",
    ]
    path.write_text("\\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="review_outputs")
    ap.add_argument("--match-window-seconds", type=float, default=60.0)
    args = ap.parse_args()

    out_dir = HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_raw = _read("kalshi_predictions_*.csv")
    elig_raw = _read("kalshi_eligible_*.csv")
    decision_raw = _read("kalshi_decisions_*.csv")
    tickers = set()
    for d in (pred_raw, elig_raw, decision_raw):
        if not d.empty and "ticker" in d.columns:
            tickers.update(d["ticker"].dropna().astype(str).unique())
    res = settlements(sorted(tickers)) if tickers else {}

    pred, tables = predictive_audit(res)
    opp, economic, gate = opportunity_audit(res, args.match_window_seconds)
    decisions, decision_attr = decision_ledger_audit(res)
    funnel = funnel_audit()

    stamp = datetime.now().strftime("%Y-%m-%d")
    xlsx = out_dir / f"NBA_Market_Maker_Audit_{stamp}.xlsx"
    md = out_dir / f"NBA_Market_Maker_Audit_{stamp}.md"
    js = out_dir / f"NBA_Market_Maker_Audit_{stamp}.json"

    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for name, table in tables.items():
            table.to_excel(w, sheet_name=name[:31], index=False)
        economic.to_excel(w, sheet_name="Economic Summary", index=False)
        gate.to_excel(w, sheet_name="Gate Attribution", index=False)
        decision_attr.to_excel(w, sheet_name="Decision Gate Attribution", index=False)
        if not decisions.empty:
            keepd = [x for x in ["time","game","clock","player","stat","directive","line","ticker","side","phase","period","stage","reason","p_model","p_mkt","p_used","p_book","gap","yes_bid","yes_ask","quote_price","fee_c","exec_ev_c","fill_prob","tier","settlement","won","counterfactual_net_c","counterfactual_pnl_$","run_id"] if x in decisions.columns]
            decisions[keepd].to_excel(w, sheet_name="Decision Ledger", index=False)
        if not opp.empty:
            show = [c for c in [
                "time","game","player","stat","directive","line","ticker","side","period","phase",
                "quote_price","fee_c","p_model","p_mkt","p_used","net_edge_c","cons_edge_c","fill_prob",
                "passes_conservative","posted_within_window","posted_delay_s","matched_order_id",
                "decision_reason","settlement","won","counterfactual_net_c","counterfactual_pnl_$",
                "counterfactual_class","tier","cell","run_id"
            ] if c in opp.columns]
            opp[show].to_excel(w, sheet_name="Decision Audit", index=False)
            missed = opp[opp["counterfactual_class"].isin(["MISSED_PROFITABLE","MISSED_LOSING","MISSED_FLAT"])]
            missed[show].to_excel(w, sheet_name="Missed Opportunities", index=False)
        if not pred.empty:
            cols = [c for c in [
                "timestamp","ticker","clock","player","stat","line","phase","directive",
                "p_model_yes","p_mkt_yes","p_dir","p_mkt_dir","y_dir","model_minus_market","settlement","engine"
            ] if c in pred.columns]
            pred[cols].to_excel(w, sheet_name="Settled Predictions", index=False)
        if not funnel.empty:
            funnel.to_excel(w, sheet_name="Funnel Raw", index=False)

    _style_workbook(xlsx)
    settled_opp = int(opp["settled"].sum()) if not opp.empty else 0
    _write_markdown(md, tables, economic, gate, decision_attr, len(pred), len(opp), settled_opp, len(decisions))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "engine": ENGINE_VERSION,
        "settled_prediction_rows": int(len(pred)),
        "eligible_observations": int(len(opp)),
        "settled_eligible_observations": settled_opp,
        "decision_ledger_rows": int(len(decisions)),
        "outputs": {"workbook": str(xlsx), "markdown": str(md)},
        "overall": tables.get("Calibration Overall", pd.DataFrame()).to_dict("records"),
        "economic_summary": economic.to_dict("records"),
        "gate_attribution": gate.to_dict("records"),
        "decision_gate_attribution": decision_attr.to_dict("records"),
    }
    js.write_text(json.dumps(summary, indent=2, default=str))

    print("=" * 88)
    print(" HIRING MANAGER MARKET-MAKER AUDIT")
    print("=" * 88)
    print(f"Workbook: {xlsx}")
    print(f"Summary:  {md}")
    print(f"JSON:     {js}")
    print(f"Settled predictions: {len(pred):,}")
    print(f"Eligible observations: {len(opp):,}")
    print(f"Decision ledger rows: {len(decisions):,}")
    if len(pred):
        o = tables["Calibration Overall"].iloc[0]
        print(f"Model Brier {o['Model_Brier']:.4f} vs market {o['Market_Brier']:.4f} "
              f"(improvement {o['Brier_Improvement_vs_Market']:+.4f})")
        print(f"Model log loss {o['Model_LogLoss']:.4f} vs market {o['Market_LogLoss']:.4f} "
              f"(improvement {o['LogLoss_Improvement_vs_Market']:+.4f})")
    if not economic.empty:
        print(economic.to_string(index=False))
    print("=" * 88)


if __name__ == "__main__":
    main()
