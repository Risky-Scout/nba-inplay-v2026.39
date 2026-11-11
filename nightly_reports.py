#!/usr/bin/env python3
"""Nightly qp7 diagnostics: opportunity capacity, P&L attribution, and clustered drawdown risk.

None of these outputs can loosen a trading gate or risk limit.
"""
from __future__ import annotations
import glob, json, os
from datetime import datetime
import numpy as np, pandas as pd

BANKROLLS=(100,250,500,1000,2500,5000,10000)

def _read(here,pat):
    fs=[f for f in sorted(glob.glob(os.path.join(here,pat))) if os.path.getsize(f)>0]
    return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True) if fs else pd.DataFrame()

def _latest_run_config(here):
    fs=sorted(glob.glob(os.path.join(here,"run_config_*.json")))
    if not fs:return {}
    try:return json.load(open(fs[-1]))
    except Exception:return {}

def _num(d,c,default=0.0):
    if c not in d:return pd.Series(default,index=d.index,dtype=float)
    return pd.to_numeric(d[c],errors="coerce").fillna(default)

def capacity_report(here):
    """Observed-support capacity diagnostic.

    This deliberately does NOT scale quantities with hypothetical bankroll. Capacity above observed action/size
    support requires a chronological replay with simultaneous orders, queue state, reserved cash, partial fills
    and the exact execution model. Until that exists, extrapolation is marked unsupported rather than guessed.
    """
    sources={
        "eligible":_read(here,"kalshi_eligible_*.csv"),
        "posted":_read(here,"kalshi_quotes_*.csv"),
        "filled":_read(here,"kalshi_fills_*.csv"),
    }
    rows=[]
    for stage,d in sources.items():
        if d.empty:
            rows.append(dict(stage=stage,observations=0,requested_contracts=0.0,
                             expected_filled_contracts=0.0,expected_alpha_usd=0.0,
                             max_observed_qty=0.0,bankroll_extrapolation_supported=False))
            continue
        if stage=="filled" and "mode" in d.columns:
            d=d[d["mode"].astype(str).str.upper()=="LIVE"].copy()
        edge=_num(d,"cond_edge_c",np.nan)
        if edge.isna().all():edge=_num(d,"net_edge_c",0.0)
        edge=edge.fillna(0.0)
        qty=_num(d,"requested_qty",np.nan)
        if qty.isna().all():qty=_num(d,"count",np.nan)
        if qty.isna().all():qty=_num(d,"contracts",1.0).abs()
        qty=qty.fillna(1.0).clip(lower=0.0)
        frac=_num(d,"effective_fill_fraction",np.nan)
        if frac.isna().all():frac=_num(d,"fill_prob",np.nan)
        if stage=="filled":
            actual=_num(d,"contracts",np.nan).abs()
            frac=np.where(qty.to_numpy()>0,np.minimum(1.0,actual.to_numpy()/np.maximum(qty.to_numpy(),1e-12)),0.0)
            frac=pd.Series(frac,index=d.index)
        else:
            frac=frac.fillna(1.0).clip(0,1)
        expq=qty*frac
        rows.append(dict(
            stage=stage,
            observations=int(len(d)),
            requested_contracts=float(qty.sum()),
            expected_filled_contracts=float(expq.sum()),
            expected_alpha_usd=float(np.sum(expq*edge/100.0)),
            max_observed_qty=float(qty.max()) if len(qty) else 0.0,
            bankroll_extrapolation_supported=False,
            note="Observed action/size support only; no linear bankroll scaling."
        ))
    out=pd.DataFrame(rows)
    date=datetime.now().strftime("%Y-%m-%d")
    path=os.path.join(here,f"capacity_{date}.csv");out.to_csv(path,index=False)
    return path,out
def settled_fills(here,settlements,mode="LIVE",entries_only=True):
    """Settled fill diagnostic. Actual-performance callers default to LIVE; PAPER is never silently mixed in."""
    f=_read(here,"kalshi_fills_*.csv")
    if f.empty:return f
    if mode is not None:
        if "mode" not in f.columns:return f.iloc[0:0].copy()
        f=f[f["mode"].astype(str).str.upper()==str(mode).upper()].copy()
    if entries_only:
        f=f[_num(f,"contracts",0)>0].copy()
    f["result"]=f["ticker"].astype(str).map(settlements)
    f=f[f.result.isin(["yes","no"])].copy()
    if f.empty:return f
    win=np.where(f["side"].astype(str)=="yes",(f.result=="yes").astype(float),(f.result=="no").astype(float))
    fee_pc=100.0*_num(f,"fee_usd",0.0)/np.maximum(_num(f,"contracts",1.0),1e-9)
    f["realized_net_c"]=win*100.0-_num(f,"price_cents",0.0)-fee_pc
    f["fee_c"]=fee_pc
    if "trade_date_ny" in f.columns:
        f["day"]=f["trade_date_ny"].astype(str)
    else:
        # Legacy rows: preserve their logged calendar date; new rows carry explicit America/New_York trade_date_ny.
        f["day"]=f.get("timestamp",pd.Series("",index=f.index)).astype(str).str[:10]
    f["game_tok"]=f["ticker"].astype(str).str.split("-").str[1]
    return f

def account_ledger_report(here,settlements):
    """Canonical trading-only LIVE ledger from signed fills.

    Buys are negative cash flow; exits are positive proceeds less actual logged fees; remaining inventory is
    settled from the exchange result. Transfers are intentionally absent because the fill log does not contain
    deposits/withdrawals; this report therefore never claims to be a full brokerage statement.
    """
    f=_read(here,"kalshi_fills_*.csv")
    if f.empty:return None,{}
    if "mode" not in f.columns:
        return None,{"status":"NO_MODE_COLUMN","note":"Cannot classify LIVE vs PAPER safely."}
    f=f[f["mode"].astype(str).str.upper()=="LIVE"].copy()
    if f.empty:return None,{"status":"NO_LIVE_FILLS"}
    f["_ts"]=pd.to_datetime(f.get("timestamp",pd.Series("",index=f.index)),errors="coerce")
    f=f.sort_values("_ts",kind="mergesort")
    f["contracts"]=_num(f,"contracts",0.0)
    f["price_cents"]=_num(f,"price_cents",0.0)
    f["fee_usd"]=_num(f,"fee_usd",0.0)
    f["signed_cash_flow_usd"]=-(f["contracts"]*f["price_cents"]/100.0+f["fee_usd"])

    positions={}
    realized_exit=0.0
    for r in f.itertuples():
        key=(str(r.ticker),str(r.side))
        p=positions.setdefault(key,{"contracts":0.0,"basis":0.0})
        q=float(r.contracts); px=float(r.price_cents)/100.0; fee=float(r.fee_usd)
        if q>0:
            p["contracts"]+=q; p["basis"]+=q*px+fee
        elif q<0:
            sold=min(-q,p["contracts"])
            avg=p["basis"]/p["contracts"] if p["contracts"]>1e-12 else 0.0
            removed=avg*sold
            realized_exit+=sold*px-fee-removed
            p["contracts"]-=sold; p["basis"]=max(0.0,p["basis"]-removed)

    settlement_cash=0.0; unsettled_notional=0.0; open_basis=0.0
    for (ticker,side),p in positions.items():
        if p["contracts"]<=1e-12:continue
        open_basis+=p["basis"]
        result=settlements.get(ticker)
        if result in ("yes","no"):
            win=(result==side)
            settlement_cash+=p["contracts"]*(1.0 if win else 0.0)
        else:
            unsettled_notional+=p["contracts"]
    trading_cash=float(f["signed_cash_flow_usd"].sum())
    settled_equity=trading_cash+settlement_cash
    out={
        "status":"OK",
        "live_fill_rows":int(len(f)),
        "signed_fill_cash_flow_usd":trading_cash,
        "realized_exit_pnl_usd":float(realized_exit),
        "open_cost_basis_usd":float(open_basis),
        "settlement_cash_usd":float(settlement_cash),
        "trading_equity_after_settlement_usd":float(settled_equity),
        "unsettled_open_contracts":float(unsettled_notional),
        "scope":"LIVE trading fills only; deposits/withdrawals/rebates outside fill rows not included",
    }
    date=datetime.now().strftime("%Y-%m-%d")
    path=os.path.join(here,f"account_ledger_{date}.json")
    json.dump(out,open(path,"w"),indent=2)
    return path,out


def pnl_attribution_report(here,settlements):
    f=settled_fills(here,settlements,mode="LIVE",entries_only=True)
    if f.empty:return None,pd.DataFrame()
    m=_read(here,"kalshi_markouts_*.csv")
    if not m.empty and "order_id" in m and "m30" in m:
        mm=m.assign(_oid=m.order_id.astype(str),_m30=pd.to_numeric(m.m30,errors="coerce")).groupby("_oid")["_m30"].mean()
        f["execution_residual_c"]=f["order_id"].astype(str).map(mm).fillna(0.0)
    else:f["execution_residual_c"]=0.0
    pused=_num(f,"p_used",0.0); pmkt=_num(f,"p_mkt",pused)
    f["predictive_alpha_c"]=100.0*(pused-pmkt)
    f["price_improvement_c"]=100.0*pmkt-_num(f,"price_cents",0.0)
    f["realized_settlement_noise_c"]=f["realized_net_c"]-(f["predictive_alpha_c"]+f["execution_residual_c"]+f["price_improvement_c"]-f["fee_c"])
    f["decomposition_residual_c"]=f["realized_net_c"]-(f["predictive_alpha_c"]+f["execution_residual_c"]+f["price_improvement_c"]-f["fee_c"]+f["realized_settlement_noise_c"])
    if "size_bucket" not in f:
        q=_num(f,"requested_qty",np.nan).fillna(_num(f,"contracts",1.0))
        f["size_bucket"]=np.select([q==1,q<=3,q<=6],["QTY1","QTY2_3","QTY4_6"],"QTY7PLUS")
    for c in ("phase","cell"):
        if c not in f:f[c]="unknown"
    grp=["day","stat","phase","cell","size_bucket","side"]
    agg=f.groupby(grp,dropna=False).apply(lambda x:pd.Series({
        "contracts":float(_num(x,"contracts",1).sum()),
        "realized_net_c":float(np.average(x.realized_net_c,weights=np.maximum(_num(x,"contracts",1),1e-9))),
        "predictive_alpha_c":float(np.average(x.predictive_alpha_c,weights=np.maximum(_num(x,"contracts",1),1e-9))),
        "execution_residual_c":float(np.average(x.execution_residual_c,weights=np.maximum(_num(x,"contracts",1),1e-9))),
        "price_improvement_c":float(np.average(x.price_improvement_c,weights=np.maximum(_num(x,"contracts",1),1e-9))),
        "fee_c":float(np.average(x.fee_c,weights=np.maximum(_num(x,"contracts",1),1e-9))),
        "settlement_noise_c":float(np.average(x.realized_settlement_noise_c,weights=np.maximum(_num(x,"contracts",1),1e-9))),
        "decomposition_residual_c":float(np.average(x.decomposition_residual_c,weights=np.maximum(_num(x,"contracts",1),1e-9))),
    })).reset_index()
    date=datetime.now().strftime("%Y-%m-%d")
    path=os.path.join(here,f"pnl_attribution_{date}.csv"); agg.to_csv(path,index=False)
    if (f.decomposition_residual_c.abs()>0.5).mean()>0.05:
        print("WARNING: P&L attribution decomposition residual exceeds 0.5c/contract materially")
    return path,agg

def risk_report(here,settlements,bankroll):
    f=settled_fills(here,settlements,mode="LIVE",entries_only=True)
    if f.empty:return None,{}
    f["pnl_usd"]=f["realized_net_c"]*_num(f,"contracts",1.0)/100.0
    blocks=f.groupby(["day","game_tok"],dropna=False)["pnl_usd"].sum().reset_index()
    days={d:g.pnl_usd.to_numpy(float) for d,g in blocks.groupby("day")}
    day_keys=list(days)
    if not day_keys:return None,{}
    rng=np.random.default_rng(39); n_days=max(30,len(day_keys)); n_boot=5000
    finals=[]; mdds=[]; underwater=[]
    for _ in range(n_boot):
        path=[]; cum=0.0; high=0.0; cur_under=0; spans=[]
        for __ in range(n_days):
            dk=day_keys[int(rng.integers(0,len(day_keys)))]
            games=days[dk]
            # GAME block inside sampled DAY: all trades inside a game remain together.
            gp=games[rng.integers(0,len(games),len(games))].sum() if len(games) else 0.0
            cum+=float(gp); high=max(high,cum); path.append(cum)
            if cum<high:cur_under+=1
            elif cur_under:spans.append(cur_under); cur_under=0
        if cur_under:spans.append(cur_under)
        arr=np.asarray(path); peak=np.maximum.accumulate(np.r_[0.0,arr]) [1:]
        mdds.append(float(np.max(peak-arr)) if len(arr) else 0.0)
        finals.append(float(arr[min(29,len(arr)-1)]))
        underwater.append(float(np.median(spans)) if spans else 0.0)
    daily=blocks.groupby("day").pnl_usd.sum()
    mdds=np.asarray(mdds)
    out={"expected_daily_pnl":float(daily.mean()),"daily_pnl_sd":float(daily.std(ddof=1) if len(daily)>1 else 0.0),
         "probability_10pct_drawdown":float(np.mean(mdds>=.10*bankroll)),
         "probability_20pct_drawdown":float(np.mean(mdds>=.20*bankroll)),
         "expected_max_drawdown":float(mdds.mean()),"median_time_underwater_days":float(np.median(underwater)),
         "p05_30day_pnl":float(np.percentile(finals,5)),"bootstrap":"DAY outer / GAME inner; trades within game preserved"}
    date=datetime.now().strftime("%Y-%m-%d"); path=os.path.join(here,f"risk_report_{date}.json")
    json.dump(out,open(path,"w"),indent=2); return path,out

def write_all(here,settlements):
    cfg=_latest_run_config(here); br=float(cfg.get("bankroll",250.0) or 250.0)
    cp,_=capacity_report(here); ap,_=account_ledger_report(here,settlements)
    pp,_=pnl_attribution_report(here,settlements); rp,_=risk_report(here,settlements,br)
    return {"capacity":cp,"account_ledger":ap,"pnl_attribution":pp,"risk_report":rp}
