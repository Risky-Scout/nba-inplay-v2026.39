#!/usr/bin/env python3
"""Fit qp7 discrete-time first-fill hazard from second-by-second queue state.

The resulting model is SHADOW unless all activation thresholds are satisfied.
No LIVE code may use a hazard model merely because this file exists.
"""
from __future__ import annotations
import glob, json, os
from datetime import datetime
import numpy as np, pandas as pd
from scipy.optimize import minimize
from execution_hazard import validated

HERE=os.path.dirname(os.path.abspath(__file__))
CONT=("remaining_ttl","queue_ahead_est","queue_depletion_rate","spread","aggr","imbalance",
      "mid_velocity","best_depth","t_since_action_s","requested_qty","qty_over_best_depth","qty_over_queue_ahead")
STOPS=("timeout","substitution","freethrow","period","instantreplay")
FEATURES=("const",)+CONT+tuple("stop_"+x for x in STOPS)
L2=1.0

def _read(pat):
    xs=[pd.read_csv(p) for p in sorted(glob.glob(os.path.join(HERE,pat))) if os.path.getsize(p)>0]
    return pd.concat(xs,ignore_index=True) if xs else pd.DataFrame()

def intervals(history:pd.DataFrame,events:pd.DataFrame)->pd.DataFrame:
    if history.empty:return pd.DataFrame()
    h=history.copy()
    h["time"]=pd.to_datetime(h["time"],errors="coerce",utc=True)
    e=events.copy()
    if len(e):
        e["time"]=pd.to_datetime(e["time"],errors="coerce",utc=True)
        e["order_id"]=e["order_id"].astype(str)
    out=[]
    for oid,g in h.dropna(subset=["time"]).groupby(h["order_id"].astype(str),sort=False):
        g=g.sort_values("time").copy()
        ev=e[e["order_id"]==str(oid)].sort_values("time") if len(e) else e
        fill=ev[ev["event"].astype(str)=="fill"] if len(ev) else ev
        term=ev[ev["event"].astype(str)=="terminal"] if len(ev) else ev
        fill_ts=fill["time"].iloc[0] if len(fill) else None
        terminal_ts=term["time"].iloc[-1] if len(term) else None
        end=fill_ts if fill_ts is not None else (terminal_ts if terminal_ts is not None else g["time"].max())
        start=g["time"].min()
        if pd.isna(start) or pd.isna(end) or end<start:continue
        grid=pd.DataFrame({"time":pd.date_range(start.floor("s"),end.ceil("s"),freq="1s")})
        base=pd.merge_asof(grid,g.drop(columns=["order_id"],errors="ignore").sort_values("time"),on="time",direction="backward")
        base["order_id"]=oid
        base["hazard"]=0.0
        if fill_ts is not None:
            ix=(base["time"]<=fill_ts)&(base["time"]+pd.Timedelta(seconds=1)>fill_ts)
            if ix.any():base.loc[ix,"hazard"]=1.0
        base["censored"]=bool(fill_ts is None and terminal_ts is None)
        out.append(base)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def design(d):
    rows=[]
    for r in d.itertuples():
        st=str(getattr(r,"stop","") or "").lower()
        vals=[1.0]+[float(getattr(r,k,0.0) or 0.0) for k in CONT]+[float(k in st) for k in STOPS]
        rows.append(vals)
    return np.asarray(rows,float)

def fit_scaler(X):
    mean=X.mean(0); sd=X.std(0)
    mean[0]=0; sd[0]=1; sd[sd<1e-9]=1
    return {"mean":mean.tolist(),"sd":sd.tolist()}
def std(sc,X):return (X-np.asarray(sc["mean"]))/np.asarray(sc["sd"])
def fit_logit(X,y):
    def obj(b):
        z=np.clip(X@b,-30,30)
        return float(np.sum(np.logaddexp(0,z)-y*z)+L2*np.sum(b[1:]**2))
    return minimize(obj,np.zeros(X.shape[1]),method="L-BFGS-B").x
def pred(b,X):return 1/(1+np.exp(-np.clip(X@b,-30,30)))
def logloss(p,y):
    p=np.clip(p,1e-6,1-1e-6); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def main():
    h,e=_read("kalshi_queue_history_*.csv"),_read("kalshi_quote_events_*.csv")
    d=intervals(h,e)
    out={"quote_policy":"qp8","features":FEATURES,"status":"SHADOW","validated":False,
         "fitted":datetime.now().isoformat(timespec="seconds")}
    if d.empty:
        json.dump(out,open(os.path.join(HERE,"challenger_fill_hazard.json"),"w"),indent=2); return out
    d=d.dropna(subset=["remaining_ttl"]).copy()
    d["date"]=d.get("run_id",pd.Series("",index=d.index)).astype(str).str[:8]
    X=design(d); sc=fit_scaler(X); Xs=std(sc,X); y=d["hazard"].to_numpy(float)
    b=fit_logit(Xs,y)
    terminal=e[e["event"].astype(str)=="terminal"]["order_id"].astype(str).nunique() if len(e) else 0
    fills=e[e["event"].astype(str)=="fill"]["order_id"].astype(str).nunique() if len(e) else 0
    games=h["ticker"].astype(str).str.split("-").str[1].nunique() if "ticker" in h else 0
    # Strict chronological one-date-ahead validation.
    folds=[]; dates=sorted(x for x in d["date"].unique() if x)
    for k in range(1,len(dates)):
        tr=d[d.date.isin(dates[:k])]; te=d[d.date==dates[k]]
        if len(tr)<100 or te.hazard.sum()<1:continue
        Xt=design(tr); sct=fit_scaler(Xt); bt=fit_logit(std(sct,Xt),tr.hazard.to_numpy(float))
        pp=pred(bt,std(sct,design(te))); folds.append(logloss(pp,te.hazard.to_numpy(float)))
    meta={"terminal_quotes":int(terminal),"fills":int(fills),"games":int(games),
          "walk_forward_logloss":None if not folds else float(np.mean(folds)),
          # Must be supplied by held-out static-model comparison before activation.
          "walk_forward_logloss_beats_static":False,"policy_value_lcb_beats_static":False}
    out.update(scaler=sc,coef=[float(x) for x in b],metadata=meta)
    out["validated"]=validated(meta)
    json.dump(out,open(os.path.join(HERE,"challenger_fill_hazard.json"),"w"),indent=2)
    print(json.dumps({"rows":len(d),**meta,"validated":out["validated"]},indent=2))
    return out
if __name__=="__main__":main()
