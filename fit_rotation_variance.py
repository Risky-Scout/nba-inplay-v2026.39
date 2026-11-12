"""Fit phase-specific remaining-minutes uncertainty with hierarchical shrinkage."""
import numpy as np, pandas as pd, json
from datetime import datetime
from nba_markov_engine import ENGINE_VERSION
SHRINK_K=25.0
def score_bucket(m):
    a=abs(float(m)); return "close" if a<=8 else "moderate" if a<=15 else "large" if a<=24 else "blowout"
def foul_bucket(f):return "0-2" if int(f)<=2 else str(min(5,int(f)))
def fit_frame(df):
    d=df.copy(); d["resid"]=pd.to_numeric(d.actual_remaining_minutes)-pd.to_numeric(d.predicted_remaining_minutes)
    d["fb"]=d.fouls.map(foul_bucket); d["sb"]=d.score_margin.map(score_bucket)
    d["role"]=np.where(d.starter.astype(bool),"starter","bench"); d["court"]=np.where(d.on_court.astype(bool),"on","off")
    parent=d.groupby("phase").resid.std().to_dict(); out={}
    for key,g in d.groupby(["phase","role","court","fb","sb"]):
        ps=float(parent.get(key[0],3.0)); sd=float(g.resid.std(ddof=1)) if len(g)>1 else ps
        if not np.isfinite(sd):sd=ps
        lam=len(g)/(len(g)+SHRINK_K); sh=lam*sd+(1-lam)*ps; base=max(1.0,float(pd.to_numeric(g.predicted_remaining_minutes).mean()))
        out["|".join(key)]={"n":len(g),"minutes_cv":float(np.clip(sh/base,.05,.60)),"rotation_cv":float(np.clip(.65*sh/base,.03,.40))}
    return {"engine_version":ENGINE_VERSION,"fitted":datetime.now().isoformat(timespec="seconds"),"fallback":{"minutes_cv":.18,"rotation_cv":.10},"buckets":out}
