"""Nested chronological selection scaffold for the current probability-engine live-information priors."""
from datetime import datetime
import numpy as np
from nba_markov_engine import ENGINE_VERSION
CANDIDATES={"SHOT_MIX_KAPPA":[6,10,15,20,30,45],"ASSIST_ALPHA":[6,10,15,20,30,45],
"PACE_TAU0":[20,30,45,60,90,120],"MINUTES_EVIDENCE_K":[5,7.5,10,15,20,30],
"FTA_ALPHA":[8,12,20,30,45],"TOV_ALPHA":[8,12,20,30,45],"ORB_ALPHA":[8,12,20,30,45],"DRB_ALPHA":[8,12,20,30,45]}
def nested_select(score_fn,dates):
    chosen={}
    for name,vals in CANDIDATES.items():
        evals=[]
        for i in range(2,len(dates)):
            inner=dates[:i-1]; validation=dates[i-1]; test=dates[i]
            ranked=sorted((sum(score_fn(name,v,inner,validation)),v) for v in vals)
            v=ranked[0][1]; evals.append((sum(score_fn(name,v,dates[:i],test)),v))
        if evals:
            chosen[name]=min(vals,key=lambda v:np.mean([s for s,vv in evals if vv==v]) if any(vv==v for _,vv in evals) else np.inf)
    return {"engine_version":ENGINE_VERSION,"validated":bool(chosen),"params":chosen,"fitted":datetime.now().isoformat(timespec="seconds")}
