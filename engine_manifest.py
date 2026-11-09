"""Prediction-engine champion/challenger manifest helpers."""
import json
from datetime import datetime
def write_manifest(path,engine_version,feature_family,status="SHADOW",metrics=None,parent=None):
    d={"engine_version":engine_version,"feature_family":feature_family,"status":status,"metrics":metrics or {},"parent":parent,"updated":datetime.now().isoformat(timespec="seconds")}
    json.dump(d,open(path,"w"),indent=2); return d
def promotable(m):
    x=m.get("metrics",{})
    return bool(x.get("games",0)>=10 and x.get("dates",0)>=3 and x.get("brier_delta",1)<=0 and x.get("logloss_delta",1)<=0 and x.get("directional_edge_improvement",0)>0 and x.get("distribution_score_delta",1)<=0)
