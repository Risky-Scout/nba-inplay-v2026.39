"""qp7 quantity-aware execution size authority."""
from __future__ import annotations
import json
from datetime import datetime
import numpy as np
import pandas as pd

QUOTE_POLICY = "qp8"
BUCKETS = ("QTY1","QTY2_3","QTY4_6","QTY7PLUS")
PILOT_LIMIT = {"QTY1":1,"QTY2_3":2,"QTY4_6":4,"QTY7PLUS":7}
RULES = {
 "QTY1": dict(orders=1,fills=0,games=1,pct=20,strict=False),
 "QTY2_3": dict(orders=30,fills=10,games=5,pct=20,strict=False),
 "QTY4_6": dict(orders=50,fills=20,games=8,pct=10,strict=False),
 "QTY7PLUS": dict(orders=100,fills=40,games=10,pct=10,strict=True),
}
def size_bucket(qty:int)->str:
    q=max(1,int(qty))
    return "QTY1" if q==1 else "QTY2_3" if q<=3 else "QTY4_6" if q<=6 else "QTY7PLUS"
def bucket_max(b):
    return {"QTY1":1,"QTY2_3":3,"QTY4_6":6,"QTY7PLUS":None}[b]
def _boot(v,g,pct,seed=39,n=4000):
    d=pd.DataFrame({"v":pd.to_numeric(v,errors="coerce"),"g":pd.Series(g).astype(str)}).dropna()
    if d.empty:return None,None
    mean=float(d.v.mean()); groups=[x.v.to_numpy() for _,x in d.groupby("g")]
    if len(groups)<2:return mean,None
    r=np.random.default_rng(seed); bs=[]
    for _ in range(n):
        pick=r.integers(0,len(groups),len(groups)); bs.append(np.concatenate([groups[j] for j in pick]).mean())
    return mean,float(np.percentile(bs,pct))
def summarize(rows,bucket):
    if rows is None or len(rows)==0:return dict(status="RESEARCH",terminal_orders=0,fills=0,games=0)
    d=rows.copy()
    if "size_bucket" not in d:d["size_bucket"]=d["requested_qty"].map(size_bucket)
    d=d[d.size_bucket.astype(str)==bucket]
    if "mode" in d:d=d[d["mode"].astype(str).str.upper()=="LIVE"]
    if "arm" in d:d=d[d["arm"].fillna("base").astype(str)=="base"]
    term=d[d["terminal"].astype(bool)] if "terminal" in d else d
    orders=int(term.order_id.astype(str).nunique()) if "order_id" in term else len(term)
    fc="filled_qty_terminal" if "filled_qty_terminal" in term else ("filled" if "filled" in term else None)
    fills=int((pd.to_numeric(term[fc],errors="coerce").fillna(0)>0).sum()) if fc else 0
    gc="game" if "game" in term else ("game_key" if "game_key" in term else None)
    games=int(term[gc].astype(str).nunique()) if gc else 0
    rc=next((c for c in ("execution_residual_c","residual_c","m30") if c in term),None)
    mean=lb=None
    if rc and gc: mean,lb=_boot(term[rc],term[gc],RULES[bucket]["pct"])
    r=RULES[bucket]
    prod=orders>=r["orders"] and fills>=r["fills"] and games>=r["games"] and mean is not None and mean>=0 and (lb is not None) and (lb>0 if r["strict"] else lb>=0)
    return dict(status="PRODUCTION" if prod else ("PILOT" if orders else "RESEARCH"),terminal_orders=orders,fills=fills,games=games,residual_mean_c=mean,residual_lcb_c=lb)
def build_authority(rows,policy_family_id):
    return {"quote_policy":QUOTE_POLICY,"policy_family_id":str(policy_family_id),"updated":datetime.now().isoformat(timespec="seconds"),"buckets":{b:summarize(rows,b) for b in BUCKETS}}
def load_authority(path,policy_family_id):
    try:a=json.load(open(path))
    except Exception:return build_authority(pd.DataFrame(),policy_family_id)
    if a.get("quote_policy")!=QUOTE_POLICY or str(a.get("policy_family_id"))!=str(policy_family_id):return build_authority(pd.DataFrame(),policy_family_id)
    return a
def write_authority(rows,policy_family_id,path):
    a=build_authority(rows,policy_family_id); json.dump(a,open(path,"w"),indent=2); return a
def highest_authorized_qty(requested,authority,production=True):
    q=max(1,int(requested)); b=size_bucket(q); info=(authority.get("buckets") or {}).get(b,{})
    if not production:return min(q,PILOT_LIMIT[b])
    if info.get("status")=="PRODUCTION":return q
    xs=[]
    for bb in BUCKETS:
        if (authority.get("buckets") or {}).get(bb,{}).get("status")=="PRODUCTION":
            m=bucket_max(bb); xs.append(q if m is None else min(q,m))
    return max([x for x in xs if x<=q],default=0)
