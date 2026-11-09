"""Strict chronological fitter for residual_alpha.py."""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from datetime import datetime
import residual_alpha as RA
L2=8.0
def fit_beta(X,p_market,y,w=None):
    X=np.asarray(X,float); y=np.asarray(y,float); off=RA.logit(p_market); w=np.ones(len(y)) if w is None else np.asarray(w,float)
    def obj(b):
        z=off+X@b
        return float(np.sum(w*(np.logaddexp(0,z)-y*z))+L2*np.sum(b[1:]**2))
    return minimize(obj,np.zeros(X.shape[1]),method="L-BFGS-B").x
def expanding_walk_forward(df,feature_cols,date_col="date",game_col="game"):
    folds=[]
    for d in sorted(df[date_col].astype(str).unique())[1:]:
        tr=df[df[date_col].astype(str)<d]; te=df[df[date_col].astype(str)==d]
        if len(tr)<50 or len(te)<10:continue
        b=fit_beta(tr[feature_cols].values,tr.p_market.values,tr.y.values)
        folds.append((d,te.copy(),RA.predict({"beta":b},te.p_market.values,te[feature_cols].values)))
    return folds
def write_shadow_model(df,feature_cols,path="challenger_residual_alpha.json"):
    b=fit_beta(df[feature_cols].values,df.p_market.values,df.y.values)
    out={"status":"SHADOW","features":list(feature_cols),"beta":[float(x) for x in b],"market_offset_coefficient":1.0,"fitted":datetime.now().isoformat(timespec="seconds")}
    json.dump(out,open(path,"w"),indent=2); return out
