"""Strict chronological fitter for residual_alpha.py."""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from datetime import datetime
import residual_alpha as RA

L2=8.0

def fit_scaler(X):
    X=np.asarray(X,float)
    mean=X.mean(axis=0); sd=X.std(axis=0)
    sd=np.where(sd<1e-9,1.0,sd)
    # preserve explicit intercept column when present
    if X.shape[1] and np.allclose(X[:,0],X[0,0]):
        mean[0]=0.0; sd[0]=1.0
    return {"mean":[float(v) for v in mean],"sd":[float(v) for v in sd]}

def apply_scaler(X,sc):
    return (np.asarray(X,float)-np.asarray(sc["mean"],float))/np.asarray(sc["sd"],float)

def evidence_weights(df,game_col="game",phase_col="phase"):
    """Equalize repeated polling within game/phase cells; normalize weights to mean one."""
    if len(df)==0:return np.array([],float)
    if game_col not in df.columns:return np.ones(len(df),float)
    keys=[game_col]+([phase_col] if phase_col in df.columns else [])
    n=df.groupby(keys,dropna=False)[game_col].transform("size").astype(float)
    w=1.0/np.maximum(n.to_numpy(),1.0)
    return w/max(w.mean(),1e-12)

def fit_beta(X,p_market,y,w=None):
    X=np.asarray(X,float); y=np.asarray(y,float); off=RA.logit(p_market)
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    def obj(b):
        z=off+X@b
        return float(np.sum(w*(np.logaddexp(0,z)-y*z))+L2*np.sum(b[1:]**2))
    return minimize(obj,np.zeros(X.shape[1]),method="L-BFGS-B").x

def fit_model(df,feature_cols,game_col="game",phase_col="phase"):
    sc=fit_scaler(df[feature_cols].values)
    X=apply_scaler(df[feature_cols].values,sc)
    w=evidence_weights(df,game_col,phase_col)
    b=fit_beta(X,df.p_market.values,df.y.values,w=w)
    return {"beta":[float(x) for x in b],"scaler":sc}

def expanding_walk_forward(df,feature_cols,date_col="date",game_col="game"):
    """Each test date is untouched: scaler, weights and beta are fitted using earlier dates only."""
    folds=[]
    for d in sorted(df[date_col].astype(str).unique())[1:]:
        tr=df[df[date_col].astype(str)<d]; te=df[df[date_col].astype(str)==d]
        if len(tr)<50 or len(te)<10:continue
        mdl=fit_model(tr,feature_cols,game_col=game_col)
        folds.append((d,te.copy(),RA.predict(mdl,te.p_market.values,te[feature_cols].values)))
    return folds

def write_shadow_model(df,feature_cols,path="challenger_residual_alpha.json"):
    mdl=fit_model(df,feature_cols)
    out={"status":"SHADOW","features":list(feature_cols),**mdl,
         "market_offset_coefficient":1.0,
         "trained_through":str(df["date"].astype(str).max()) if "date" in df.columns and len(df) else None,
         "fitted":datetime.now().isoformat(timespec="seconds")}
    json.dump(out,open(path,"w"),indent=2); return out
