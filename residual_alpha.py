"""Continuous predictive residual-alpha layer; Kalshi is a fixed coefficient-1 offset."""
import numpy as np
EPS=1e-4
FEATURES=("const","model_market_logit_gap","current_stat","projected_remaining_stat","projected_remaining_minutes","current_stint","starter","on_court","completed_stints","foul_count","score_margin","pace_surprise","usage_surprise","shot_mix_surprise","assist_role_surprise","rebound_role_surprise","fta_role_surprise","market_price","spread")
def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-30,30)))
def predict(model,p_market,X):
    X=np.asarray(X,float)
    sc=model.get("scaler")
    if sc:
        X=(X-np.asarray(sc["mean"],float))/np.asarray(sc["sd"],float)
    return sigmoid(logit(p_market)+X@np.asarray(model["beta"],float))
def can_promote(m):
    return bool(m.get("games",0)>=20 and m.get("dates",0)>=5 and m.get("logloss_improvement",0)>0 and m.get("brier_improvement",0)>0 and m.get("directional_edge_lcb90",-1)>=0 and .8<=m.get("calibration_slope",0)<=1.2 and abs(m.get("calibration_intercept",99))<=.10)
