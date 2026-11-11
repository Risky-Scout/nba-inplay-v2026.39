"""Informational block-bootstrap drawdown report; never loosens risk."""
import numpy as np
def _mdd(x):
    """Maximum drawdown from starting equity (0 cumulative P&L), not only from an observed post-start peak."""
    x=np.asarray(x,float)
    if not len(x): return 0.0
    path=np.r_[0.0,x]
    peak=np.maximum.accumulate(path)
    return float(np.max(peak-path))
def risk_report(trades,bankroll,n_boot=5000,seed=39):
    if trades is None or len(trades)==0:return {}
    daily=trades.groupby("day")["pnl_usd"].sum()
    v=daily.to_numpy(float); n=max(30,len(v)); r=np.random.default_rng(seed)
    paths=np.array([r.choice(v,n,replace=True) for _ in range(n_boot)])
    cum=np.cumsum(paths,axis=1); m=np.array([_mdd(x) for x in cum]); final=cum[:,min(29,n-1)]
    return {"expected_daily_pnl":float(daily.mean()),"daily_pnl_sd":float(daily.std(ddof=1) if len(daily)>1 else 0),
            "prob_10pct_drawdown":float(np.mean(m>=.10*bankroll)),"prob_20pct_drawdown":float(np.mean(m>=.20*bankroll)),
            "expected_max_drawdown":float(m.mean()),"p05_30day_pnl":float(np.percentile(final,5))}
