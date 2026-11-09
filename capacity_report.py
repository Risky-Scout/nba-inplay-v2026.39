"""Diagnostic quantity-aware opportunity capacity report."""
import numpy as np, pandas as pd
BANKROLLS=(100,250,500,1000,2500,5000,10000)
def expected_alpha_usd(expected_filled_qty,conditional_edge_c):
    return float(expected_filled_qty)*float(conditional_edge_c)/100.0
def make_capacity(candidates,fill_fraction_fn):
    rows=[]
    for br in BANKROLLS:
        total=0.0
        for c in candidates:
            q=max(0,int(c.get("requested_qty",0)))
            frac=float(np.clip(fill_fraction_fn(c,q,br),0,1))
            total+=expected_alpha_usd(q*frac,c.get("conditional_edge_c",0))
        rows.append({"bankroll":br,"expected_alpha_usd":total})
    return pd.DataFrame(rows)
