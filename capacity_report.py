"""Observed-support capacity diagnostics.

This module intentionally does NOT extrapolate quantities by hypothetical bankroll.
A larger-bankroll capacity claim requires chronological replay with exact action support,
queue state, simultaneous orders, reserved cash, partial fills, fees, and position limits.
"""
import numpy as np
import pandas as pd


def expected_alpha_usd(expected_filled_qty,conditional_edge_c):
    return float(expected_filled_qty)*float(conditional_edge_c)/100.0


def make_capacity(candidates,fill_fraction_fn):
    """Summarize only the logged/observed requested quantities.

    fill_fraction_fn(candidate, requested_qty, None) may estimate the fraction expected to fill
    at the OBSERVED requested quantity. No bankroll multiplier is supplied or inferred.
    """
    rows=[]
    total=0.0
    requested=0.0
    expected_filled=0.0
    max_qty=0.0
    for c in candidates:
        q=max(0,int(c.get("requested_qty",0)))
        frac=float(np.clip(fill_fraction_fn(c,q,None),0,1))
        ef=q*frac
        total+=expected_alpha_usd(ef,c.get("conditional_edge_c",0))
        requested+=q
        expected_filled+=ef
        max_qty=max(max_qty,q)
    rows.append({
        "observations":len(candidates),
        "requested_contracts":requested,
        "expected_filled_contracts":expected_filled,
        "expected_alpha_usd":total,
        "max_observed_qty":max_qty,
        "bankroll_extrapolation_supported":False,
        "note":"Observed action/size support only; larger-bankroll capacity requires chronological replay."
    })
    return pd.DataFrame(rows)
