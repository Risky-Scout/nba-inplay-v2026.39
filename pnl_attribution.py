"""Daily qp7 P&L attribution."""
import pandas as pd
def attribute(df):
    d=df.copy()
    for c in ("predictive_alpha_c","execution_residual_c","price_improvement_c","fee_c","realized_net_c"):
        if c not in d:d[c]=0.0
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0.0)
    d["realized_settlement_noise_c"]=d["realized_net_c"]-(d["predictive_alpha_c"]+d["execution_residual_c"]+d["price_improvement_c"]-d["fee_c"])
    d["decomposition_residual_c"]=d["realized_net_c"]-(d["predictive_alpha_c"]+d["execution_residual_c"]+d["price_improvement_c"]-d["fee_c"]+d["realized_settlement_noise_c"])
    return d
