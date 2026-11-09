"""Engine champion/challenger, drift and recency governance."""
def can_promote_engine(m):
    return bool(m.get("games",0)>=10 and m.get("dates",0)>=3 and m.get("brier_delta",1)<=0 and m.get("logloss_delta",1)<=0 and m.get("directional_edge_improvement",0)>0 and m.get("distribution_score_delta",1)<=0)
def drift_tier(tier,recent10_edge,recent_brier_diff,exec_lcb,adverse_change,recent20_lb=None,consecutive_fail_windows=0):
    if tier=="PRODUCTION" and (recent10_edge<0 or recent_brier_diff>.01 or exec_lcb<0 or adverse_change):return "PILOT"
    if tier=="PILOT" and ((recent20_lb is not None and recent20_lb<0) or consecutive_fail_windows>=2):return "RESEARCH"
    return tier
def recency_weight(age_days,half_life_days):
    if half_life_days is None or half_life_days==float("inf"):return 1.0
    return float(2**(-max(0,float(age_days))/float(half_life_days)))
