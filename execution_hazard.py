"""qp8 discrete-time fill-hazard helpers and exact action identity."""
import numpy as np
MIN_TERMINAL,MIN_FILLS,MIN_GAMES=500,75,10
def cumulative_fill_probability(hazards):
    s=1.0
    for h in hazards:s*=1-float(np.clip(h,0,1))
    return float(np.clip(1-s,0,1))
def validated(m):
    return bool(m and m.get("terminal_quotes",0)>=MIN_TERMINAL and m.get("fills",0)>=MIN_FILLS and m.get("games",0)>=MIN_GAMES and m.get("walk_forward_logloss_beats_static") is True and m.get("policy_value_lcb_beats_static") is True)
def ttl_bucket(ttl):
    t=float(ttl)
    return "<=5" if t<=5 else "6-10" if t<=10 else "11-20" if t<=20 else ">20"
def action_key(price,requested_qty,ttl):return f"{int(price)}|{int(requested_qty)}|{ttl_bucket(ttl)}"
def continuation_value(expected_future_filled_qty,conditional_edge_c,queue_reset_penalty_c=0.0,explicit_cost_c=0.0):
    return max(0.0,float(expected_future_filled_qty))*float(conditional_edge_c)-max(0.0,float(queue_reset_penalty_c))-max(0.0,float(explicit_cost_c))
