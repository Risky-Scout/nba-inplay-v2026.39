"""Central foul / bonus / free-throw / endgame state helpers."""
from dataclasses import dataclass
@dataclass
class FoulState:
    period:int
    home_team_fouls:int=0
    away_team_fouls:int=0
    possession_after_technical:bool=False
    ft_remaining:int=0
    ft_shooter_id:int=0
    final_ft_can_rebound:bool=False
def in_bonus(team_fouls,period,seconds_remaining_in_period,last_two_min_fouls=None):
    """NBA team-penalty state.

    team_fouls and last_two_min_fouls are counts BEFORE the next common foul.
    Regulation: penalty after four team fouls (5th+ foul). Overtime: after three (4th+).
    In the final 2:00, if the period threshold has not been reached, the second team foul
    committed in the final two minutes puts the opponent in the penalty. We therefore
    require the explicit last-two-minute counter; total period fouls cannot substitute for it.
    """
    tf=max(0,int(team_fouls or 0)); per=max(1,int(period or 1)); sec=max(0.0,float(seconds_remaining_in_period or 0.0))
    threshold=4 if per<=4 else 3
    if tf>=threshold+1:
        return True
    if sec<=120.0 and last_two_min_fouls is not None and int(last_two_min_fouls)>=2:
        return True
    return False
def intentional_foul_probability(score_margin_from_defense,clock_s,defense_has_possession=False):
    if defense_has_possession or clock_s>90 or score_margin_from_defense>=0:return 0.0
    t=abs(score_margin_from_defense)
    if t<=3 and clock_s<=35:return .78
    if t<=6 and clock_s<=55:return .62
    if t<=9 and clock_s<=75:return .35
    return .08
