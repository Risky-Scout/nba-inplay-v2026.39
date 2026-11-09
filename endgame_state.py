"""Central foul/bonus/FT/endgame state helpers for engine 2026.29."""
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
def in_bonus(team_fouls,period,seconds_remaining_in_period):
    return int(team_fouls)>=5 or (float(seconds_remaining_in_period)<=120 and int(team_fouls)>=2)
def intentional_foul_probability(score_margin_from_defense,clock_s,defense_has_possession=False):
    if defense_has_possession or clock_s>90 or score_margin_from_defense>=0:return 0.0
    t=abs(score_margin_from_defense)
    if t<=3 and clock_s<=35:return .78
    if t<=6 and clock_s<=55:return .62
    if t<=9 and clock_s<=75:return .35
    return .08
