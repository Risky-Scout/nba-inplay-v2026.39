"""Current substitution / return hazard helpers with score-dependent path behavior."""
from dataclasses import dataclass,field
import math, numpy as np
ENGINE_VERSION="2026.31"
def score_bucket(m):
    a=abs(float(m)); return "close" if a<=8 else "moderate" if a<=15 else "large" if a<=24 else "blowout"
def _sig(x):return 1/(1+math.exp(-max(-30,min(30,x))))
@dataclass
class RotationHazardModel:
    league_sub_base:float=-4.0
    league_return_base:float=-4.0
    team_effects:dict=field(default_factory=dict)
    role_effects:dict=field(default_factory=dict)
    player_effects:dict=field(default_factory=dict)
    player_event_counts:dict=field(default_factory=dict)
    training_through_date:str=""
    fit_id:str="default"
    def _fx(self,pid,team,role):
        z=self.team_effects.get(str(team),0)+self.role_effects.get(str(role),0)
        if self.player_event_counts.get(str(pid),0)>=20:z+=self.player_effects.get(str(pid),0)
        return z
    def sub_hazard_1s(self,*,player_id,team_id,role,starter,current_stint_minutes,completed_stints,period,seconds_remaining_in_period,score_margin,fouls,projected_remaining_minutes,normal_window=True,**_):
        z=self.league_sub_base+self._fx(player_id,team_id,role)+.22*max(0,current_stint_minutes-5)+.16*completed_stints+.20*max(0,fouls-2)+(.45 if normal_window else 0)
        if period>=4:
            b=score_bucket(score_margin); z+=({"close":-.55 if starter else .05,"moderate":-.1,"large":.35,"blowout":1.10 if starter else .2})[b]
            if seconds_remaining_in_period<=360 and b=="blowout":z+=.65
        z-=.03*max(0,projected_remaining_minutes-8); return _sig(z)
    def return_hazard_1s(self,*,player_id,team_id,role,starter,bench_minutes,period,seconds_remaining_in_period,score_margin,fouls,projected_remaining_minutes,normal_window=True,**_):
        z=self.league_return_base+self._fx(player_id,team_id,role)+.25*max(0,bench_minutes-2)+.04*projected_remaining_minutes-.22*max(0,fouls-3)+(.35 if normal_window else 0)
        if period>=4:
            b=score_bucket(score_margin); z+=({"close":.9 if starter else .15,"moderate":.35,"large":-.25,"blowout":-1.35 if starter else .15})[b]
            if seconds_remaining_in_period<=360 and b=="blowout":z-=.75
        return _sig(z)
    @staticmethod
    def within(p,sec):return float(1-(1-np.clip(p,0,1))**int(sec))

def pathwise_blowout_probability(score_margin, seconds_remaining_in_game):
    """Probability of using a deep-bench rotation path at the next boundary.
    Uses absolute margin for blowout likelihood; sign is preserved elsewhere for team perspective.
    """
    a=abs(float(score_margin)); t=float(seconds_remaining_in_game)
    if t>1440:return 0.02 if a>=25 else 0.0
    if t>720:
        return 0.18 if a>=25 else (0.05 if a>=16 else 0.0)
    if t>360:
        return 0.45 if a>=25 else (0.16 if a>=16 else (0.04 if a>=9 else 0.0))
    return 0.88 if a>=25 else (0.42 if a>=16 else (0.10 if a>=9 else 0.0))
