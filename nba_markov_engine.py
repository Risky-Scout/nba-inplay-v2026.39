#!/usr/bin/env python3
"""
NBA In-Play Markov Chain Predictive Engine (v2026.18, calibrated)
=============================================================================
- Real-time NBA CDN (cdn.nba.com) box score, substitutions, and PBP tracking.
- Absorbing Markov Chain Monte Carlo simulations conditioned on Dunks & Threes EPM.
- Minutes: D&T projected minutes -> NBAStintEngine (foul windows, rotation evidence, blowouts,
  foul-outs) -> water-filled to exactly 5 x time-left per team -> deterministic lineup schedule.
- Pace: each compiled chain's time-per-touch is solved from its fundamental matrix so simulated
  possessions match the D&T projected pace (blended with live pace).
- Calibration: a pre-game full-game sim is anchored to the D&T projected box score per player/stat;
  live increments are rescaled with the same multipliers (binomial thinning / thickening).
"""

import os
import json
import math
from statistics import NormalDist
import random
import bisect
import unicodedata
import subprocess
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional
from nba_stint_engine import NBAStintEngine
from rotation_hazard import pathwise_blowout_probability

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_secure_api_key(key_name: str) -> str:
    env_val = os.getenv(key_name)
    if env_val and env_val not in ("", "your_api_token_here"):
        return env_val
    try:
        val = subprocess.check_output(
            ["security", "find-generic-password", "-a", os.environ.get("USER", ""), "-s", key_name, "-w"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        if val:
            return val
    except Exception:
        pass
    return ""

DT_API_KEY = get_secure_api_key("DT_API_KEY")
DT_BASE_URL = "https://dunksandthrees.com/api/v2"

TODAY_STR = datetime.now().strftime("%Y-%m-%d")
NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
NBA_PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}

def normalize_player_name(name: str) -> str:
    if not name:
        return ""
    if "(" in name:
        name = name.split("(")[0]
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = ascii_str.lower().replace(".", "").replace("'", "").replace("-", " ").strip()
    tokens = [t for t in clean.split() if t not in ("jr", "sr", "ii", "iii", "iv")]
    return " ".join(tokens)

@dataclass
class DTPlayerSkillProfile:
    player_id: int
    player_name: str
    team_id: int
    position_code: int
    inches: int
    weight: float
    p_pace: float
    p_mp: float
    p_usg: float
    p_fga_rim_100: float
    p_fga_mid_100: float
    p_fg3a_100: float
    p_fta_100: float
    p_ast_100: float
    p_tov_100: float
    p_orb_100: float
    p_drb_100: float
    p_stl_100: float
    p_blk_100: float
    p_fgpct_rim: float
    p_fgpct_mid: float
    p_fg3pct: float
    p_ftpct: float
    oepm: float
    depm: float
    epm: float
    p_start: Optional[float] = None   # D&T p_pct_start as a 0-1 probability (None = not supplied)

    @property
    def role_archetype(self) -> str:
        if self.p_usg >= 0.28 and self.p_ast_100 >= 6.0:
            return "Heliocentric Creator"
        elif self.p_usg >= 0.23 and self.p_ast_100 < 6.0:
            return "Secondary Scorer"
        elif self.position_code >= 4 and (self.p_fg3a_100 >= 4.5 or self.p_ast_100 >= 5.0):
            return "Stretch/Playmaking Big"
        elif self.position_code >= 4:
            return "Rim-Runner/Anchor Big"
        else:
            return "3&D / Movement Wing"

@dataclass
class MarketOffer:
    player_id: int
    player_name: str
    stat: str
    line: float
    over_odds: int
    over_book: str
    under_odds: int
    under_book: str

@dataclass
class LivePBPContext:
    last_possession_team_id: Optional[int] = None
    last_ball_handler_id: Optional[int] = None
    assist_pairs: Dict[Tuple[int, int], int] = field(default_factory=dict)
    live_rim_fga: Dict[int, int] = field(default_factory=dict)
    live_mid_fga: Dict[int, int] = field(default_factory=dict)
    live_3pt_fga: Dict[int, int] = field(default_factory=dict)
    live_fta: Dict[int, int] = field(default_factory=dict)
    live_tov: Dict[int, int] = field(default_factory=dict)
    live_orb: Dict[int, int] = field(default_factory=dict)
    live_drb: Dict[int, int] = field(default_factory=dict)
    live_orb_opportunities: Dict[int, int] = field(default_factory=dict)  # keyed by team_id
    live_drb_opportunities: Dict[int, int] = field(default_factory=dict)  # keyed by team_id
    current_stint_minutes: Dict[int, float] = field(default_factory=dict)
    completed_stints: Dict[int, int] = field(default_factory=dict)
    last_action_type: str = ""
    last_action_number: int = 0

@dataclass
class LiveInPlayState:
    game_id: str
    matchup_title: str
    clock_display: str
    period: int
    seconds_remaining_in_game: float
    home_team_id: int
    away_team_id: int
    home_alias: str
    away_alias: str
    home_score: int
    away_score: int
    possession_team: str
    on_court_home: List[int]
    on_court_away: List[int]
    live_box: Dict[int, Dict[str, float]]
    pbp_context: LivePBPContext = field(default_factory=LivePBPContext)
    game_code: str = ""          # NBA gameCode, e.g. "20261105/ORLPHX"
    game_status: int = 1         # 1 scheduled, 2 live, 3 final

def nba_get_json(url: str, timeout: int = 10):
    try:
        from curl_cffi import requests as c_requests
        resp = c_requests.get(
            url,
            headers={"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com", "Accept": "application/json"},
            impersonate="chrome120",
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    try:
        resp = requests.get(url, headers=NBA_HEADERS, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

class NBALivePlayByPlayParser:
    @staticmethod
    def fetch_and_parse_pbp(full_nba_game_id: str, current_game_seconds_rem: float, on_court: Optional[List[int]] = None,
                            period_now: int = 1) -> LivePBPContext:
        url = NBA_PBP_URL.format(game_id=full_nba_game_id)
        data = nba_get_json(url, timeout=8)
        if not data:
            return LivePBPContext()
        actions = data.get("game", {}).get("actions", [])

        assist_pairs = {}
        live_rim_fga, live_mid_fga, live_3pt_fga = {}, {}, {}
        live_fta, live_tov, live_orb, live_drb = {}, {}, {}, {}
        miss_by_team = {}
        last_sub_in_sec, subbed, completed = {}, set(), {}
        last_poss_team_id, last_active_player_id = None, None
        last_type, last_num = "", 0

        for act in actions:
            last_type, last_num = str(act.get("actionType", "")), int(act.get("actionNumber", 0) or 0)
            act_type = act.get("actionType", "")
            act_type_l = str(act_type).lower().replace("_", "")
            sub_type = act.get("subType", "")
            sub_type_l = str(sub_type).lower()
            team_id = int(act.get("teamId", act.get("teamId", 0)) or 0)
            pid = int(act.get("personId", 0) or 0)
            period = int(act.get("period", 1) or 1)
            clock_str = act.get("clock", "PT12M00.00S") or "PT12M00.00S"

            clean = clock_str.replace("PT", "").replace("S", "")
            sec_in_p = (float(clean.split("M")[0]) * 60.0 + float(clean.split("M")[1])) if "M" in clean else float(clean or 0.0)
            act_sec_rem = sec_in_p + max(0, 4 - period) * 720.0

            if act.get("possession"):
                last_poss_team_id = int(act["possession"])
            if pid != 0:
                last_active_player_id = pid

            if act_type == "substitution" and pid != 0:
                subbed.add(pid)
                if sub_type == "in":
                    last_sub_in_sec[pid] = act_sec_rem
                elif sub_type == "out":
                    completed[pid] = completed.get(pid, 0) + 1
                    last_sub_in_sec.pop(pid, None)
            elif act_type in ("2pt", "3pt"):
                if act_type == "3pt":
                    live_3pt_fga[pid] = live_3pt_fga.get(pid, 0) + 1
                else:
                    dist = float(act.get("shotDistance", 10.0) or 10.0)
                    quals = act.get("qualifiers", []) or []
                    # v2026.28: rim = restricted area (<= 4 ft), matching D&T's rim zone. "Points in the paint" also
                    # covers 5-15 ft floaters/hooks, which D&T counts as mid-range (the field was unused before).
                    if dist <= 4.0:
                        live_rim_fga[pid] = live_rim_fga.get(pid, 0) + 1
                    else:
                        live_mid_fga[pid] = live_mid_fga.get(pid, 0) + 1

                if act.get("shotResult") == "Made" and act.get("assistPersonId"):
                    assister_id = int(act["assistPersonId"])
                    pair = (assister_id, pid)
                    assist_pairs[pair] = assist_pairs.get(pair, 0) + 1
                if act.get("shotResult") != "Made" and team_id:
                    miss_by_team[team_id] = miss_by_team.get(team_id, 0) + 1
            elif act_type_l in ("freethrow", "freethrows") and pid:
                live_fta[pid] = live_fta.get(pid, 0) + 1
            elif act_type_l == "turnover" and pid:
                live_tov[pid] = live_tov.get(pid, 0) + 1
            elif act_type_l == "rebound" and pid:
                if "offensive" in sub_type_l:
                    live_orb[pid] = live_orb.get(pid, 0) + 1
                elif "defensive" in sub_type_l:
                    live_drb[pid] = live_drb.get(pid, 0) + 1

        teams = set(miss_by_team)
        live_orb_opportunities = dict(miss_by_team)
        live_drb_opportunities = {t: sum(v for ot, v in miss_by_team.items() if ot != t) for t in teams}

        # Current stint, v2026.28: (a) opening starters never have a 'sub in' event, so a player on the floor with no
        # substitution yet has been on since tip; (b) halftime (and an OT break) resets fatigue, so a stint never starts
        # before the current half. Before, an opening starter showed no stint at all and never got the fatigue term.
        half_start = 2880.0 if period_now <= 2 else (1440.0 if period_now <= 4 else 300.0)
        current_stint_minutes = {}
        for pid, check_in_sec in last_sub_in_sec.items():
            stint_secs = max(0.0, min(check_in_sec, half_start) - current_game_seconds_rem)
            current_stint_minutes[pid] = round(stint_secs / 60.0, 2)
        for pid in (on_court or []):
            if pid not in last_sub_in_sec and pid not in subbed:
                current_stint_minutes[pid] = round(max(0.0, half_start - current_game_seconds_rem) / 60.0, 2)

        return LivePBPContext(
            last_possession_team_id=last_poss_team_id,
            last_ball_handler_id=last_active_player_id,
            assist_pairs=assist_pairs,
            live_rim_fga=live_rim_fga,
            live_mid_fga=live_mid_fga,
            live_3pt_fga=live_3pt_fga,
            live_fta=live_fta,
            live_tov=live_tov,
            live_orb=live_orb,
            live_drb=live_drb,
            live_orb_opportunities=live_orb_opportunities,
            live_drb_opportunities=live_drb_opportunities,
            current_stint_minutes=current_stint_minutes,
            completed_stints=completed,
            last_action_type=last_type,
            last_action_number=last_num
        )

    @staticmethod
    def latest_action(full_nba_game_id: str, after_number: int) -> Optional[Tuple[int, set, str]]:
        """Like new_actions, but None when the feed cannot be read. Execution must treat None as UNKNOWN state
        (new_actions returns after_number on failure, i.e. 'nothing happened', which is the wrong default for trading)."""
        data = nba_get_json(NBA_PBP_URL.format(game_id=full_nba_game_id), timeout=4)
        if not data or not isinstance(data.get("game", {}).get("actions"), list):
            return None
        pids, last_num, last_type = set(), after_number, ""
        for a in data["game"]["actions"] or []:
            num = int(a.get("actionNumber", 0) or 0)
            if num <= after_number:
                continue
            last_num, last_type = max(last_num, num), str(a.get("actionType", ""))
            for k in ("personId", "assistPersonId", "foulDrawnPersonId", "blockPersonId", "stealPersonId"):
                if a.get(k):
                    pids.add(int(a[k]))
        return last_num, pids, last_type

    @staticmethod
    def new_actions(full_nba_game_id: str, after_number: int) -> Tuple[int, set, str]:
        """(latest actionNumber, player ids in actions after `after_number`, latest action type)."""
        data = nba_get_json(NBA_PBP_URL.format(game_id=full_nba_game_id), timeout=4)
        if not data:
            return after_number, set(), ""
        pids, last_num, last_type = set(), after_number, ""
        for a in data.get("game", {}).get("actions", []) or []:
            num = int(a.get("actionNumber", 0) or 0)
            if num <= after_number:
                continue
            last_num, last_type = max(last_num, num), str(a.get("actionType", ""))
            for k in ("personId", "assistPersonId", "foulDrawnPersonId", "blockPersonId", "stealPersonId"):
                if a.get(k):
                    pids.add(int(a[k]))
        return last_num, pids, last_type

class NBALiveFeedClient:
    @staticmethod
    def get_todays_games() -> List[dict]:
        data = nba_get_json(NBA_SCOREBOARD_URL, timeout=10)
        return data.get("scoreboard", {}).get("games", []) if data else []

    @staticmethod
    def parse_clock_to_seconds(clock_str: str, period: int) -> Tuple[float, str]:
        if not clock_str:
            rem_in_period = 720.0 if period == 1 else 0.0
            disp = f"Q{period} 12:00" if period == 1 else f"Q{period} 0:00"
        else:
            clean = clock_str.replace("PT", "").replace("S", "")
            if "M" in clean:
                mins, secs = clean.split("M")
                rem_in_period = float(mins) * 60.0 + float(secs)
                disp = f"Q{period} {int(mins)}:{int(float(secs)):02d}"
            else:
                rem_in_period = float(clean)
                disp = f"Q{period} 0:{int(rem_in_period):02d}"

        future_quarters = max(0, 4 - period) if period <= 4 else 0
        return rem_in_period + (future_quarters * 720.0), disp

    @classmethod
    def fetch_live_game_state(cls, game_summary: dict) -> LiveInPlayState:
        full_nba_game_id = game_summary.get("gameId")
        home_meta = game_summary.get("homeTeam", {})
        away_meta = game_summary.get("awayTeam", {})
        home_id, away_id = int(home_meta.get("teamId", 0)), int(away_meta.get("teamId", 0))
        home_tri, away_tri = home_meta.get("teamTricode", "HOM"), away_meta.get("teamTricode", "AWY")

        url = NBA_BOXSCORE_URL.format(game_id=full_nba_game_id)
        data = nba_get_json(url, timeout=8)
        if not data:
            return LiveInPlayState(
                game_id=full_nba_game_id, matchup_title=f"{away_tri} @ {home_tri}",
                clock_display="Q1 12:00", period=1, seconds_remaining_in_game=2880.0,
                home_team_id=home_id, away_team_id=away_id, home_alias=home_tri, away_alias=away_tri,
                home_score=0, away_score=0, possession_team="HOME",
                on_court_home=[], on_court_away=[], live_box={}, pbp_context=LivePBPContext(),
                game_code=str(game_summary.get("gameCode", "")), game_status=int(game_summary.get("gameStatus", 1) or 1)
            )
        game = data.get("game", {})
        period = max(1, int(game.get("period", 1) or 1))
        clock_str = game.get("gameClock", "PT12M00.00S")
        secs_rem, clock_disp = cls.parse_clock_to_seconds(clock_str, period)

        home, away = game.get("homeTeam", {}), game.get("awayTeam", {})
        live_box = {}
        on_court_home, on_court_away = [], []

        for team_obj, on_court_list in [(home, on_court_home), (away, on_court_away)]:
            for p in team_obj.get("players", []):
                pid = int(p["personId"])
                s = p.get("statistics", {})
                min_str = s.get("minutes", "PT00M00.00S").replace("PT", "").replace("S", "")
                mins_played = (float(min_str.split("M")[0]) + float(min_str.split("M")[1]) / 60.0) if "M" in min_str else 0.0

                live_box[pid] = {
                    "PTS": int(s.get("points", 0) or 0),
                    "REB": int(s.get("reboundsTotal", 0) or 0),
                    "AST": int(s.get("assists", 0) or 0),
                    "3PM": int(s.get("threePointersMade", 0) or 0),
                    "FGA": int(s.get("fieldGoalsAttempted", 0) or 0),
                    "FTA": int(s.get("freeThrowsAttempted", 0) or 0),
                    "ORB": int(s.get("reboundsOffensive", 0) or 0),
                    "STL": int(s.get("steals", 0) or 0),
                    "BLK": int(s.get("blocks", 0) or 0),
                    "TOV": int(s.get("turnovers", 0) or 0),
                    "PF": int(s.get("foulsPersonal", 0) or 0),
                    "MIN": mins_played,
                    "ACTIVE": str(p.get("status", "ACTIVE")).upper() != "INACTIVE",
                    "STARTER": str(p.get("starter", "")) == "1",
                }
                if p.get("oncourt") == "1":
                    on_court_list.append(pid)

        # PREGAME_CLOCK_FIX: before tip the box shows period 1 / clock 0:00 with nobody on the floor yet.
        if period == 1 and not any(v["MIN"] > 0 for v in live_box.values()):
            secs_rem, clock_disp = 2880.0, "Q1 12:00"
        pbp_ctx = NBALivePlayByPlayParser.fetch_and_parse_pbp(full_nba_game_id, secs_rem, on_court_home[:5] + on_court_away[:5], period)
        if pbp_ctx.last_possession_team_id == away_id:
            poss_team = "AWAY"
        elif pbp_ctx.last_possession_team_id == home_id:
            poss_team = "HOME"
        else:
            poss_team = "UNKNOWN"     # the simulation then draws possession 50/50 (was: silently HOME)

        return LiveInPlayState(
            game_id=full_nba_game_id, matchup_title=f"{away_tri} @ {home_tri}",
            clock_display=clock_disp, period=period, seconds_remaining_in_game=secs_rem,
            home_team_id=home_id, away_team_id=away_id, home_alias=home_tri, away_alias=away_tri,
            home_score=int(home.get("score", 0)), away_score=int(away.get("score", 0)),
            possession_team=poss_team, on_court_home=on_court_home[:5],
            on_court_away=on_court_away[:5], live_box=live_box, pbp_context=pbp_ctx,
            game_code=str(game_summary.get("gameCode", "")),
            game_status=int(game.get("gameStatus", game_summary.get("gameStatus", 1)) or 1)
        )

class RoleCalibratedMarkovCompiler:
    LG_AVG_STL_100 = 1.45
    LG_AVG_BLK_100 = 1.15

    def __init__(
        self,
        offense: List[DTPlayerSkillProfile],
        defense: List[DTPlayerSkillProfile],
        opp_team_depm_baseline: float = 0.0,
        live_box: Optional[Dict[int, Dict[str, float]]] = None,
        pbp_context: Optional[LivePBPContext] = None,
        current_period: int = 1,
        is_clutch_stint: bool = False
    ):
        self.offense = sorted(offense, key=lambda x: (x.position_code, -x.p_ast_100))
        self.defense = sorted(defense, key=lambda x: (x.position_code, x.inches))
        self.opp_team_depm_baseline = opp_team_depm_baseline
        self.live_box = live_box or {}
        self.pbp = pbp_context or LivePBPContext()
        self.current_period = current_period
        self.is_clutch_stint = is_clutch_stint

        self.Q, self.R, self.init_probs = self._compile_matrices()
        self.P_cum = np.cumsum(np.hstack([self.Q, self.R]), axis=1)
        self.init_cum = np.cumsum(self.init_probs)

    def _foul_discount(self, pid: int) -> float:
        pf = self.live_box.get(pid, {}).get("PF", 0)
        if (self.current_period == 2 and pf >= 3) or (self.current_period == 3 and pf >= 4) or (self.current_period >= 4 and pf >= 5):
            return 0.88
        return 1.0

    def _stint_fatigue_penalty(self, pid: int) -> float:
        stint_mins = self.pbp.current_stint_minutes.get(pid, 0.0)
        return max(FATIGUE_FLOOR, 1.0 - (stint_mins - FATIGUE_START_MIN) * FATIGUE_SLOPE) if stint_mins > FATIGUE_START_MIN else 1.0

    def _role_adjusted_usage(self, p: DTPlayerSkillProfile, helio_on_court: bool) -> float:
        base_usg = p.p_usg
        role = p.role_archetype
        if role == "Heliocentric Creator" and self.is_clutch_stint:
            base_usg *= HELIO_CLUTCH_MULT
        elif role == "Secondary Scorer" and not helio_on_court:
            base_usg *= SECONDARY_NO_HELIO_MULT

        stats = self.live_box.get(p.player_id)
        if not stats or stats.get("MIN", 0.0) < 4.0:
            return base_usg
        mins = stats["MIN"]
        live_actions = stats.get("FGA", 0) + 0.44 * stats.get("FTA", 0) + stats.get("TOV", 0)
        team_act, team_min = stats.get("_team_actions", 0.0), stats.get("_team_minutes", 0.0)
        if team_act > 0 and team_min > 0:
            # USG% = actions x (team minutes / 5) / (minutes x team actions): the player's share of his team's
            # possessions while he is on the floor (replaces the rough actions-per-minute x 0.48 conversion)
            obs_usg = float(np.clip(live_actions * (team_min / 5.0) / (mins * team_act), 0.06, 0.45))
            w_live = min(0.25, team_act / 80.0)          # weight grows with team possessions observed
        else:
            obs_usg = float(np.clip((live_actions / mins) * 0.48, 0.08, 0.45))
            w_live = min(0.15, mins / 150.0)

        # v2026.25: no hard bench cap. A bench player promoted by injuries/fouls is a real role change; the live weight
        # w_live (<= 0.25, growing with team possessions observed) is the sample-size shrinkage toward D&T.

        return (1.0 - w_live) * base_usg + w_live * obs_usg

    def _shot_mix(self, p: DTPlayerSkillProfile, usg_scalar: float) -> Tuple[float, float, float]:
        """Attempts per 100 by zone. Total = D&T total x usage scalar (unchanged); the SPLIT is a Dirichlet posterior:
        prior shares from D&T worth SHOT_MIX_KAPPA attempts + tonight's attempts by zone. Shooting percentages stay D&T:
        where a player shoots from tonight is information, whether the first few went in mostly is not."""
        prior = np.array([max(0.0, p.p_fga_rim_100), max(0.0, p.p_fga_mid_100), max(0.0, p.p_fg3a_100)])
        tot = float(prior.sum())
        if tot <= 0:
            return 0.4, 0.4, 0.4
        live = np.array([self.pbp.live_rim_fga.get(p.player_id, 0), self.pbp.live_mid_fga.get(p.player_id, 0),
                         self.pbp.live_3pt_fga.get(p.player_id, 0)], dtype=float)
        share = (SHOT_MIX_KAPPA * prior / tot + live) / (SHOT_MIX_KAPPA + live.sum())
        return tuple(float(max(0.4, x)) for x in tot * usg_scalar * share)

    def _compile_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Q = np.zeros((5, 5))
        R = np.zeros((5, 9))

        helio_on_court = any(p.role_archetype == "Heliocentric Creator" for p in self.offense)
        active_def_depm = np.mean([d.depm * self._foul_discount(d.player_id) for d in self.defense])
        depm_live_delta = active_def_depm - (self.opp_team_depm_baseline / 5.0)

        frontcourt_blk = np.mean([self.defense[i].p_blk_100 * self._foul_discount(self.defense[i].player_id) for i in [3, 4]])
        rim_prot_mult = np.clip(frontcourt_blk / self.LG_AVG_BLK_100, 0.70, 1.45)
        team_stl = np.mean([d.p_stl_100 * self._foul_discount(d.player_id) for d in self.defense])
        stl_mult = np.clip(team_stl / self.LG_AVG_STL_100, 0.75, 1.40)
        # Extra matchup layer scaled continuously by alpha = MATCHUP_STRENGTH in [0, 1] (0 = D&T game-optimized skills
        # only; 1 = the full v2026.24 layer). Multipliers are scaled in logs so alpha interpolates smoothly.
        alpha = float(MATCHUP_STRENGTH)
        depm_live_delta *= alpha
        rim_prot_mult = float(np.exp(alpha * np.log(max(rim_prot_mult, 1e-6))))
        stl_mult = float(np.exp(alpha * np.log(max(stl_mult, 1e-6))))

        usg_vec = np.array([self._role_adjusted_usage(p, helio_on_court) for p in self.offense])
        init_scores = np.array([0.65 * p.p_ast_100 + 0.35 * (usg_vec[i] * 100.0) for i, p in enumerate(self.offense)])
        init_probs = init_scores / np.sum(init_scores)

        for i in range(5):
            off_p = self.offense[i]
            def_p = self.defense[i]
            disc = self._foul_discount(def_p.player_id)
            fatigue_mod = self._stint_fatigue_penalty(off_p.player_id)

            ind_def = np.exp(alpha * (def_p.depm * disc) * 0.035)
            ind_stl = float(np.exp(alpha * np.log(np.clip((def_p.p_stl_100 * disc) / self.LG_AVG_STL_100, 0.75, 1.40))))
            usg_scalar = (usg_vec[i] / max(0.08, off_p.p_usg)) * fatigue_mod

            fga_rim, fga_mid, fga_3pt = self._shot_mix(off_p, usg_scalar)

            total_foul_events = max(0.2, (off_p.p_fta_100 * usg_scalar) * 0.44)
            tov_100 = max(0.4, off_p.p_tov_100 * (usg_scalar ** 0.7))
            est_passes_100 = (off_p.p_ast_100 * 3.6) + 18.0

            adj_rim = fga_rim / (0.65 * rim_prot_mult + 0.35 * ind_def)
            adj_mid = fga_mid / ind_def
            adj_3pt = fga_3pt / (ind_def ** 0.75)
            adj_foul2 = total_foul_events * 0.88
            adj_foul3 = total_foul_events * 0.12
            adj_tov = tov_100 * (0.6 * ind_stl + 0.4 * stl_mult)

            eff_scalar = np.exp(-depm_live_delta * 0.012) * (fatigue_mod ** 0.5)
            adj_rim_pct = np.clip(off_p.p_fgpct_rim * eff_scalar / (rim_prot_mult ** 0.25), 0.35, 0.85)
            adj_mid_pct = np.clip(off_p.p_fgpct_mid * eff_scalar, 0.25, 0.62)
            adj_3pt_pct = np.clip(off_p.p_fg3pct * eff_scalar, 0.20, 0.55)

            pass_targets = np.copy(usg_vec)
            pass_targets[i] = 0.0
            pass_targets = (pass_targets / np.sum(pass_targets)) * est_passes_100

            row_sum = np.sum(pass_targets) + adj_rim + adj_mid + adj_3pt + adj_foul2 + adj_foul3 + adj_tov
            Q[i, :] = pass_targets / row_sum

            R[i, 0] = (adj_rim * adj_rim_pct) / row_sum
            R[i, 1] = (adj_rim * (1.0 - adj_rim_pct)) / row_sum
            R[i, 2] = (adj_mid * adj_mid_pct) / row_sum
            R[i, 3] = (adj_mid * (1.0 - adj_mid_pct)) / row_sum
            R[i, 4] = (adj_3pt * adj_3pt_pct) / row_sum
            R[i, 5] = (adj_3pt * (1.0 - adj_3pt_pct)) / row_sum
            R[i, 6] = adj_foul2 / row_sum
            R[i, 7] = adj_foul3 / row_sum
            R[i, 8] = adj_tov / row_sum

        return Q, R, init_probs

SEG_SEC = 120.0          # legacy constant (v2026.25 and earlier: fixed 2-minute lineup segments)
LINEUP_HYST_SEC = 90.0   # a bench player replaces an on-court player only with this much more remaining allocation
PF_RATE_PRIOR = (0.050, 0.075)   # personal fouls per minute played: guards/wings, bigs (position code >= 4)
PF_PRIOR_MIN = 12.0      # prior strength in minutes for the live foul-rate update
PF_RATE_CLAMP = (0.015, 0.12)   # temporary bounds (NOT calibrated truth): every bind is logged in FOUL_CLAMP_LOG
FOUL_CLAMP_LOG: List[dict] = []


def segment_lengths(T_sec: float) -> List[float]:
    """Lineup segments over the remaining game: 45 s while more than 12:00 remain, 30 s to 6:00, 15 s after.
    v2026.25 used fixed 120 s blocks, so a player with 0.4 minutes left could play a full 2 minutes, and the final
    6 minutes had only three lineup decisions."""
    out, t = [], float(T_sec)
    while t > 1e-9:
        L = min(15.0 if t <= 360.0 else (30.0 if t <= 720.0 else 45.0), t)
        out.append(L)
        t -= L
    return out or [max(1e-6, float(T_sec))]
CALIB_STATS = (("PTS", "p_pts"), ("REB", "p_reb"), ("AST", "p_ast"), ("3PM", "p_fg3m"))
CALIB_TRUST = (0.67, 1.50)   # raw D&T/sim ratio outside this band => stat flagged untradeable
ENGINE_VERSION = "2026.29"    # calibrations are only valid for the engine version they were fitted on
# v2026.25: the extra matchup layer (DEPM, rim protection, steal pressure on top of D&T's GAME-OPTIMIZED skills, which
# already account for opponents) is a coefficient, not a switch. Production default 0 = D&T only, until the logged A/B
# (p_shadow = strength 1) earns it weight out of sample. Older code used MATCHUP_LAYER (True == strength 1).
MATCHUP_STRENGTH = 0.0
FOULOUT_HAZARD = True        # v2026.26: minutes scenarios can draw a future sixth foul
DISPERSION_K = 1.0           # v2026.21: shrink removed. Nov 5 outcomes: engine spreads were ~20% too NARROW, not too wide
MINUTES_CV = 0.15            # player-specific night-to-night uncertainty in remaining minutes (fouls, matchups, coach)
ROTATION_CV = 0.10           # team-level rotation shock: starters up / bench down together (close game) or the reverse (blowout)
PACE_CV = 0.04               # shared game-environment shock: the whole game plays faster or slower in a scenario
MAX_OT = 3                   # overtime periods simulated if regulation (or the current OT) ends tied
N_SCENARIOS = 24             # minutes scenarios per run; each scenario has its own allocation and lineup schedule
# v2026.28 (H1 specialists): live-information priors. Each is a PRIOR STRENGTH in the unit of the evidence it is pooled
# with, chosen from first principles and logged for walk-forward fitting; none is fitted yet.
SHOT_MIX_KAPPA = 30.0        # shot-zone mix: prior worth 30 attempts (5 hot attempts move a 30% three-rate ~4 points)
ASSIST_ALPHA = 12.0          # assister routing: prior worth 12 assists spread over teammates by D&T ast/100
PACE_TAU0 = 180.0            # pace: prior worth 180 team possessions. From sampling noise: a quarter (~25 possessions)
                             # gives a pace s.d. ~8/48 vs a pregame pace error ~3, i.e. weight ~12% (old clock rule: 8.75%)
MINUTES_EVIDENCE_K = 10.0    # rotation evidence: weight = expected minutes so far / (that + 10); replaces elapsed/48
FTA_ALPHA = 20.0              # prior possessions for FTA opportunity-rate updating
TOV_ALPHA = 20.0              # prior possessions for turnover opportunity-rate updating
ORB_ALPHA = 20.0              # prior rebound opportunities for ORB capture-share updating
DRB_ALPHA = 20.0              # prior rebound opportunities for DRB capture-share updating
# Hand-coded role multipliers, v2026.28: shrunk to 1.0 (no effect) until they earn weight out of sample. v2026.27 used
# 1.12 (heliocentric creator in clutch) and 1.16 (secondary scorer with no creator on the floor): judgment calls that
# act on the whole remaining game, i.e. hardest in Q1-Q2.
HELIO_CLUTCH_MULT = 1.00
SECONDARY_NO_HELIO_MULT = 1.00
FATIGUE_START_MIN, FATIGUE_SLOPE, FATIGUE_FLOOR = 9.0, 0.015, 0.92   # unchanged values, now named parameters
# Remaining-minutes uncertainty by game phase. Values = the v2026.27 constants (no behavior change) until final minutes
# are logged and fitted; the hook exists so Q1 can be widened by data, not by hand.
MINUTES_CV_BY_PHASE = {"Q1": 0.15, "Q2": 0.15, "Q3": 0.15, "Q4": 0.15}
ROTATION_CV_BY_PHASE = {"Q1": 0.10, "Q2": 0.10, "Q3": 0.10, "Q4": 0.10}

def _load_validated_engine_artifacts():
    """Load only validated artifacts for this exact ENGINE_VERSION; otherwise preserve named safe defaults."""
    global SHOT_MIX_KAPPA, ASSIST_ALPHA, PACE_TAU0, MINUTES_EVIDENCE_K
    global FTA_ALPHA, TOV_ALPHA, ORB_ALPHA, DRB_ALPHA, MINUTES_CV_BY_PHASE, ROTATION_CV_BY_PHASE
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        p = os.path.join(here, "engine_priors.json")
        d = json.load(open(p))
        if d.get("validated") is True and d.get("engine_version") == ENGINE_VERSION:
            x = d.get("params", {})
            for name in ("SHOT_MIX_KAPPA","ASSIST_ALPHA","PACE_TAU0","MINUTES_EVIDENCE_K",
                         "FTA_ALPHA","TOV_ALPHA","ORB_ALPHA","DRB_ALPHA"):
                if name in x and np.isfinite(float(x[name])):
                    globals()[name] = float(x[name])
    except Exception:
        pass
    try:
        p = os.path.join(here, "rotation_variance.json")
        d = json.load(open(p))
        if d.get("validated") is True and d.get("engine_version") == ENGINE_VERSION:
            phase = d.get("phase", {})
            for ph in ("Q1","Q2","Q3","Q4"):
                if ph in phase:
                    if "minutes_cv" in phase[ph]:
                        MINUTES_CV_BY_PHASE[ph] = float(np.clip(phase[ph]["minutes_cv"], 0.05, 0.60))
                    if "rotation_cv" in phase[ph]:
                        ROTATION_CV_BY_PHASE[ph] = float(np.clip(phase[ph]["rotation_cv"], 0.03, 0.40))
    except Exception:
        pass

_load_validated_engine_artifacts()


def is_starter_now(p, box_row: Optional[dict], game_started: bool) -> bool:
    """Starter status, v2026.28: tonight's actual opening five once the game has started (NBA box 'starter' flag), else
    D&T's start probability, else (no data) the old projected-minutes proxy p_mp >= 26."""
    if game_started and box_row is not None and "STARTER" in box_row:
        return bool(box_row["STARTER"])
    if p.p_start is not None:
        return p.p_start >= 0.5
    return p.p_mp >= 26.0


def replacement_profile(pid: int, team_id: int, name: str = "") -> "DTPlayerSkillProfile":
    """Emergency profile for a player on the floor whom D&T did not project (two-way call-up, late scratch replacement):
    replacement-level rates, 12 projected minutes. He must be in the simulation, or the other four play 5-man minutes."""
    return DTPlayerSkillProfile(
        player_id=pid, player_name=name or f"replacement {pid}", team_id=team_id, position_code=3, inches=79, weight=215.0,
        p_pace=99.5, p_mp=12.0, p_usg=0.16, p_fga_rim_100=4.5, p_fga_mid_100=2.5, p_fg3a_100=5.0, p_fta_100=3.0,
        p_ast_100=3.0, p_tov_100=2.0, p_orb_100=1.5, p_drb_100=5.0, p_stl_100=1.2, p_blk_100=0.7, p_fgpct_rim=0.62,
        p_fgpct_mid=0.40, p_fg3pct=0.34, p_ftpct=0.75, oepm=-2.0, depm=-1.0, epm=-3.0, p_start=0.0)


def _water_fill(E: np.ndarray, C: np.ndarray, S: float, hard: Optional[np.ndarray] = None) -> np.ndarray:
    """Scale expected minutes E (capped by C) so the team plays exactly S minutes.
    C is the SOFT rotation cap (the second pass may relax it to time left so five players are always on the floor);
    `hard` is a PHYSICAL cap (fouled out, simulated sixth foul) that no pass may exceed."""
    H = np.where(C > 0, S / 5.0, 0.0) if hard is None else np.minimum(np.where(C > 0, S / 5.0, 0.0), hard)
    C = np.minimum(C, H)
    x = np.minimum(E, C)
    for cap in (C, H):   # 2nd pass relaxes soft caps to time left, never past a hard cap
        for _ in range(30):
            gap = S - x.sum()
            if gap <= 1e-6:
                break
            free = x < cap - 1e-9
            if not free.any():
                break
            w = np.where(free, np.maximum(E, 0.05), 0.0)
            x = np.minimum(cap, x + gap * w / w.sum())
        if S - x.sum() <= 1e-6:
            break
    if x.sum() > S + 1e-6:
        x *= S / x.sum()
    return np.minimum(x, H)


class _Side:
    __slots__ = ("idx", "opp", "P", "init", "ft", "pa2", "pa3", "acum", "reb", "dt_half", "steps")


class InPlayMarkovEngine:
    def __init__(
        self,
        live_state: LiveInPlayState,
        home_roster: List[DTPlayerSkillProfile],
        away_roster: List[DTPlayerSkillProfile],
        home_depm_base: float,
        away_depm_base: float,
        calib: Optional[Dict[int, Dict[str, float]]] = None,
        legacy_post_calibration: bool = False
    ):
        self.state = live_state
        self.home_roster = self._eligible(home_roster)
        self.away_roster = self._eligible(away_roster)
        self.home_depm_base = home_depm_base
        self.away_depm_base = away_depm_base
        self.calib = calib or {}
        self.legacy_post_calibration = bool(legacy_post_calibration)

    def _calibrated_profile(self, p: DTPlayerSkillProfile) -> DTPlayerSkillProfile:
        """Apply D&T calibration to generative event intensities, never to shooting percentages."""
        c = self.calib.get(p.player_id, {}) if self.calib else {}
        m_pts = float(np.clip(c.get("PTS", 1.0), *CALIB_TRUST))
        m_3 = float(np.clip(c.get("3PM", m_pts), *CALIB_TRUST))
        m_reb = float(np.clip(c.get("REB", 1.0), *CALIB_TRUST))
        m_ast = float(np.clip(c.get("AST", 1.0), *CALIB_TRUST))
        fta = max(0.0, p.p_fta_100 * m_pts)
        tov = max(0.0, p.p_tov_100)
        orb = max(0.0, p.p_orb_100 * m_reb)
        drb = max(0.0, p.p_drb_100 * m_reb)
        pbp = self.state.pbp_context
        box = self.state.live_box.get(p.player_id, {})
        team_poss = float(box.get("_team_actions", 0.0) or 0.0)
        if team_poss > 0:
            obs_fta = float(pbp.live_fta.get(p.player_id, box.get("FTA", 0)) or 0.0) * 100.0 / team_poss
            obs_tov = float(pbp.live_tov.get(p.player_id, box.get("TOV", 0)) or 0.0) * 100.0 / team_poss
            fta = (FTA_ALPHA * fta + team_poss * obs_fta) / (FTA_ALPHA + team_poss)
            tov = (TOV_ALPHA * tov + team_poss * obs_tov) / (TOV_ALPHA + team_poss)
        team_roster = self.home_roster if p.team_id == self.state.home_team_id else self.away_roster
        orb_total = sum(max(0.2, q.p_orb_100 * float(np.clip(self.calib.get(q.player_id, {}).get("REB", 1.0), *CALIB_TRUST))) for q in team_roster)
        drb_total = sum(max(0.8, q.p_drb_100 * float(np.clip(self.calib.get(q.player_id, {}).get("REB", 1.0), *CALIB_TRUST))) for q in team_roster)
        oo = float(pbp.live_orb_opportunities.get(p.team_id, 0) or 0.0)
        do = float(pbp.live_drb_opportunities.get(p.team_id, 0) or 0.0)
        if oo > 0 and orb_total > 0:
            prior_share = orb / orb_total
            post_share = (ORB_ALPHA * prior_share + float(pbp.live_orb.get(p.player_id, box.get("ORB", 0)) or 0.0)) / (ORB_ALPHA + oo)
            orb = max(0.0, post_share * orb_total)
        if do > 0 and drb_total > 0:
            prior_share = drb / drb_total
            post_share = (DRB_ALPHA * prior_share + float(pbp.live_drb.get(p.player_id, 0) or 0.0)) / (DRB_ALPHA + do)
            drb = max(0.0, post_share * drb_total)
        return replace(
            p,
            p_fga_rim_100=max(0.0, p.p_fga_rim_100 * m_pts),
            p_fga_mid_100=max(0.0, p.p_fga_mid_100 * m_pts),
            p_fg3a_100=max(0.0, p.p_fg3a_100 * m_3),
            p_fta_100=fta,
            p_tov_100=tov,
            p_orb_100=orb,
            p_drb_100=drb,
            p_ast_100=max(0.0, p.p_ast_100 * m_ast),
        )

    def _eligible(self, roster: List[DTPlayerSkillProfile]) -> List[DTPlayerSkillProfile]:
        box = self.state.live_box
        keep = [p for p in roster
                if box.get(p.player_id, {}).get("ACTIVE", True) and box.get(p.player_id, {}).get("PF", 0) < NBAStintEngine.FOUL_OUT]
        on_court = set(self.state.on_court_home) | set(self.state.on_court_away)
        core = sorted(keep, key=lambda x: -x.p_mp)[:10]
        extra = [p for p in keep if p not in core and (p.player_id in on_court or float(box.get(p.player_id, {}).get("MIN", 0.0)) > 0)]
        out = core + extra[:3]   # unexpected rotation players (injury, fouls, garbage time) stay in the simulation
        # v2026.28: a player ON THE FLOOR whom D&T never projected gets a replacement-level emergency profile.
        if roster:
            team_id = roster[0].team_id
            side = self.state.on_court_home if team_id == self.state.home_team_id else self.state.on_court_away
            known = {p.player_id for p in roster}
            for pid in side:
                if pid not in known and box.get(pid, {}).get("ACTIVE", True) and box.get(pid, {}).get("PF", 0) < NBAStintEngine.FOUL_OUT:
                    out.append(replacement_profile(pid, team_id))
        return out

    def _game_started(self) -> bool:
        return any(float(v.get("MIN", 0.0)) > 0 for v in self.state.live_box.values())

    def _on_court_ids(self) -> set:
        return set(self.state.on_court_home) | set(self.state.on_court_away)

    # ── minutes ─────────────────────────────────────────────
    def _allocate_minutes(self, roster, T_sec: float, team_diff: int, return_parts: bool = False):
        E, C = np.zeros(len(roster)), np.zeros(len(roster))
        for k, p in enumerate(roster):
            b = self.state.live_box.get(p.player_id, {})
            kw = dict(is_starter=is_starter_now(p, b, self._game_started()), current_minutes_played=float(b.get("MIN", 0.0)),
                      current_sec_rem=T_sec, fouls=int(b.get("PF", 0)), score_diff=team_diff, proj_minutes=p.p_mp,
                      period=self.state.period, completed_stints=int(self.state.pbp_context.completed_stints.get(p.player_id, 0)),
                      on_court=p.player_id in self._on_court_ids(), evidence_k=MINUTES_EVIDENCE_K)
            C[k] = NBAStintEngine.get_remaining_minute_cap(**kw)
            E[k] = min(C[k], NBAStintEngine.expected_remaining_minutes(**kw))
        x = _water_fill(E, C, 5.0 * T_sec / 60.0)
        return (x, E, C) if return_parts else x

    @staticmethod
    def _schedule(roster, alloc_min: np.ndarray, on_court: List[int], T_sec: float, use_current: bool,
                  seg_lens: Optional[List[float]] = None, blowout: bool = False) -> List[Tuple[int, ...]]:
        """Lineup per segment with persistence: the five on the floor stay unless a player has used his allocation
        (less than half a segment left) or a bench player has LINEUP_HYST_SEC more remaining allocation than the
        weakest on-court player. Short segments without persistence would substitute every 15 s."""
        seg_lens = seg_lens or segment_lengths(T_sec)
        rem = alloc_min * 60.0
        pmp = [p.p_mp for p in roster]
        ok = lambda k: alloc_min[k] > 0
        rank = lambda k: ((0 if (blowout and pmp[k] <= 24.0) else (1 if blowout else 0)), -(rem[k] if ok(k) else -1e9), -pmp[k])
        cur = [] if blowout else ([k for k, p in enumerate(roster) if p.player_id in set(on_court) and ok(k)][:5] if use_current else [])
        out = []
        for s, L in enumerate(seg_lens):
            lineup = [k for k in cur if ok(k) and rem[k] >= 0.5 * L]
            bench = sorted([k for k in range(len(roster)) if k not in lineup], key=rank)
            while len(lineup) < 5 and bench:
                lineup.append(bench.pop(0))
            if len(lineup) == 5:
                while bench and ok(bench[0]):
                    worst = min(lineup, key=lambda k: rem[k] if ok(k) else -1e9)
                    if rem[bench[0]] > rem[worst] + LINEUP_HYST_SEC or not ok(worst):
                        lineup[lineup.index(worst)] = bench.pop(0)
                        bench.append(worst)
                        bench.sort(key=rank)
                    else:
                        break
            for k in lineup:
                rem[k] -= L
            cur = lineup
            out.append(tuple(sorted(lineup)))
        return out

    def _pf_rate(self, p, pf: int, mins: float, rem_min: float) -> float:
        """Shrunk foul rate per minute, clamped to PF_RATE_CLAMP; every bind is recorded once per player and state so a
        frequently binding clamp (i.e. a wrong estimator) is visible rather than masked."""
        prior = PF_RATE_PRIOR[1] if p.position_code >= 4 else PF_RATE_PRIOR[0]
        raw = (prior * PF_PRIOR_MIN + pf) / (PF_PRIOR_MIN + mins)
        rate = float(np.clip(raw, *PF_RATE_CLAMP))
        if rate != raw:
            st = self.state
            key = (st.game_id, st.pbp_context.last_action_number, p.player_id)
            seen = getattr(self, "_clamp_seen", set())
            if key not in seen:
                seen.add(key)
                self._clamp_seen = seen
                FOUL_CLAMP_LOG.append(dict(player=p.player_name, raw_rate=round(raw, 4), clamped=rate, position=p.position_code,
                                           pf=pf, minutes=round(mins, 1), period=st.period, clock=st.clock_display,
                                           score_diff=st.home_score - st.away_score, proj_remaining_min=round(rem_min, 2)))
        return rate

    def _foulout_caps(self, roster, alloc_min: np.ndarray, rng) -> np.ndarray:
        """Remaining-minute caps from a simulated sixth foul: future fouls ~ Poisson(rate x allocated minutes), the rate
        shrunk from a position prior toward tonight's PF per minute; if the sixth arrives, its time ~ the k-th arrival
        of N uniform on the stint, i.e. alloc x Beta(k, N - k + 1). Inf = no foul-out."""
        caps = np.full(len(roster), np.inf)
        for k, p in enumerate(roster):
            a = float(alloc_min[k])
            if a <= 0:
                continue
            b = self.state.live_box.get(p.player_id, {})
            pf, mins = int(b.get("PF", 0)), float(b.get("MIN", 0.0))
            need = NBAStintEngine.FOUL_OUT - pf
            if need <= 0:
                caps[k] = 0.0
                continue
            rate = self._pf_rate(p, pf, mins, a)
            n = rng.poisson(rate * a)
            if n >= need:
                caps[k] = a * rng.beta(need, n - need + 1)
        return caps

    # ── pace ────────────────────────────────────────────────
    def _game_pace(self, T0: float) -> float:
        pace = float(np.mean([p.p_pace for p in (self.home_roster + self.away_roster)])) if (self.home_roster + self.away_roster) else 99.5
        elapsed = 60.0 * NBAStintEngine.elapsed_minutes(self.state.period, T0)   # period-aware (OT)
        box = self.state.live_box
        if 360.0 <= elapsed and box:
            poss = []
            for roster_ids in ({p.player_id for p in self.home_roster} | set(self.state.on_court_home),
                               {p.player_id for p in self.away_roster} | set(self.state.on_court_away)):
                tb = [box[i] for i in box if i in roster_ids]
                poss.append(sum(b.get("FGA", 0) - b.get("ORB", 0) + b.get("TOV", 0) + 0.44 * b.get("FTA", 0) for b in tb))
            live_pace = float(np.mean(poss)) / (elapsed / 60.0) * 48.0
            if 70.0 < live_pace < 130.0:   # ignore broken / partial box data
                n_obs = float(np.mean(poss))
                w = n_obs / (n_obs + PACE_TAU0)    # v2026.28: weight by possessions observed (was 0.35 x elapsed / 48 min)
                pace = (1.0 - w) * pace + w * float(np.clip(live_pace, 88.0, 112.0))
        return pace

    @staticmethod
    def _steps_per_possession(c: RoleCalibratedMarkovCompiler, orb_share: float) -> float:
        v = c.init_probs @ np.linalg.inv(np.eye(5) - c.Q)       # expected visits per chain run
        B = v @ c.R                                              # absorption probabilities
        miss = B[1] + B[3] + B[5]
        return float(v.sum() / max(0.05, 1.0 - miss * orb_share))

    def run_monte_carlo(self, n_sims: int = 2000, seed: int = 42) -> Dict[str, any]:
        st = self.state
        T0 = float(max(0.0, st.seconds_remaining_in_game))
        all_players = {p.player_id: p for p in (self.home_roster + self.away_roster)}
        tracked_ids = list(all_players.keys())
        id_to_idx = {pid: i for i, pid in enumerate(tracked_ids)}
        n_p = len(tracked_ids)

        base = {k: np.zeros(n_p, dtype=np.int64) for k in ("PTS", "REB", "AST", "3PM")}
        for pid, i in id_to_idx.items():
            b = st.live_box.get(pid)
            if b:
                for k in base:
                    base[k][i] = int(b.get(k, 0))
        if T0 <= 0.0 or len(self.home_roster) < 5 or len(self.away_roster) < 5:
            reps = {k: np.tile(v, (n_sims, 1)) for k, v in base.items()}
            reps["PRA"] = reps["PTS"] + reps["REB"] + reps["AST"]
            return {"id_to_idx": id_to_idx, "players": all_players, **reps, "MIN_PROJ": {}, "SIM_PACE": 0.0}

        # ── minutes -> lineup schedule ──────────────────────
        diff = st.home_score - st.away_score
        h_alloc, hE, hC = self._allocate_minutes(self.home_roster, T0, diff, return_parts=True)
        a_alloc, aE, aC = self._allocate_minutes(self.away_roster, T0, -diff, return_parts=True)
        per_clock = T0 - max(0, 4 - st.period) * 720.0 if st.period <= 4 else T0
        use_current = st.game_status == 2 and 0.0 < per_clock < 720.0
        S_team = 5.0 * T0 / 60.0

        # team totals for live usage (USG%): actions and minutes of everyone on each roster with a box line
        aug = {pid: dict(b) for pid, b in st.live_box.items()}
        for roster in (self.home_roster, self.away_roster):
            rows = [aug[p.player_id] for p in roster if p.player_id in aug]
            t_act = sum(r.get("FGA", 0) + 0.44 * r.get("FTA", 0) + r.get("TOV", 0) for r in rows)
            t_min = sum(float(r.get("MIN", 0.0)) for r in rows)
            for r in rows:
                r["_team_actions"], r["_team_minutes"] = t_act, t_min
        self._aug_box = aug

        pace = self._game_pace(T0)
        sec_per_poss = 2880.0 / (2.0 * pace)
        # Minutes are uncertain: each scenario draws every player's remaining minutes around the expectation
        # (lognormal, CV = MINUTES_CV), re-fills each team to exactly 5 x time left within the foul/blowout caps,
        # and builds its own lineup schedule. A player's PTS, REB and AST then move together with his minutes,
        # which is the correlation PRA needs. Without this every simulated game had identical minutes.
        phase = f"Q{min(4, max(1, st.period))}"
        m_cv = MINUTES_CV_BY_PHASE.get(phase, MINUTES_CV)
        r_cv = ROTATION_CV_BY_PHASE.get(phase, ROTATION_CV)
        n_scen = max(1, min(N_SCENARIOS, n_sims // 50)) if m_cv > 0 else 1
        srng = np.random.default_rng(seed + 7)
        sig = math.sqrt(math.log(1.0 + m_cv ** 2))
        # Stratified draws: each player's n_scen multipliers are the n_scen normal quantiles in a random order, so his
        # average minutes across scenarios equal the expectation exactly (random draws left +-3.7% noise in the mean).
        q = np.array([NormalDist().inv_cdf((i + 0.5) / n_scen) for i in range(n_scen)]) if n_scen > 1 else np.zeros(1)
        zh = np.array([srng.permutation(q) for _ in range(len(hE))]).T
        za = np.array([srng.permutation(q) for _ in range(len(aE))]).T
        sig_r = math.sqrt(math.log(1.0 + r_cv ** 2))
        ch, ca = srng.permutation(q), srng.permutation(q)          # one rotation shock per team per scenario
        sig_p = math.sqrt(math.log(1.0 + PACE_CV ** 2))
        # mean-one lognormal duration multiplier: exp(-s z - s^2/2) (v2026.24 had exp(-s z + s^2/2), mean exp(s^2) > 1)
        scen_dt = np.exp(-sig_p * srng.permutation(q) - 0.5 * sig_p * sig_p) if n_scen > 1 else np.ones(1)
        gs = self._game_started()
        h_role = np.array([1.0 if is_starter_now(p, st.live_box.get(p.player_id), gs) else -1.0 for p in self.home_roster])
        a_role = np.array([1.0 if is_starter_now(p, st.live_box.get(p.player_id), gs) else -1.0 for p in self.away_roster])
        packs, scen_packs, scen_blowout_packs, min_proj = {}, [], [], {}
        seg_lens = segment_lengths(T0)
        seg_start = [0.0]
        for L in seg_lens[:-1]:
            seg_start.append(seg_start[-1] + L)
        seg_edges = [a + L for a, L in zip(seg_start, seg_lens)][:-1]    # elapsed-time boundaries for the sim loop
        frng = np.random.default_rng(seed + 11)
        for j in range(n_scen):
            if n_scen == 1:
                ha, aa = h_alloc, a_alloc
                hEj, aEj = hE, aE
            else:
                hEj = hE * np.exp(sig * zh[j] - sig * sig / 2 + h_role * sig_r * ch[j] - sig_r * sig_r / 2)
                aEj = aE * np.exp(sig * za[j] - sig * sig / 2 + a_role * sig_r * ca[j] - sig_r * sig_r / 2)
                ha = _water_fill(hEj, hC, S_team)
                aa = _water_fill(aEj, aC, S_team)
            if FOULOUT_HAZARD:
                for roster, Ej, C, which in ((self.home_roster, hEj, hC, "h"), (self.away_roster, aEj, aC, "a")):
                    alloc = ha if which == "h" else aa
                    caps = self._foulout_caps(roster, alloc, frng)
                    if np.isfinite(caps).any():
                        newa = _water_fill(np.minimum(Ej, caps), C, S_team, hard=caps)   # foul-out caps are hard
                        if which == "h":
                            ha = newa
                        else:
                            aa = newa
            h_sched = self._schedule(self.home_roster, ha, st.on_court_home, T0, use_current, seg_lens)
            a_sched = self._schedule(self.away_roster, aa, st.on_court_away, T0, use_current, seg_lens)
            h_blow = self._schedule(self.home_roster, ha, st.on_court_home, T0, False, seg_lens, blowout=True)
            a_blow = self._schedule(self.away_roster, aa, st.on_court_away, T0, False, seg_lens, blowout=True)
            seg_packs, blow_packs = [], []
            for s, (hs, as_, hb, ab) in enumerate(zip(h_sched, a_sched, h_blow, a_blow)):
                clutch = (T0 - seg_start[s]) <= 300.0 and abs(diff) <= 10
                key = (hs, as_, clutch)
                if key not in packs:
                    packs[key] = self._compile_pair([self.home_roster[k] for k in hs], [self.away_roster[k] for k in as_],
                                                    clutch, id_to_idx, sec_per_poss)
                seg_packs.append(packs[key])
                bkey = (hb, ab, False)
                if bkey not in packs:
                    packs[bkey] = self._compile_pair([self.home_roster[k] for k in hb], [self.away_roster[k] for k in ab],
                                                     False, id_to_idx, sec_per_poss)
                blow_packs.append(packs[bkey])
            scen_packs.append(seg_packs)
            scen_blowout_packs.append(blow_packs)
            for roster, sched in ((self.home_roster, h_sched), (self.away_roster, a_sched)):
                for s, pick in enumerate(sched):
                    L = seg_lens[s] / 60.0 / n_scen
                    for k in pick:
                        min_proj[roster[k].player_id] = min_proj.get(roster[k].player_id, 0.0) + L
        for pid in min_proj:
            min_proj[pid] += float(st.live_box.get(pid, {}).get("MIN", 0.0))
        seg_packs = scen_packs[0]

        # ── simulation (stdlib RNG + bisect: ~10x faster than per-step numpy calls) ──
        rnd = random.Random(seed)
        R, LOG, BIS = rnd.random, math.log, bisect.bisect_right
        last = len(scen_packs[0]) - 1
        if st.game_status == 2 and st.possession_team in ("HOME", "AWAY"):
            poss0 = 0 if st.possession_team == "HOME" else 1
        else:
            poss0 = -1
        out = {k: np.zeros((n_sims, n_p), dtype=np.int64) for k in ("PTS", "REB", "AST", "3PM")}
        n_poss = 0
        h_idx = [id_to_idx[p.player_id] for p in self.home_roster]
        a_idx = [id_to_idx[p.player_id] for p in self.away_roster]
        h0, a0 = st.home_score, st.away_score

        for sim in range(n_sims):
            seg_packs = scen_packs[sim % n_scen]
            blow_packs = scen_blowout_packs[sim % n_scen]
            pf = float(scen_dt[sim % n_scen])
            pts, reb, ast, tpm = [0] * n_p, [0] * n_p, [0] * n_p, [0] * n_p
            clock = T0
            ot_left, in_ot = MAX_OT, False
            poss = poss0 if poss0 >= 0 else (0 if R() < 0.5 else 1)
            si = 0
            pk = seg_packs[0]
            sd = pk[poss]
            h = min(4, BIS(sd.init, R()))
            while True:
                clock += pf * sd.dt_half * LOG(R() * R() + 1e-300)    # Gamma(2) touch duration, scenario pace
                if clock <= 0.0:
                    # Tied at the horn -> 5-minute overtime on the closing lineups (up to MAX_OT periods).
                    if ot_left > 0 and h0 + sum(pts[i] for i in h_idx) == a0 + sum(pts[i] for i in a_idx):
                        ot_left, in_ot, clock = ot_left - 1, True, 300.0
                        si = last
                        pk = seg_packs[last]
                        poss = 0 if R() < 0.5 else 1          # overtime opens with a jump ball, not the last possession
                        sd = pk[poss]
                        h = min(4, BIS(sd.init, R()))
                        continue
                    break
                s_new = last if in_ot else BIS(seg_edges, T0 - clock)
                if s_new > last:
                    s_new = last
                if s_new != si:
                    si = s_new
                    sim_margin = (h0 + sum(pts[i] for i in h_idx)) - (a0 + sum(pts[i] for i in a_idx))
                    p_blow = pathwise_blowout_probability(sim_margin, clock)
                    pk = blow_packs[si] if R() < p_blow else seg_packs[si]
                    sd = pk[poss]
                nx = BIS(sd.P[h], R())
                if nx < 5:
                    h = nx
                    continue
                o = nx - 5 if nx < 14 else 8
                g = sd.idx[h]
                if o == 0 or o == 2:
                    pts[g] += 2
                    if R() < sd.pa2:
                        ast[sd.idx[BIS(sd.acum[h], R())]] += 1
                elif o == 4:
                    pts[g] += 3
                    tpm[g] += 1
                    if R() < sd.pa3:
                        ast[sd.idx[BIS(sd.acum[h], R())]] += 1
                elif o == 1 or o == 3 or o == 5:
                    rw = min(9, BIS(sd.reb, R()))
                    if rw < 5:
                        reb[sd.idx[rw]] += 1
                        h = rw
                        continue
                    reb[sd.opp[rw - 5]] += 1
                elif o == 6 or o == 7:
                    fp = sd.ft[h]
                    for _ in range(2 if o == 6 else 3):
                        if R() < fp:
                            pts[g] += 1
                poss ^= 1
                n_poss += 1
                sd = pk[poss]
                h = min(4, BIS(sd.init, R()))
            out["PTS"][sim], out["REB"][sim], out["AST"][sim], out["3PM"][sim] = pts, reb, ast, tpm

        # ── engine 2026.29: calibration is generative in _calibrated_profile / _compile_pair. ──
        # Legacy post-simulation thinning/thickening exists only for explicit diagnostic comparison.
        rng = np.random.default_rng(seed)
        for k in out:
            if self.legacy_post_calibration:
                M = np.array([self.calib.get(pid, {}).get(k, 1.0) for pid in tracked_ids])
                if self.calib and np.any(np.abs(M - 1.0) > 1e-9):
                    inc = out[k]
                    out[k] = rng.binomial(inc, np.minimum(M, 1.0)) + rng.binomial(inc, np.clip(M - 1.0, 0.0, 1.0))
            if DISPERSION_K != 1.0:
                inc = out[k].astype(float)
                mu = inc.mean(axis=0, keepdims=True)
                out[k] = np.maximum(0.0, np.floor(mu + DISPERSION_K * (inc - mu) + rng.random(inc.shape))).astype(np.int64)
            out[k] += base[k][None, :]
        sim_pace = (n_poss / n_sims) / 2.0 / (T0 / 60.0) * 48.0
        return {
            "id_to_idx": id_to_idx, "players": all_players,
            "PTS": out["PTS"], "REB": out["REB"], "AST": out["AST"], "3PM": out["3PM"],
            "PRA": out["PTS"] + out["REB"] + out["AST"],
            "MIN_PROJ": min_proj, "SIM_PACE": sim_pace, "TARGET_PACE": pace
        }

    def _compile_pair(self, h_u, a_u, clutch: bool, id_to_idx: Dict[int, int], sec_per_poss: float):
        st = self.state
        box = getattr(self, "_aug_box", None) or st.live_box
        h_eff = [self._calibrated_profile(p) for p in h_u]
        a_eff = [self._calibrated_profile(p) for p in a_u]
        c_h = RoleCalibratedMarkovCompiler(h_eff, a_eff, self.away_depm_base, box, st.pbp_context, st.period, clutch)
        c_a = RoleCalibratedMarkovCompiler(a_eff, h_eff, self.home_depm_base, box, st.pbp_context, st.period, clutch)
        sides = []
        for off, dfn in ((c_h, c_a), (c_a, c_h)):
            s = _Side()
            s.idx = [id_to_idx[p.player_id] for p in off.offense]
            s.opp = [id_to_idx[p.player_id] for p in dfn.offense]          # rebound slots aligned with s.opp
            s.P = [list(row) for row in off.P_cum]
            s.init = list(off.init_cum)
            s.ft = [p.p_ftpct for p in off.offense]
            orb = np.array([max(0.2, p.p_orb_100) for p in off.offense])
            drb = np.array([max(0.8, p.p_drb_100) for p in dfn.offense])
            tot = np.concatenate([orb, drb])
            s.reb = list(np.cumsum(tot / tot.sum()))
            s.steps = self._steps_per_possession(off, orb.sum() / tot.sum())
            # Assists: team assisted-FG rate = sum of on-court D&T ast/100 over the chain's FGM/100;
            # assister drawn among the shooter's teammates in proportion to their D&T ast/100.
            a100 = np.array([max(0.3, p.p_ast_100) for p in off.offense])
            v = off.init_probs @ np.linalg.inv(np.eye(5) - off.Q)
            B = v @ off.R
            runs = 1.0 / max(0.05, 1.0 - (B[1] + B[3] + B[5]) * (orb.sum() / tot.sum()))
            fgm2, fgm3 = (B[0] + B[2]) * runs * 100.0, B[4] * runs * 100.0
            pa = float(np.clip(a100.sum() / max(1.0, fgm2 + 1.2 * fgm3), 0.25, 0.90))
            s.pa2, s.pa3 = pa, min(0.97, 1.2 * pa)
            s.acum = []
            pairs = st.pbp_context.assist_pairs or {}
            for i in range(5):
                w = a100.copy(); w[i] = 0.0
                w = w / w.sum()
                # v2026.28: Dirichlet posterior over who assists this shooter: D&T prior worth ASSIST_ALPHA assists plus
                # tonight's (assister -> this shooter) counts from play-by-play.
                shooter = off.offense[i].player_id
                n = np.array([0.0 if k == i else float(pairs.get((off.offense[k].player_id, shooter), 0)) for k in range(5)])
                w = (ASSIST_ALPHA * w + n) / (ASSIST_ALPHA + n.sum())
                s.acum.append(list(np.cumsum(w / w.sum())[:4]))   # 4 cut points -> index 0..4
            s.dt_half = (sec_per_poss / s.steps) / 2.0
            sides.append(s)
        return sides


def calibrate_to_dt(
    home_roster: List[DTPlayerSkillProfile],
    away_roster: List[DTPlayerSkillProfile],
    box_df: pd.DataFrame,
    home_depm_base: float,
    away_depm_base: float,
    home_team_id: int,
    away_team_id: int,
    n_sims: int = 1500,
    seed: int = 7
) -> Tuple[Dict[int, Dict[str, float]], Dict[Tuple[int, str], bool], Dict[str, any]]:
    """Full-game sim from tip vs the D&T projected box -> per player/stat multipliers + trust flags."""
    st = LiveInPlayState(
        game_id="calib", matchup_title="calib", clock_display="Q1 12:00", period=1, seconds_remaining_in_game=2880.0,
        home_team_id=home_team_id, away_team_id=away_team_id, home_alias="H", away_alias="A", home_score=0, away_score=0,
        possession_team="HOME", on_court_home=[], on_court_away=[], live_box={}, game_status=1)
    out = InPlayMarkovEngine(st, home_roster, away_roster, home_depm_base, away_depm_base).run_monte_carlo(n_sims, seed)
    rows = {int(r["player_id"]): r for _, r in box_df.iterrows()} if not box_df.empty else {}
    calib, trusted = {}, {}
    for pid, i in out["id_to_idx"].items():
        r = rows.get(pid)
        if r is None:
            continue
        # v2 box projections are driven by p_mp_48 (incl. chance of not playing); the sim is conditional on playing.
        mp_all = float(r.get("p_mp_48") or r.get("p_mp") or 0.0)
        mp_act = float(r.get("p_mp_48_active") or mp_all or 0.0)
        scale = (mp_act / mp_all) if mp_all > 0 and mp_act > 0 else 1.0
        for stat, col in CALIB_STATS:
            tgt = r.get(col)
            if tgt is None or pd.isna(tgt):
                continue
            tgt = float(tgt) * scale
            sim_mean = float(out[stat][:, i].mean())
            if sim_mean < 0.25 or tgt < 0.25:
                calib.setdefault(pid, {})[stat], trusted[(pid, stat)] = 1.0, False
                continue
            raw = tgt / sim_mean
            calib.setdefault(pid, {})[stat] = float(np.clip(raw, 0.5, 2.0))
            trusted[(pid, stat)] = CALIB_TRUST[0] <= raw <= CALIB_TRUST[1]
    for pid in list(calib):
        trusted[(pid, "PRA")] = all(trusted.get((pid, s), False) for s in ("PTS", "REB", "AST"))
    return calib, trusted, out


class DunksAndThreesClient:
    """
    Dunks & Threes API Version 2 Client
    ==================================
    - Uses /api/v2 endpoints.
    - Implements HTTP ETag conditional caching (returns 304 Not Modified).
    - Uses scope=game on /epm to fetch only active game-day participants.
    - Captures p_pct_start, p_mp_48_active, pre_game_status, and team_depm_next.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = api_key
        self.session.headers["Accept"] = "application/json"
        self._cache: Dict[str, Tuple[str, List[dict]]] = {}  # url -> (etag, data)

    def _get(self, endpoint: str, params: dict = None) -> List[dict]:
        req = requests.Request("GET", f"{DT_BASE_URL}/{endpoint}", params=params).prepare()
        url = req.url
        headers = {}
        if url in self._cache:
            headers["If-None-Match"] = self._cache[url][0]

        try:
            resp = self.session.get(url, headers=headers, timeout=15)
            if resp.status_code == 304 and url in self._cache:
                return self._cache[url][1]
            if resp.status_code == 200:
                data = resp.json()
                etag = resp.headers.get("ETag", "")
                if etag:
                    self._cache[url] = (etag, data)
                return data
            raise RuntimeError(f"D&T V2 API Error [{resp.status_code}] on {url}: {resp.text}")
        except Exception as e:
            if url in self._cache:
                return self._cache[url][1]
            raise e

    def fetch_game_day_data(self, date_str: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # scope=game limits /epm strictly to teams playing on date_str
        epm_opt = pd.DataFrame(self._get("epm", {"date": date_str, "game_optimized": 1, "scope": "game"}))
        box_preds = pd.DataFrame(self._get("game-predictions-box", {"date": date_str}))
        team_epm = pd.DataFrame(self._get("team-epm", {"date": date_str}))
        return epm_opt, box_preds, team_epm

def _pstart(v) -> Optional[float]:
    """D&T start probability as 0-1 (accepts 0-1 or 0-100); None when missing."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return float(np.clip(x / 100.0 if x > 1.0 else x, 0.0, 1.0))


def build_dt_rosters_and_fallback_lines(
    epm_df: pd.DataFrame,
    box_df: pd.DataFrame,
    team_id: int
) -> Tuple[List[DTPlayerSkillProfile], List[MarketOffer]]:
    team_epm = epm_df[epm_df["team_id"] == team_id].copy()
    team_box = box_df[box_df["team_id"] == team_id].copy()
    box_lookup = {int(row["player_id"]): row for _, row in team_box.iterrows()}
    pos_map = {"G": 1, "PG": 1, "G-F": 2, "SG": 2, "F-G": 3, "SF": 3, "F": 4, "PF": 4, "F-C": 4, "C-F": 5, "C": 5}
    profiles = []

    for _, row in team_epm.iterrows():
        pid = int(row["player_id"])
        b_row = box_lookup.get(pid, {})
        if box_lookup and pid not in box_lookup:
            continue  # D&T does not project this player into tonight's game

        # Filter out players officially listed as Out in pre-game injury report
        if str(b_row.get("pre_game_status") or "").strip().lower() == "out":
            continue

        # In V2: p_mp_48_active is expected minutes if active; p_t_poss_48 is projected pace
        def _first(*vals):
            for v in vals:
                if v is not None and not pd.isna(v):
                    return float(v)
            return 0.0
        # A box row that says 0 minutes means 0 minutes; only fall back to /epm when the box has no value.
        p_mp = _first(b_row.get("p_mp_48_active"), b_row.get("p_mp_48"), b_row.get("p_mp")) if b_row is not None and len(b_row) \
            else _first(row.get("p_mp_48_active"), row.get("p_mp_48"))
        if p_mp < 1.0:   # simulation roster keeps fringe players; the trader decides who is tradeable
            continue

        pos_str = str(b_row.get("position") or "G-F")
        p_pace = float(b_row.get("p_t_poss_48") or row.get("p_t_poss_48", 99.5) or 99.5)

        profiles.append(DTPlayerSkillProfile(
            player_id=pid, player_name=str(row["player_name"]), team_id=team_id,
            position_code=pos_map.get(pos_str, 3), inches=int(b_row.get("inches", 78) or 78),
            weight=float(b_row.get("weight", 215.0) or 215.0), p_pace=p_pace,
            p_mp=p_mp, p_usg=float(row.get("p_usg", 0.20) or 0.20),
            p_fga_rim_100=float(row.get("p_fga_rim_100", 5.0) or 5.0),
            p_fga_mid_100=float(row.get("p_fga_mid_100", 3.0) or 3.0),
            p_fg3a_100=float(row.get("p_fg3a_100", 5.5) or 5.5),
            p_fta_100=float(row.get("p_fta_100", 4.0) or 4.0),
            p_ast_100=float(row.get("p_ast_100", 4.5) or 4.5),
            p_tov_100=float(row.get("p_tov_100", 2.5) or 2.5),
            p_orb_100=float(row.get("p_orb_100", 1.5) or 1.5),
            p_drb_100=float(row.get("p_drb_100", 5.5) or 5.5),
            p_stl_100=float(row.get("p_stl_100", 1.3) or 1.3),
            p_blk_100=float(row.get("p_blk_100", 0.8) or 0.8),
            p_fgpct_rim=float(row.get("p_fgpct_rim", 0.64) or 0.64),
            p_fgpct_mid=float(row.get("p_fgpct_mid", 0.41) or 0.41),
            p_fg3pct=float(row.get("p_fg3pct", 0.36) or 0.36),
            p_ftpct=float(row.get("p_ftpct", 0.78) or 0.78),
            oepm=float(row.get("oepm", 0.0) or 0.0), depm=float(row.get("depm", 0.0) or 0.0),
            epm=float(row.get("epm", 0.0) or 0.0),
            p_start=_pstart(b_row.get("p_pct_start") if b_row is not None and len(b_row) else row.get("p_pct_start"))
        ))

    return profiles, []