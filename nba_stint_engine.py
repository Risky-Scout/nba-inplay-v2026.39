import numpy as np
from rotation_hazard import RotationHazardModel


class NBAStintEngine:
    """Minute model for the in-play sim.

    get_remaining_minute_cap   -> hard ceiling on remaining minutes (fouled out = 0).
    expected_remaining_minutes -> best estimate of remaining minutes: projected on-court share,
                                  updated by in-game rotation evidence, foul-trouble windows
                                  (the player sits for the rest of the window, then plays normally),
                                  and late blowouts.
    """
    ROTATION_TEMPLATES = {
        "starter": {"max_minutes": 36.0},
        "bench_core": {"max_minutes": 20.0},
        "deep_reserve": {"max_minutes": 10.0},
    }
    FOUL_OUT = 6
    HAZARD_MODEL = RotationHazardModel()

    @staticmethod
    def elapsed_minutes(period: int, current_sec_rem: float) -> float:
        """Game minutes already played. current_sec_rem is time left in regulation (Q1-Q4) or in the current OT period."""
        T = max(0.0, current_sec_rem / 60.0)
        if period is None or period <= 4:
            return max(0.0, 48.0 - T)
        return 48.0 + 5.0 * max(0, period - 5) + max(0.0, 5.0 - T)

    @classmethod
    def substitution_probabilities(cls, **kwargs):
        """P(sub out within 15/30/60s), hierarchically shrunk in RotationHazardModel."""
        p = cls.HAZARD_MODEL.sub_hazard_1s(**kwargs)
        return {15: cls.HAZARD_MODEL.within(p, 15), 30: cls.HAZARD_MODEL.within(p, 30), 60: cls.HAZARD_MODEL.within(p, 60)}

    @classmethod
    def return_probabilities(cls, **kwargs):
        """P(return within 60/120/180s)."""
        p = cls.HAZARD_MODEL.return_hazard_1s(**kwargs)
        return {60: cls.HAZARD_MODEL.within(p, 60), 120: cls.HAZARD_MODEL.within(p, 120), 180: cls.HAZARD_MODEL.within(p, 180)}

    @classmethod
    def get_remaining_minute_cap(
        cls,
        is_starter: bool,
        current_minutes_played: float,
        current_sec_rem: float,
        fouls: int,
        score_diff: int,
        proj_minutes: float = 0.0,
        period: int = None,
        **_ignored
    ) -> float:
        if fouls >= cls.FOUL_OUT:
            return 0.0
        tier = "starter" if is_starter else "bench_core"
        # Template cap, but never below what D&T projects for this player (+15% overtime/close-game room)
        hard_cap = max(cls.ROTATION_TEMPLATES[tier]["max_minutes"], 1.15 * proj_minutes)
        if period is not None and period >= 5:
            # In OT a regulation ceiling must never stop a player taking OT minutes (v2026.24 could give a star 0).
            hard_cap = max(hard_cap, current_minutes_played + current_sec_rem / 60.0)
        rem = max(0.0, hard_cap - current_minutes_played)
        if current_sec_rem <= 480 and abs(score_diff) >= 18 and is_starter and (period is None or period <= 4):
            rem *= 0.35
        return float(max(0.0, min(rem, current_sec_rem / 60.0)))

    @classmethod
    def expected_remaining_minutes(
        cls,
        is_starter: bool,
        current_minutes_played: float,
        current_sec_rem: float,
        fouls: int,
        score_diff: int,
        proj_minutes: float = 0.0,
        period: int = None,
        completed_stints: int = 0,
        on_court: bool = False,
        evidence_k: float = 10.0
    ) -> float:
        T = current_sec_rem / 60.0
        if T <= 0 or fouls >= cls.FOUL_OUT:
            return 0.0
        share = float(np.clip(proj_minutes / 48.0, 0.0, 0.95))
        elapsed = cls.elapsed_minutes(period, current_sec_rem)

        # Rotation evidence: a player far below / above his projected pace so far is likely out of (or promoted in)
        # tonight's rotation. v2026.28: the weight is evidence-based, exp_so_far / (exp_so_far + k), where exp_so_far is
        # the minutes D&T expected him to have played by now (v2026.27: elapsed / 48, i.e. 0.125 at 6:00 regardless of
        # how much of his role had been observed). A player who has already completed a stint and is back on the floor
        # (or has completed two) has revealed his rotation: his ratio counts in full sooner.
        exp_so_far = share * elapsed
        if elapsed >= 6.0 and exp_so_far >= 2.0:
            if current_minutes_played <= 0.0 and elapsed >= 12.0:
                share *= 0.15
            else:
                r = float(np.clip(current_minutes_played / exp_so_far, 0.2, 1.4))
                ev = exp_so_far * (1.5 if (completed_stints >= 2 or (completed_stints >= 1 and on_court)) else 1.0)
                share *= r ** min(0.6, ev / (ev + max(1e-6, evidence_k)))

        E = share * T
        # Foul trouble: sits most of the current window, then returns to normal share.
        if current_sec_rem > 1440 and fouls >= 3:
            E -= share * ((current_sec_rem - 1440) / 60.0) * 0.85
        elif current_sec_rem > 720 and fouls >= 4:
            E -= share * ((current_sec_rem - 720) / 60.0) * 0.70
        elif current_sec_rem > 300 and fouls >= 5:
            E -= share * ((current_sec_rem - 300) / 60.0) * 0.35

        if current_sec_rem <= 480 and abs(score_diff) >= 18 and (period is None or period <= 4):
            E *= 0.30 if is_starter else 1.25
        return float(max(0.0, E))