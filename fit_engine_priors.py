"""Nested chronological prior-selection scaffold; diagnostic scores never grant authority."""
from datetime import datetime, timezone
import numpy as np
from nba_markov_engine import ENGINE_VERSION

CANDIDATES = {"SHOT_MIX_KAPPA": [6, 10, 15, 20, 30, 45], "ASSIST_ALPHA": [6, 10, 15, 20, 30, 45],
              "PACE_TAU0": [20, 30, 45, 60, 90, 120], "MINUTES_EVIDENCE_K": [5, 7.5, 10, 15, 20, 30],
              "FTA_ALPHA": [8, 12, 20, 30, 45], "TOV_ALPHA": [8, 12, 20, 30, 45],
              "ORB_ALPHA": [8, 12, 20, 30, 45], "DRB_ALPHA": [8, 12, 20, 30, 45]}


def nested_select(score_fn, dates):
    """score_fn(name, value, training_dates, evaluation_date) yields finite losses.

    Each outer fold freezes the winner of its preceding inner validation. The
    returned parameters are the last such frozen choice, never the parameter
    with the lowest outer-test loss. An independent final holdout is still needed.
    """
    dates = list(dates)
    if dates != sorted(set(dates)):
        raise ValueError("dates must be unique and chronological")
    chosen, folds = {}, []
    def score(name, value, train, day):
        x = np.asarray(list(score_fn(name, value, train, day)), float)
        if not x.size or not np.isfinite(x).all():
            raise ValueError("nonempty finite validation losses required")
        return float(x.mean())
    for name, values in CANDIDATES.items():
        for i in range(2, len(dates)):
            training, validation, test = dates[:i-1], dates[i-1], dates[i]
            _, value = min((score(name, v, training, validation), v) for v in values)
            chosen[name] = value
            folds.append(dict(parameter=name, value=value, validation_date=validation, test_date=test,
                              test_loss=score(name, value, dates[:i], test)))
    return dict(engine_version=ENGINE_VERSION, validated=False, params=chosen, outer_folds=folds,
                selected_through=dates[-2] if len(dates) >= 3 else None,
                final_holdout_date=dates[-1] if len(dates) >= 3 else None,
                reason="Research selection only; independent as-of holdout evidence and review required",
                fitted=datetime.now(timezone.utc).isoformat(timespec="seconds"))
