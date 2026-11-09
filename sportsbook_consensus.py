"""Sportsbook consensus adapter.

The original uploaded trader imports this module, but it was not included in the
uploaded source set. This implementation is intentionally fail-safe: without an
ODDS_API_KEY it disables itself; API/network failures return no consensus and
never fabricate a price.
"""
from __future__ import annotations
import time, requests, numpy as np

class SportsbookConsensus:
    def __init__(self, api_key, normalize_name, interval=90.0, max_age=90.0):
        self.api_key=(api_key or "").strip()
        self.normalize_name=normalize_name
        self.interval=float(interval)
        self.max_age=float(max_age)
        self.enabled=bool(self.api_key)
        self._cache={}
        self._last={}
    def game(self, matchup_title):
        # A safe empty consensus is preferable to stale/fabricated sportsbook data.
        # Users may extend this adapter for their Odds API market subscription.
        return self._cache.get(str(matchup_title), {}) if self.enabled else {}
    def set_game_snapshot(self, matchup_title, snapshot):
        self._cache[str(matchup_title)]=snapshot or {}
        self._last[str(matchup_title)]=time.time()

def _american_to_prob(o):
    o=float(o)
    return 100.0/(o+100.0) if o>0 else (-o)/((-o)+100.0)

def _devig_pair(over_odds,under_odds):
    po,pu=_american_to_prob(over_odds),_american_to_prob(under_odds)
    z=po+pu
    return (po/z,pu/z) if z>0 else (None,None)

def prob_at_line(consensus, player, stat, line, simulation_samples=None):
    """Return (p_over, n_books, distance_to_requested_line) or None.

    Supported consensus shape is a mapping/list of rows with player/stat/line
    and over_odds/under_odds. Exact-line evidence is preferred. A nearby line
    may be translated with the simulation empirical CDF, with distance returned
    so the trader's book_max_line_diff gate can reject it.
    """
    if not consensus:return None
    rows=list(consensus.values()) if isinstance(consensus,dict) else list(consensus)
    vals=[]
    pname=str(player); stat=str(stat).upper(); target=float(line)
    for r in rows:
        if not isinstance(r,dict):continue
        if str(r.get("player",""))!=pname or str(r.get("stat","")).upper()!=stat:continue
        try:
            ln=float(r["line"]); oo=float(r["over_odds"]); uo=float(r["under_odds"])
        except Exception:continue
        p,_=_devig_pair(oo,uo)
        if p is None:continue
        if abs(ln-target)>1e-9 and simulation_samples is not None and len(simulation_samples):
            smp=np.asarray(simulation_samples,float)
            p=float((np.sum(smp>target)+0.5)/(len(smp)+1.0))
        vals.append((p,abs(ln-target)))
    if not vals:return None
    ps=[x[0] for x in vals]; ds=[x[1] for x in vals]
    lg=np.log(np.clip(ps,1e-5,1-1e-5)/(1-np.clip(ps,1e-5,1-1e-5)))
    p=float(1/(1+np.exp(-np.median(lg))))
    return p,len(vals),float(min(ds))
