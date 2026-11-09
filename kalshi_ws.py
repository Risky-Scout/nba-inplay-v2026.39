"""Kalshi WebSocket market state (v2026.36): a locally maintained L2 order book per subscribed ticker.

Protocol (docs.kalshi.com asyncapi, Trade API WS v2):
  connect   wss://external-api-ws.kalshi.com/trade-api/ws/v2 with the REST signing headers over
            timestamp + "GET" + "/trade-api/ws/v2"
  subscribe {"id": n, "cmd": "subscribe", "params": {"channels": ["orderbook_delta"], "market_tickers": [...]}}
  messages  orderbook_snapshot {sid, seq, msg: {market_ticker, yes_dollars_fp: [[price, qty]], no_dollars_fp: [...]}}
            orderbook_delta    {sid, seq, msg: {market_ticker, price_dollars, delta_fp, side}}
  Kalshi books hold BIDS on both sides: best YES ask = 100 - best NO bid.
Integrity: seq must increase by exactly 1 per subscription (sid). Any gap marks every book on that sid STALE and
forces a fresh subscription (new snapshot); a stale or old book is never returned to the trader, which then falls back
to REST. The trader still sends orders over REST (Kalshi's order entry is REST).
v2026.37: on_update(ticker) is called after every applied snapshot/delta (the trader's MARKET CLOCK cancels stale quotes
from it immediately); untrack() prunes subscriptions; a short top-of-book history per ticker gives the execution model its
velocity / depletion features (micro()); depth_at() reads any level for queue-ahead estimates."""
import json
import logging
import threading
import time
from collections import deque
from typing import Callable, Dict, Iterable, Optional, Tuple

logger = logging.getLogger("KalshiTrader")
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"   # production host per docs.kalshi.com
WS_PATH = "/trade-api/ws/v2"


def _cents(x) -> int:
    return int(round(float(x) * 100.0))


class KalshiWSBook:
    def __init__(self, headers_fn: Callable[[], Dict[str, str]], url: str = WS_URL, max_age_s: float = 2.0):
        self.headers_fn, self.url, self.max_age_s = headers_fn, url, max_age_s
        self.books: Dict[str, Dict[str, Dict[int, float]]] = {}
        self.updated: Dict[str, float] = {}
        self.sid_of: Dict[str, int] = {}
        self.seq: Dict[int, int] = {}
        self.stale: set = set()
        self.want: set = set()
        self.lock = threading.Lock()
        self.ws = None
        self.connected = False
        self.msg_id = 0
        self.gaps = 0
        self.msgs = 0
        self._stop = False
        self._thread = None
        self.on_update: Optional[Callable[[str], None]] = None
        self.hist: Dict[str, deque] = {}
        self.queue_tracks: Dict[str, dict] = {}   # order_id -> conservative queue-ahead estimate; never claimed exact

    # public -------------------------------------------------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="kalshi-ws", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def track(self, tickers: Iterable[str]):
        """Add tickers to the subscription set (existing books are kept)."""
        with self.lock:
            new = set(tickers) - self.want
            self.want |= set(tickers)
            pending = set(self.sid_of)
        new -= pending
        if new and self.connected:
            self._subscribe(sorted(new))

    def untrack(self, tickers: Iterable[str]):
        """Stop wanting these tickers: their local books are dropped (never served again) and a subscription whose every
        ticker is unwanted is unsubscribed. Shared subscriptions keep streaming; their unwanted tickers are ignored."""
        drop = set(tickers)
        with self.lock:
            self.want -= drop
            sids = {}
            for t, sid in self.sid_of.items():
                sids.setdefault(sid, set()).add(t)
            dead = [sid for sid, tks in sids.items() if tks and tks <= drop]
            for t in drop:
                self.books.pop(t, None)
                self.updated.pop(t, None)
                self.sid_of.pop(t, None)
                self.hist.pop(t, None)
                for oid in [o for o, q in self.queue_tracks.items() if q.get("ticker") == t]:
                    self.queue_tracks.pop(oid, None)
                self.stale.discard(t)
        for sid in dead:
            self._unsubscribe(sid)
        return dead

    def depth_at(self, ticker: str, side: str, price_c: int) -> Optional[float]:
        """Resting quantity at one YES-price level for buying `side` (YES bids; NO bids at 100 - YES ask)."""
        with self.lock:
            b = self.books.get(ticker)
            if b is None or ticker in self.stale:
                return None
            lvl = b["yes"] if side == "yes" else b["no"]
            return float(lvl.get(int(price_c), 0.0))

    def start_queue_estimate(self, order_id: str, ticker: str, side: str, price_c: int,
                             initial_depth: Optional[float] = None) -> dict:
        """Begin a conservative queue-ahead estimate for one resting order."""
        with self.lock:
            b = self.books.get(ticker)
            lvl = None if b is None else (b["yes"] if side == "yes" else b["no"])
            depth = float((lvl or {}).get(int(price_c), 0.0))
            q0 = max(0.0, float(depth if initial_depth is None else initial_depth))
            now = time.time()
            self.queue_tracks[str(order_id)] = dict(
                ticker=ticker, side=side, price=int(price_c), queue_ahead_initial=q0,
                queue_ahead_est=q0, last_depth=depth, submitted_ts=now, last_ts=now,
                depleted=0.0, seconds_at_price=0.0, price_level_resets=0,
                level_present=depth > 0.0)
            return dict(self.queue_tracks[str(order_id)])

    def queue_estimate(self, order_id: str) -> Optional[dict]:
        with self.lock:
            q = self.queue_tracks.get(str(order_id))
            if q is None:
                return None
            out = dict(q)
            elapsed = max(0.0, time.time() - float(out.get("submitted_ts", time.time())))
            out["seconds_at_price"] = elapsed
            out["queue_depletion_rate"] = float(out.get("depleted", 0.0)) / max(elapsed, 1e-6)
            return out

    def stop_queue_estimate(self, order_id: str) -> Optional[dict]:
        with self.lock:
            q = self.queue_tracks.pop(str(order_id), None)
            if q is None:
                return None
            elapsed = max(0.0, time.time() - float(q.get("submitted_ts", time.time())))
            q["seconds_at_price"] = elapsed
            q["queue_depletion_rate"] = float(q.get("depleted", 0.0)) / max(elapsed, 1e-6)
            return dict(q)

    def _update_queue_tracks_locked(self, ticker: str):
        """Caller holds self.lock. Decreases deplete estimated queue; later additions are behind us."""
        b = self.books.get(ticker)
        if b is None:
            return
        now = time.time()
        for q in [v for v in self.queue_tracks.values() if v.get("ticker") == ticker]:
            lvl = b["yes"] if q["side"] == "yes" else b["no"]
            depth = max(0.0, float(lvl.get(int(q["price"]), 0.0)))
            prev = max(0.0, float(q.get("last_depth", 0.0)))
            was = bool(q.get("level_present", prev > 0))
            if was and depth < prev:
                dec = prev - depth
                q["queue_ahead_est"] = max(0.0, float(q["queue_ahead_est"]) - dec)
                q["depleted"] = float(q.get("depleted", 0.0)) + dec
            elif (not was) and depth > 0.0:
                q["price_level_resets"] = int(q.get("price_level_resets", 0)) + 1
                q["queue_ahead_est"] = depth
            # depth increases while continuously present are conservatively assumed behind us.
            q["last_depth"] = depth
            q["level_present"] = depth > 0.0
            q["last_ts"] = now
            q["seconds_at_price"] = max(0.0, now - float(q.get("submitted_ts", now)))

    def micro(self, ticker: str) -> dict:
        """Velocity / depletion features over the last 250 ms and 1 s of top-of-book history (cents and contracts per
        second). Zeros with has_ws False when the history is too short or the book is not synchronized."""
        h = list(self.hist.get(ticker) or [])
        out = dict(mid_vel_250ms=0.0, mid_vel_1s=0.0, spread_vel=0.0, imb_vel=0.0, bid_depletion=0.0, ask_replenish=0.0,
                   has_ws=False)
        if len(h) < 2 or ticker in self.stale:
            return out
        t1, yb1, qb1, ya1, qa1 = h[-1]
        def at(dt):
            for rec in reversed(h):
                if t1 - rec[0] >= dt:
                    return rec
            return h[0]
        out["has_ws"] = True
        for name, dt in (("mid_vel_250ms", 0.25), ("mid_vel_1s", 1.0)):
            t0, yb0, qb0, ya0, qa0 = at(dt)
            span = max(1e-3, t1 - t0)
            out[name] = ((yb1 + ya1) - (yb0 + ya0)) / 2.0 / span
        t0, yb0, qb0, ya0, qa0 = at(1.0)
        span = max(1e-3, t1 - t0)
        imb = lambda qb, qa: (qb - qa) / max(qb + qa, 1e-9)
        out["spread_vel"] = ((ya1 - yb1) - (ya0 - yb0)) / span
        out["imb_vel"] = (imb(qb1, qa1) - imb(qb0, qa0)) / span
        out["bid_depletion"] = max(0.0, qb0 - qb1) / span if yb1 == yb0 else 0.0
        out["ask_replenish"] = max(0.0, qa1 - qa0) / span if ya1 == ya0 else 0.0
        return out

    def _record(self, tk: str):
        """Append the current top of book to the ticker's history (caller holds the lock)."""
        b = self.books.get(tk)
        if not b:
            return
        yes = {p: q for p, q in b["yes"].items() if q > 0}
        no = {p: q for p, q in b["no"].items() if q > 0}
        if not yes or not no:
            return
        yb, nb = max(yes), max(no)
        self.hist.setdefault(tk, deque(maxlen=64)).append((time.time(), yb, yes[yb], 100 - nb, no[nb]))

    def best(self, ticker: str) -> Optional[Tuple[int, float, int, float, float]]:
        """(yes_bid, bid_qty, yes_ask, ask_qty, age_s) from the local book, or None if not trustworthy right now."""
        with self.lock:
            b, t = self.books.get(ticker), self.updated.get(ticker)
            if b is None or t is None or ticker in self.stale:
                return None
            age = time.time() - t
            yes = {p: q for p, q in b["yes"].items() if q > 0}
            no = {p: q for p, q in b["no"].items() if q > 0}
        if not yes or not no:
            return None
        yb = max(yes)
        nb = max(no)
        return yb, yes[yb], 100 - nb, no[nb], age

    def fresh(self, ticker: str) -> Optional[Tuple[int, float, int, float]]:
        """best() only if the book is synchronized AND heard from within max_age_s (a quiet book is not proof of a
        live connection, so the age is measured against the last message on the connection as well)."""
        r = self.best(ticker)
        if r is None or not self.connected:
            return None
        return r[:4] if (r[4] <= self.max_age_s or time.time() - self._last_msg <= self.max_age_s) else None

    # internals ----------------------------------------------------------------------------------------------------
    _last_msg = 0.0

    def _unsubscribe(self, sid):
        if sid is None or self.ws is None:
            return
        self.msg_id += 1
        try:
            self.ws.send(json.dumps({"id": self.msg_id, "cmd": "unsubscribe", "params": {"sids": [sid]}}))
        except Exception as e:
            logger.warning(f"[WS] unsubscribe failed: {e}")

    def _subscribe(self, tickers):
        if not tickers or self.ws is None:
            return
        self.msg_id += 1
        try:
            self.ws.send(json.dumps({"id": self.msg_id, "cmd": "subscribe",
                                     "params": {"channels": ["orderbook_delta"], "market_tickers": list(tickers)}}))
        except Exception as e:
            logger.warning(f"[WS] subscribe failed: {e}")

    def _run(self):
        import websocket   # websocket-client
        backoff = 1.0
        while not self._stop:
            try:
                self.ws = websocket.create_connection(self.url, header=self.headers_fn(), timeout=10)
                self.connected, backoff = True, 1.0
                with self.lock:
                    self.stale |= set(self.books)       # every book must be re-snapshotted on a new connection
                    self.seq.clear()
                    self.sid_of.clear()
                    want = sorted(self.want)
                self._subscribe(want)
                logger.info(f"[WS] connected; {len(self.want)} market(s) subscribed")
                while not self._stop:
                    try:
                        raw = self.ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        raise ConnectionError("closed")
                    self._last_msg = time.time()
                    self._handle(json.loads(raw))
            except Exception as e:
                self.connected = False
                with self.lock:
                    self.stale |= set(self.books)
                if self._stop:
                    break
                logger.warning(f"[WS] disconnected ({e}); reconnecting in {backoff:.0f}s (REST book used meanwhile)")
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

    def _handle(self, d: dict):
        self.msgs += 1
        typ, sid, seq, m = d.get("type"), d.get("sid"), d.get("seq"), d.get("msg") or {}
        if typ == "error":
            logger.warning(f"[WS] error {m}")
            return
        if typ not in ("orderbook_snapshot", "orderbook_delta"):
            return
        tk = m.get("market_ticker")
        if not tk:
            return
        applied = False
        old_sid = None
        with self.lock:
            # Sequence bookkeeping FIRST, for every message on the sid (review fix): a delta for an untracked ticker or
            # from a superseded subscription still advances its sid's counter; skipping it made the next delta of a
            # wanted ticker on the same sid look like a gap.
            if sid is not None and seq is not None:
                last = self.seq.get(sid)
                if typ == "orderbook_delta" and last is not None and seq != last + 1:
                    self.gaps += 1
                    gap_tks = [t for t, s in self.sid_of.items() if s == sid]
                    self.stale |= set(gap_tks)
                    self.seq.pop(sid, None)
                    for t in gap_tks:
                        self.sid_of.pop(t, None)
                    resub, old_sid = gap_tks, sid
                else:
                    self.seq[sid] = seq
                    resub = None
            else:
                resub = None
            skip = tk not in self.want or (typ == "orderbook_delta" and self.sid_of.get(tk) is not None
                                           and sid != self.sid_of.get(tk))
            if resub is None and not skip:
                if typ == "orderbook_snapshot":
                    self.books[tk] = {"yes": {_cents(p): float(q) for p, q in (m.get("yes_dollars_fp") or [])},
                                      "no": {_cents(p): float(q) for p, q in (m.get("no_dollars_fp") or [])}}
                    self.sid_of[tk] = sid
                    self.stale.discard(tk)
                    self.updated[tk] = time.time()
                    self._record(tk)
                    self._update_queue_tracks_locked(tk)
                    applied = True
                elif tk in self.books and tk not in self.stale:
                    side = m.get("side")
                    if side in ("yes", "no"):
                        px = _cents(m.get("price_dollars", 0))
                        lvl = self.books[tk][side]
                        lvl[px] = lvl.get(px, 0.0) + float(m.get("delta_fp", 0) or 0)
                        if lvl[px] <= 1e-9:
                            lvl.pop(px, None)
                        self.updated[tk] = time.time()
                        self._record(tk)
                        self._update_queue_tracks_locked(tk)
                        applied = True
        if resub:
            logger.warning(f"[WS] sequence gap on sid {sid}: {len(resub)} book(s) marked stale; unsubscribing and re-snapshotting")
            self._unsubscribe(old_sid)
            self._subscribe(sorted(resub))
            cb = self.on_update
            for t in resub:                          # a now-stale book: the market clock re-checks quotes (REST fallback)
                if cb:
                    try:
                        cb(t)
                    except Exception:
                        pass
        elif applied and self.on_update is not None:
            try:
                self.on_update(tk)
            except Exception as e:
                logger.debug(f"[WS] on_update failed: {e}")


def ws_selftest(headers_fn, tickers, rest_top_fn, seconds: float = 15.0) -> bool:
    """Connect, subscribe, and compare the WS top of book with REST for each ticker. Prints a verdict."""
    wb = KalshiWSBook(headers_fn)
    wb.track(tickers)
    wb.start()
    t0 = time.time()
    while time.time() - t0 < seconds and not all(wb.best(t) for t in tickers):
        time.sleep(0.5)
    ok = True
    for t in tickers:
        w, r = wb.best(t), rest_top_fn(t)
        print(f"  {t}: WS {None if w is None else w[:4]} | REST {r}")
        if w is None or r is None or abs(w[0] - r[0]) > 1 or abs(w[2] - r[2]) > 1:
            ok = False
    print(f"  messages {wb.msgs}, sequence gaps {wb.gaps}, connected {wb.connected}")
    wb.stop()
    print("WS SELFTEST " + ("PASSED: --ws is safe to use" if ok else "FAILED: keep running without --ws"))
    return ok