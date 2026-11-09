"""
Correlated fractional-Kelly portfolio QP for Kalshi binary contracts (single source of truth).

Per-$ return of buying one contract at all-in cost c (price + fee, in $):  win -> (1-c)/c, lose -> -1.
    mu_i      = p_i / c_i - 1
    sigma_ij  = rho_ij * sd_i * sd_j,   sd_i = sqrt(p_i (1-p_i)) / c_i
rho_ij is measured directly from the Monte Carlo joint outcomes of the same simulated games
(exact under the model: same-player ladders, PTS vs PRA, teammates, opponents). Different games: 0.

Model-risk factors (v2026.29): the simulated correlations are exact only if the model is right. If it is biased, every
bet it likes is wrong in the same direction, in every game at once. Each bet's true win probability is modelled as
p_i - e, with shared errors e = global + stat + game (s.d. model_sd, stat_sd, game_sd, in probability units), adding
    Sigma_model_ij = (model_sd^2 + stat_sd^2 [same stat] + game_sd^2 [same game]) / (c_i c_j)
so twenty "diversified" bets that all express one modelling assumption are sized as the one bet they are.

Solve  max  mu'x - (gamma/2) x' Sigma x,  x = existing + new  (fractions of bankroll)
       s.t. session cap on total open risk, per-game cap, per-position cap, new >= 0.
gamma = 2 -> half Kelly.

v2026.39 quantity-aware fill objective. A resting quote returns F x R_contract with F ~ Bernoulli(phi) (phi = P(fill)):
    mu_i      = confidence_i x phi_i x conditional_edge_i / unit_cost_i
    Var_q(i)  = phi_i Var(R_i | fill) + phi_i (1 - phi_i) mu_c,i^2          (mu_c = conditional return per $)
    Cov_q(ij) = sqrt(phi_i phi_j) Cov(R_i, R_j)                             (conservative until joint fill data exist)
Confidence and P(fill) discount EXPECTED return only. Every hard cap (session, game, player, position, group caps) and
the integer contract count stay at FULL notional: an order that can fill is sized as if it will.
"""
import numpy as np
from scipy.optimize import minimize
from typing import Optional, List, Dict, Tuple


def kalshi_fee_cents(price_cents: float, is_maker: bool, taker_mult: float = 1.0, maker_mult: float = 1.0) -> float:
    """Per-contract fee in cents. Taker 0.07*P*(1-P); maker 0.0175*P*(1-P) only on series with maker fees.
    Kalshi rounds the whole order up to the cent, so +0.25c per contract is added as a rounding allowance."""
    p = price_cents / 100.0
    rate = (0.0175 * maker_mult) if is_maker else (0.07 * taker_mult)
    raw = rate * p * (1.0 - p) * 100.0
    return raw + 0.25 if raw > 0 else 0.0


class CorrelatedKellyQP:
    def __init__(self, session_cap: float = 0.60, game_cap: float = 0.15, single_cap: float = 0.05,
                 half_kelly: bool = True, fav_taper: bool = True, player_cap: float = 0.06, kelly_fraction: float = None,
                 model_sd: float = 0.0, stat_sd: float = 0.0, game_sd: float = 0.0, team_sd: float = 0.0,
                 player_sd: float = 0.0):
        self.session_cap = session_cap   # max total open risk (cost basis) as fraction of bankroll for the night
        self.team_sd, self.player_sd = team_sd, player_sd   # v2026.36 model-risk factors: team rotation, player role
        self.game_cap = game_cap
        self.single_cap = single_cap
        self.gamma = 1.0 / kelly_fraction if kelly_fraction else (2.0 if half_kelly else 1.0)   # fraction of full Kelly
        # Favorites are low-variance, so Kelly sizes them to the cap on small edges; a model error there
        # costs the full stake. Taper: full cap at <= 50c, 50% at 75c, 30% at 85c.
        self.fav_taper = fav_taper
        # One player's markets share model risk (role, minutes, matchup) that the simulated
        # correlations cannot see, so total exposure per player is capped separately.
        self.player_cap = player_cap
        self.model_sd, self.stat_sd, self.game_sd = model_sd, stat_sd, game_sd

    @staticmethod
    def build_covariance_matrix(candidates: List[dict], model_sd: float = 0.0, stat_sd: float = 0.0,
                                game_sd: float = 0.0, team_sd: float = 0.0, player_sd: float = 0.0) -> np.ndarray:
        n = len(candidates)
        p = np.array([c["win_prob"] for c in candidates])
        cost = np.array([(c["price_cents"] + c["fee_cents"]) / 100.0 for c in candidates])
        sd = np.sqrt(np.clip(p * (1.0 - p), 1e-4, None)) / cost
        rho = np.eye(n)
        games: Dict[str, List[int]] = {}
        for i, c in enumerate(candidates):
            games.setdefault(c["game_key"], []).append(i)
        for idxs in games.values():
            if len(idxs) < 2:
                continue
            Y = np.stack([candidates[i]["outcome"].astype(float) for i in idxs])
            Y -= Y.mean(axis=1, keepdims=True)
            nrm = np.sqrt((Y ** 2).sum(axis=1))
            ok = nrm > 0
            C = np.zeros((len(idxs), len(idxs)))
            if ok.any():
                Yn = Y[ok] / nrm[ok][:, None]
                C[np.ix_(ok, ok)] = Yn @ Yn.T
            np.fill_diagonal(C, 1.0)
            rho[np.ix_(idxs, idxs)] = C
        cov = rho * np.outer(sd, sd) + np.eye(n) * 1e-6
        if model_sd > 0 or stat_sd > 0 or game_sd > 0 or team_sd > 0 or player_sd > 0:
            # Low-rank MODEL-ERROR factors (cents of shared error per contract), added on top of the simulated outcome
            # correlation: global engine, stat, game, and (v2026.36) team rotation and player role. Outcomes in different
            # games are independent; our errors about them are not.
            st = np.array([str(c.get("stat", "")) for c in candidates])
            gk = np.array([str(c.get("game_key", "")) for c in candidates])
            tm = np.array([str(c.get("team_id", "")) for c in candidates])
            pl = np.array([str(c.get("player_name", "")) for c in candidates])
            same = lambda a: (a[:, None] == a[None, :]) & (a[:, None] != "")
            f = model_sd ** 2 + stat_sd ** 2 * same(st) + game_sd ** 2 * (gk[:, None] == gk[None, :]) \
                + team_sd ** 2 * same(tm) + player_sd ** 2 * same(pl)
            cov = cov + f / np.outer(cost, cost)
        return cov

    def solve_allocations(self, candidates: List[dict], bankroll: float, max_bet_usd: float,
                          positions: Dict[Tuple[str, str], dict], held: Optional[List[dict]] = None,
                          group_caps: Optional[List[dict]] = None) -> List[dict]:
        """`positions` must already include reserved risk of resting/pending orders. `held` = positions already owned in
        games being priced, with their simulated outcome vectors: they enter the covariance as FIXED exposures h, so the
        objective carries the cross term gamma * f' Sigma_NH h (a new bet correlated with what we hold costs more).
        group_caps: extra budgets [{"members": [candidate indices], "room": fraction of bankroll still free}] (v2026.37:
        the PILOT session budget and per-player pilot budgets), enforced in the continuous and the integer stage.
        Candidate fields: confidence (default 1), effective_fill_fraction (preferred; fallback fill_prob/default 1), max_contracts."""
        self.last_stats = {"positive": 0, "below_one": 0, "rounded_up": 0}
        if not candidates or bankroll <= 0:
            return []
        n = len(candidates)
        p = np.array([c["win_prob"] for c in candidates])
        unit_c = np.array([c["price_cents"] + c["fee_cents"] for c in candidates], dtype=float)
        cost = unit_c / 100.0
        # Expected return per $ risked. A candidate may carry an explicit expected_edge_c (cents per contract we expect to
        # KEEP: for a maker, directional alpha + conditional execution alpha - fee - uncertainty). Otherwise the
        # immediate-execution value p/c - 1. win_prob still drives the outcome covariance.
        edge_c = np.array([float(c.get("expected_edge_c", 100.0 * c["win_prob"] - unit_c[i])) for i, c in enumerate(candidates)])
        conf = np.array([float(np.clip(c.get("confidence", 1.0) if c.get("confidence") is not None else 1.0, 0.0, 1.0)) for c in candidates])
        phi = np.array([float(np.clip(c.get("effective_fill_fraction", c.get("fill_prob", 1.0)) if c.get("effective_fill_fraction", c.get("fill_prob", 1.0)) is not None else 1.0, 1e-4, 1.0)) for c in candidates])
        mu_c = conf * edge_c / np.maximum(unit_c, 1e-9)        # conditional on a fill, confidence-discounted
        mu = phi * mu_c                                        # expected return of the QUOTE
        fs = dict(model_sd=self.model_sd, stat_sd=self.stat_sd, game_sd=self.game_sd, team_sd=self.team_sd,
                  player_sd=self.player_sd)
        sigma_c = self.build_covariance_matrix(candidates, **fs)
        sq = np.sqrt(phi)
        sigma = sigma_c * np.outer(sq, sq)                     # sqrt(phi_i phi_j) Cov; diagonal = phi Var(R | fill)
        sigma[np.diag_indices(n)] += phi * (1.0 - phi) * mu_c ** 2
        cross = np.zeros(n)
        if held:
            keys = {(c["ticker"], c["side"]) for c in candidates}
            H = [h for h in held if (h["ticker"], h["side"]) not in keys and h.get("exposure_usd", 0) > 0]
            if H:
                full = self.build_covariance_matrix(candidates + H, **fs)
                hv = np.array([h["exposure_usd"] / bankroll for h in H])
                cross = sq * (full[:n, n:] @ hv)               # held exposure is already filled (phi = 1)

        e = np.array([max(0.0, positions.get((c["ticker"], c["side"]), {}).get("cost_usd", 0.0)) / bankroll for c in candidates])
        tot_open = sum(max(0.0, v.get("cost_usd", 0.0)) for v in positions.values()) / bankroll
        game_open: Dict[str, float] = {}
        for v in positions.values():
            game_open[v.get("game_key", "")] = game_open.get(v.get("game_key", ""), 0.0) + max(0.0, v.get("cost_usd", 0.0)) / bankroll

        room_total = max(0.0, self.session_cap - tot_open)
        taper = np.minimum(1.0, 2.0 * (1.0 - cost)) if self.fav_taper else np.ones(n)
        upper = np.maximum(0.0, self.single_cap * taper - e)
        # per-candidate contract ceiling (pilot orders, exploration arms, probes: 1) enforced INSIDE the optimizer, so
        # the budget a capped candidate cannot use is re-allocated instead of being cut away afterwards (review fix)
        maxc = np.array([float(c["max_contracts"]) if c.get("max_contracts") else np.inf for c in candidates])
        upper = np.minimum(upper, np.where(np.isfinite(maxc), maxc * cost / bankroll, np.inf))
        games: Dict[str, List[int]] = {}
        for i, c in enumerate(candidates):
            games.setdefault(c["game_key"], []).append(i)
        cons = [{"type": "ineq", "fun": lambda f: room_total - f.sum(), "jac": lambda f: -np.ones(n)}]
        for gk, idxs in games.items():
            room = max(0.0, self.game_cap - game_open.get(gk, 0.0))
            if room <= 1e-9:
                upper[idxs] = 0.0
                continue
            mask = np.zeros(n); mask[idxs] = 1.0
            cons.append({"type": "ineq", "fun": lambda f, m=mask, r=room: r - m @ f, "jac": lambda f, m=mask: -m})
        player_open: Dict[str, float] = {}
        for v in positions.values():
            player_open[v.get("player", "")] = player_open.get(v.get("player", ""), 0.0) + max(0.0, v.get("cost_usd", 0.0)) / bankroll
        players: Dict[str, List[int]] = {}
        for i, c in enumerate(candidates):
            players.setdefault(c.get("player_name", str(i)), []).append(i)
        for pk, idxs in players.items():
            room = max(0.0, self.player_cap - player_open.get(pk, 0.0))
            if room <= 1e-9:
                upper[idxs] = 0.0
                continue
            mask = np.zeros(n); mask[idxs] = 1.0
            cons.append({"type": "ineq", "fun": lambda f, m=mask, r=room: r - m @ f, "jac": lambda f, m=mask: -m})
        groups = []
        for gc in (group_caps or []):
            idxs = [int(i) for i in gc.get("members", []) if 0 <= int(i) < n]
            if not idxs:
                continue
            room = max(0.0, float(gc.get("room", 0.0)))
            groups.append((idxs, room))
            if room <= 1e-9:
                upper[idxs] = 0.0
                continue
            mask = np.zeros(n); mask[idxs] = 1.0
            cons.append({"type": "ineq", "fun": lambda f, m=mask, r=room: r - m @ f, "jac": lambda f, m=mask: -m})
        if room_total <= 1e-9 or not upper.any():
            return []

        obj = lambda f: 0.5 * self.gamma * (f + e) @ sigma @ (f + e) + self.gamma * f @ cross - mu @ (f + e)
        jac = lambda f: self.gamma * sigma @ (f + e) + self.gamma * cross - mu
        res = minimize(obj, np.zeros(n), jac=jac, method="SLSQP", bounds=list(zip(np.zeros(n), upper)),
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 300})
        if not res.success:
            return []
        f_opt = np.clip(res.x, 0.0, upper)

        # Discrete contracts. Kalshi trades whole contracts; flooring the continuous Kelly answer drops every stake below
        # one contract. Instead: floor, then integer marginal ascent on the SAME objective (EV minus the risk penalty,
        # covariance with everything allocated and held included) under every cap; see the move loop below.
        n_contracts = np.zeros(n, dtype=int)
        for i, c in enumerate(candidates):
            n_contracts[i] = int(min(min(f_opt[i] * bankroll, max_bet_usd) * 100.0 // unit_c[i], maxc[i]))
        x1 = unit_c / 100.0 / bankroll                       # bankroll fraction of one contract
        stats = {"positive": int((f_opt > 1e-9).sum()), "below_one": int(((f_opt > 1e-9) & (n_contracts < 1)).sum()), "rounded_up": 0}
        stake_wanted = [float(f_opt[i] * bankroll) for i in range(n) if f_opt[i] > 1e-9]

        def f_of(k):
            return k * x1

        def caps_ok(k):
            if np.any(k > maxc):
                return False
            fk = f_of(k)
            if fk.sum() > room_total + 1e-12 or np.any(fk > upper + 1e-12) or np.any(k * unit_c / 100.0 > max_bet_usd + 1e-9):
                return False
            return all(fk[[j for j in range(n) if candidates[j]["game_key"] == gk]].sum() <= max(0.0, self.game_cap - game_open.get(gk, 0.0)) + 1e-12
                       for gk in games) and all(
                fk[idxs].sum() <= max(0.0, self.player_cap - player_open.get(pk, 0.0)) + 1e-12 for pk, idxs in players.items()) and all(
                fk[idxs].sum() <= room + 1e-12 for idxs, room in groups)

        # v2026.36: full integer marginal ascent. Every candidate may gain a contract (0->1, 1->2, ...), not only the ones
        # that floored to zero; then single-contract swap moves (remove one from i, add one to j) escape greedy local
        # optima. Each move must raise the objective and keep every cap. Candidate sets are small (< 30): exhaustive.
        def try_moves():
            base = obj(f_of(n_contracts))
            best, best_gain = None, 1e-12
            for i in range(n):
                if mu[i] <= 0:
                    continue
                k = n_contracts.copy(); k[i] += 1
                if caps_ok(k):
                    g = base - obj(f_of(k))
                    if g > best_gain:
                        best, best_gain = ("add", i, None), g
            for i in range(n):
                if n_contracts[i] < 1:
                    continue
                for j in range(n):
                    if j == i or mu[j] <= 0:
                        continue
                    k = n_contracts.copy(); k[i] -= 1; k[j] += 1
                    if caps_ok(k):
                        g = base - obj(f_of(k))
                        if g > best_gain:
                            best, best_gain = ("swap", i, j), g
            return best

        for _ in range(200):
            mv = try_moves()
            if mv is None:
                break
            kind, i, j = mv
            if kind == "add":
                if n_contracts[i] == 0:
                    stats["rounded_up"] += 1
                n_contracts[i] += 1
            else:
                n_contracts[i] -= 1
                n_contracts[j] += 1
                stats["swaps"] = stats.get("swaps", 0) + 1
        stats["added"] = int(n_contracts.sum())
        stats["median_wanted_usd"] = float(np.median(stake_wanted)) if stake_wanted else 0.0
        stats["median_contract_usd"] = float(np.median(unit_c / 100.0)) if n else 0.0
        self.last_stats = stats

        out = []
        for i, c in enumerate(candidates):
            contracts = int(n_contracts[i])
            if contracts < 1:
                continue
            item = dict(c)
            item["joint_kelly_fraction"] = float(f_opt[i])
            item["contracts"] = contracts
            item["risk_usd"] = round(contracts * (c["price_cents"] + c["fee_cents"]) / 100.0, 2)
            item["ev_usd"] = round(contracts * phi[i] * conf[i] * edge_c[i] / 100.0, 2)   # expected, fill- and confidence-weighted
            item["ev_if_filled_usd"] = round(contracts * edge_c[i] / 100.0, 2)
            out.append(item)
        return sorted(out, key=lambda x: x["ev_usd"], reverse=True)