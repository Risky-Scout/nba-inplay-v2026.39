"""Empirical tail risk with exact probability mass, including tied discrete losses."""
import numpy as np


def expected_shortfall(losses, q=0.95):
    """Mean of the worst (1-q) probability mass, interpolating its boundary atom."""
    x = np.sort(np.asarray(losses, dtype=float).ravel())[::-1]
    if not len(x) or not np.isfinite(x).all() or not 0 <= q < 1:
        raise ValueError("finite nonempty losses and q in [0, 1) required")
    mass = (1.0 - q) * len(x)
    whole = int(np.floor(mass))
    fraction = mass - whole
    return float((x[:whole].sum() + (fraction * x[whole] if whole < len(x) else 0)) / mass)
