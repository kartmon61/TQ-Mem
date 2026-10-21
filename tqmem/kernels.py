"""Temporal kernels on absolute day coordinates, width table, weight table, candidate windows."""
from __future__ import annotations
import math
import numpy as np

# width grade -> (sigma_days for anchored, tau_days for range, halflife_days for recent)
WIDTH_TABLE = {"exact": (1.0, 1.0, 3.0), "narrow": (4.0, 3.0, 7.0), "medium": (10.0, 7.0, 14.0), "wide": (30.0, 14.0, 30.0)}
# mode -> (alpha, beta)
WEIGHT_TABLE = {"none": (1.0, 0.0), "recent": (0.7, 0.3), "range": (0.5, 0.5), "anchored": (0.4, 0.6)}
THETA_SEM = 0.5      # relative: fraction of the best semantic score among candidates (theta_mode="rel")
# multiplicative variant exponents (E3 V4)
GAMMA_TABLE = {"none": 0.0, "recent": 0.5, "range": 1.0, "anchored": 1.5}

def scaled_width(width: str, scale: float = 1.0):
    s, t, h = WIDTH_TABLE[width]
    return s * scale, t * scale, h * scale

def _sig(x): return 1.0 / (1.0 + np.exp(-x))

def kernel(mode: str, t: np.ndarray, now: float, center: float | None, span: tuple[float, float] | None,
           sigma: float, tau: float, halflife: float) -> np.ndarray:
    """t: array of memory day coordinates. Returns K in [0,1]."""
    if mode == "none":
        return np.zeros_like(t, dtype=np.float64)
    if mode == "recent":
        lam = halflife / math.log(2)
        return np.exp(-np.clip(now - t, 0, None) / lam)
    if mode == "anchored":
        return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    if mode == "range":
        l, r = span
        return _sig((t - l) / tau) * _sig((r - t) / tau)
    raise ValueError(mode)

def candidate_window(mode: str, now: float, center, span, sigma, tau, halflife, expand: int = 0):
    """Return [lo, hi] day window. expand=0,1,2 widens per the design table."""
    if mode == "none":
        return None
    if mode == "recent":
        lam = halflife / math.log(2); m = [5, 8, 12][expand]
        return (now - m * lam, now)
    if mode == "anchored":
        m = [3, 4, 6][expand]
        return (center - m * sigma, center + m * sigma)
    if mode == "range":
        m = [3, 5, 8][expand]; l, r = span
        return (l - m * tau, r + m * tau)
    raise ValueError(mode)

def hard_window(mode: str, now: float, center, span, sigma, tau, halflife):
    """TSM-style hard filter window (no soft boundary)."""
    if mode == "none":
        return None
    if mode == "recent":
        return (now - halflife, now)
    if mode == "anchored":
        return (center - sigma, center + sigma)
    if mode == "range":
        return tuple(span)
    raise ValueError(mode)
