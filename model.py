"""Tone curves, print model and metrics for paper 2 (dark-tone prepress)."""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure

R_SOLID = 10.0 ** -1.8
DOT_GAIN = 0.15
YN_N = 1.7
CELL = 8
SIGMA = 3.0
SHADOW = 0.1
X8 = np.linspace(0.0, 1.0, 256)


def exp_curve(x, lam):
    """f(x; lam) = (e^(lam x) - 1) / (e^lam - 1); lam < 0 brightens, lam > 0 darkens."""
    return x if abs(lam) < 1e-9 else np.expm1(lam * x) / np.expm1(lam)


def gamma_curve(x, g):
    return np.power(x, g)


def log_curve(x, k):
    return np.log1p(k * x) / np.log1p(k)


def rational_curve(x, k):
    return x * (1.0 + k) / (x + k)


FAMILIES = {
    "exp": (exp_curve, -30.0, -1e-3),
    "gamma": (gamma_curve, 1.0, 0.02),
    "log": (log_curve, 1e-3, 5000.0),
    "rational": (rational_curve, 50.0, 1e-4),
}


def match_mean(fn, lo, hi, hist, target, iters=80):
    """Parameter in [lo, hi] giving the target mean output for an 8-bit histogram."""
    w = hist / hist.sum()

    def mean(p):
        return float((fn(X8, p) * w).sum())

    sign = 1.0 if mean(hi) > mean(lo) else -1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sign * (mean(mid) - target) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def to8(y):
    return np.clip(np.rint(np.asarray(y) * 255), 0, 255).astype(np.uint8)


def banding(y8_in, y8_out, shadow=SHADOW):
    """Occupied output levels and the largest gap among them, for shadow inputs."""
    m = y8_in <= int(shadow * 255)
    lv = np.unique(y8_out[m])
    return len(lv), int(np.diff(lv).max()) if len(lv) > 1 else 0


def _threshold_matrix(n=CELL):
    """Clustered-dot threshold matrix: dots grow from the cell centre outwards."""
    c = (n - 1) / 2.0
    i, j = np.mgrid[0:n, 0:n]
    d = np.hypot(i - c, j - c) + 1e-3 * np.arctan2(i - c, j - c)
    order = np.argsort(np.argsort(d.ravel())).reshape(n, n)
    return (order + 0.5) / (n * n)


def halftone(k):
    """AM halftone: 1 where ink is placed, for ink coverage k in [0, 1]."""
    t = _threshold_matrix()
    h, w = k.shape
    tt = np.tile(t, (h // CELL + 1, w // CELL + 1))[:h, :w]
    return (k > tt).astype(float)


def lstar(r):
    r = np.clip(r, 1e-6, 1.0)
    return np.where(r > 0.008856, 116.0 * np.cbrt(r) - 16.0, 903.3 * r)


def print_l(y):
    """Printed and intended L* for a gamma-encoded tone image y in [0, 1]."""
    r_t = np.maximum(y ** 2.2, R_SOLID)
    k = (1.0 - r_t) / (1.0 - R_SOLID)
    cov = gaussian_filter(halftone(k), SIGMA)
    cov = np.clip(cov + 4.0 * DOT_GAIN * cov * (1.0 - cov), 0.0, 1.0)
    r = (1.0 - cov * (1.0 - R_SOLID ** (1.0 / YN_N))) ** YN_N
    return lstar(r), lstar(r_t)


def shadow_survival(lut):
    """Fraction of the 8-bit shadow input levels that stay >= 1 L* apart after printing."""
    reps, rows = CELL * 2, CELL * 8
    ramp = np.repeat(np.arange(256), reps)[None, :].repeat(rows, 0)
    lp, _ = print_l(lut[ramp])
    prof = lp.reshape(rows, 256, reps).mean(axis=(0, 2))
    n = int(SHADOW * 255)
    kept, last = 1, prof[0]
    for v in prof[1:n + 1]:
        if abs(v - last) >= 1.0:
            kept += 1
            last = v
    return kept / (n + 1)


def clahe(y):
    return exposure.equalize_adapthist(y, clip_limit=0.02)


def k_limit(lam):
    """Limit of the relative raster contrast K = (S - S0)/(S + S0) as L0 -> 1."""
    lam = np.asarray(lam, dtype=float)
    r = lam * np.exp(lam) / np.expm1(lam)
    return (r - 1.0) / (r + 1.0)
