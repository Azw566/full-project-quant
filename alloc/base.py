"""
alloc/base.py — Pure portfolio math extracted from portfolio-risk-quant/utils.py.

All functions are pure (no I/O, no randomness) and operate on numpy arrays.
"""

import numpy as np
import pandas as pd


# ── Return helpers ─────────────────────────────────────────────────────────────

def log_returns(prices) -> pd.DataFrame:
    """Daily log returns: r_t = ln(P_t / P_{t-1}). Drops the first NaN row."""
    df = prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)
    return np.log(df / df.shift(1)).dropna()


def sample_moments(returns, annualize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate mu (N,) and Sigma (N, N) from a returns DataFrame.

    Returns (mu, Sigma) as plain ndarrays. Annualises by 252 when annualize=True.
    """
    scale = 252 if annualize else 1
    df = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(returns)
    mu = df.mean().values * scale
    Sigma = df.cov().values * scale
    return mu, Sigma


# ── Portfolio math ─────────────────────────────────────────────────────────────

def port_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Expected return: w^T μ."""
    return float(np.dot(w, mu))


def port_vol(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Portfolio volatility: sqrt(w^T Σ w)."""
    v = float(w @ Sigma @ w)
    return float(np.sqrt(max(v, 0.0)))


def port_sharpe(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> float:
    """Sharpe ratio: (E[Rp] - rf) / σ_p."""
    return (port_return(w, mu) - rf) / port_vol(w, Sigma)


def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Euler risk contributions: RC_i = w_i * (Σw)_i / σ_p.  Sum = σ_p."""
    Sigma_w = Sigma @ w
    sigma_p = float(np.sqrt(max(w @ Sigma_w, 0.0)))
    return w * Sigma_w / sigma_p


# ── Standard constructions ─────────────────────────────────────────────────────

def equal_weight(n: int) -> np.ndarray:
    """Equal-weight portfolio."""
    return np.ones(n) / n


def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio (closed-form, unconstrained)."""
    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(Sigma.shape[0])
    w = Sigma_inv @ ones
    return w / (ones @ w)


def tangency_weights(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Tangency (max-Sharpe) portfolio weights."""
    Sigma_inv = np.linalg.inv(Sigma)
    excess = mu - rf
    ones = np.ones(len(mu))
    w = Sigma_inv @ excess
    return w / (ones @ w)
