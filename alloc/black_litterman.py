"""
alloc/black_litterman.py — Black-Litterman posterior expected returns.

Extracted from portfolio-risk-quant/module_4_modern_allocation.py.
"""

import numpy as np


def reverse_optimize(
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """
    Implied equilibrium returns from the reverse-optimisation step.

    Pi = lambda * Sigma @ w_mkt

    Parameters
    ----------
    Sigma          : (N, N) covariance matrix
    w_mkt          : (N,) market-cap weights (or equal-weight proxy)
    risk_aversion  : lambda — estimated from market Sharpe and vol
    """
    return risk_aversion * Sigma @ w_mkt


def posterior_mean(
    Sigma: np.ndarray,
    Pi: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Black-Litterman posterior expected returns.

    Bayesian update: prior = implied returns Pi, likelihood = views (P, q, Omega).

    Parameters
    ----------
    Sigma  : (N, N) covariance matrix
    Pi     : (N,) implied equilibrium returns from reverse_optimize()
    P      : (K, N) pick matrix — each row encodes one view
    q      : (K,) view expected returns
    Omega  : (K, K) view uncertainty covariance (diagonal in practice)
    tau    : prior uncertainty scale (typically 0.01–0.10)

    Returns
    -------
    mu_bar : (N,) posterior expected returns

    Validation: when Omega -> inf (no-view limit), mu_bar -> Pi.
    """
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)
    M = tau_Sigma_inv + P.T @ Omega_inv @ P
    return np.linalg.inv(M) @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ q)
