"""
risk/var_cvar.py — Value-at-Risk, Expected Shortfall, and Min-CVaR optimisation.

Extracted from portfolio-risk-quant/module_3_risk_measures.py.

PUBLIC INTERFACE
────────────────
    var_parametric(mu, sigma, confidence)   → float
    var_historical(losses, confidence)       → float
    var_mc(mu, sigma, confidence, ...)       → float
    es_parametric(mu, sigma, confidence)    → float
    es_historical(losses, confidence)        → float
    min_cvar_weights(returns, confidence)    → np.ndarray | None

All *_parametric / *_mc functions operate on daily mu and sigma.
Losses are defined as positive values (loss = -return).
"""

import numpy as np
from scipy import stats as _stats
import cvxpy as cp


# ── Value-at-Risk ──────────────────────────────────────────────────────────────

def var_parametric(mu_daily: float, sigma_daily: float, confidence: float = 0.95) -> float:
    """Parametric Gaussian VaR: -mu + sigma * Phi^{-1}(c)."""
    return float(-mu_daily + sigma_daily * _stats.norm.ppf(confidence))


def var_historical(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Historical simulation VaR: empirical quantile of the loss distribution."""
    return float(np.quantile(losses, confidence))


def var_mc(
    mu_daily: float,
    sigma_daily: float,
    confidence: float = 0.95,
    n_sims: int = 200_000,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR via Gaussian simulation."""
    rng = np.random.default_rng(seed)
    mc_losses = -rng.normal(mu_daily, sigma_daily, n_sims)
    return float(np.quantile(mc_losses, confidence))


# ── Expected Shortfall (CVaR) ──────────────────────────────────────────────────

def es_parametric(mu_daily: float, sigma_daily: float, confidence: float = 0.95) -> float:
    """
    Parametric Gaussian ES: -mu + sigma * phi(Phi^{-1}(c)) / (1 - c).

    ES is coherent (satisfies subadditivity), unlike VaR.
    """
    z_c = float(_stats.norm.ppf(confidence))
    return float(-mu_daily + sigma_daily * _stats.norm.pdf(z_c) / (1.0 - confidence))


def es_historical(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall: mean loss in the tail beyond VaR."""
    var_h = float(np.quantile(losses, confidence))
    tail = losses[losses >= var_h]
    return float(np.mean(tail))


# ── Min-CVaR portfolio (Rockafellar-Uryasev LP) ────────────────────────────────

def min_cvar_weights(returns, confidence: float = 0.95) -> np.ndarray | None:
    """
    Minimum-CVaR portfolio via the Rockafellar-Uryasev linearisation:

        min_{w, zeta}  zeta + 1/((1-c)T) * sum_t max(L_t(w) - zeta, 0)
        s.t.           w >= 0, sum(w) = 1

    where L_t(w) = -r_t^T w are the scenario losses.

    Returns
    -------
    (N,) weight vector, or None if the solver does not converge.
    """
    R = returns.values if hasattr(returns, "values") else np.asarray(returns)
    T, N = R.shape

    w = cp.Variable(N, nonneg=True)
    zeta = cp.Variable()
    u = cp.Variable(T, nonneg=True)

    losses = -R @ w
    obj = cp.Minimize(zeta + (1.0 / ((1.0 - confidence) * T)) * cp.sum(u))
    cons = [u >= losses - zeta, cp.sum(w) == 1]
    cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False)

    return w.value
