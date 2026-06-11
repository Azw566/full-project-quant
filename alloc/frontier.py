"""
alloc/frontier.py — Closed-form Markowitz efficient frontier.

Extracted from portfolio-risk-quant/module_1_mean_variance.py (Sessions 1–4).

PUBLIC INTERFACE
────────────────
    frontier_scalars(mu, Sigma)                → (A, B, C, D)
    frontier_weights(mu, Sigma, target_return) → np.ndarray
    estimation_error_experiment(...)           → dict
"""

import numpy as np


def frontier_scalars(
    mu: np.ndarray, Sigma: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Four scalars (A, B, C, D) that parametrise the Markowitz frontier parabola.

        A = 1^T Σ^{-1} 1
        B = 1^T Σ^{-1} μ
        C = μ^T Σ^{-1} μ
        D = AC - B²

    The frontier parabola is σ²(μ*) = (A μ*² − 2B μ* + C) / D.
    GMV point: μ_gmv = B/A,  σ²_gmv = 1/A  (no dependence on μ).
    """
    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(len(mu))
    A = float(ones @ Sigma_inv @ ones)
    B = float(ones @ Sigma_inv @ mu)
    C = float(mu @ Sigma_inv @ mu)
    D = A * C - B ** 2
    return A, B, C, D


def frontier_weights(
    mu: np.ndarray, Sigma: np.ndarray, target_return: float
) -> np.ndarray:
    """
    Closed-form unconstrained efficient frontier weights for a target return.

    Lagrange multipliers solution: w = Σ^{-1}(λμ + γ1)
    where λ = (Aμ* − B)/D  and  γ = (C − Bμ*)/D.

    May include short positions. Use alloc.mean_variance.mvp_weights for
    constrained (long-only) versions.
    """
    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(len(mu))
    A, B, C, D = frontier_scalars(mu, Sigma)
    lam = (A * target_return - B) / D
    gamma = (C - B * target_return) / D
    return Sigma_inv @ (lam * mu + gamma * ones)


def estimation_error_experiment(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_sims: int = 200,
    T_days: int = 1260,
    rf: float = 0.04,
    seed: int = 42,
) -> dict:
    """
    Session 4 experiment: reveals that μ̂ noise drives Markowitz weight explosion.

    Simulates n_sims independent histories from the true (mu, Sigma) and computes
    the tangency portfolio under three conditions:
        1. Both μ̂ and Σ̂ estimated  — the pathological case
        2. True μ, estimated Σ̂      — stable; Σ estimation is the easier problem
        3. Estimated μ̂, true Σ      — almost as bad as (1) — μ̂ is the culprit

    Parameters
    ----------
    mu, Sigma : true annualised moments
    n_sims    : number of bootstrap samples
    T_days    : simulated history length (trading days per sample)
    rf        : risk-free rate for tangency portfolio
    seed      : RNG seed for reproducibility

    Returns
    -------
    dict with keys 'W_both', 'W_true_mu', 'W_true_sig' — each (n_sims, N).
    """
    from alloc.base import tangency_weights

    N = len(mu)
    daily_mu = mu / 252.0
    daily_Sig = Sigma / 252.0
    L = np.linalg.cholesky(daily_Sig)
    rng = np.random.default_rng(seed)

    W_both, W_true_mu, W_true_sig = [], [], []

    for _ in range(n_sims):
        eps = rng.standard_normal((T_days, N))
        R = daily_mu + eps @ L.T
        mu_hat = R.mean(axis=0) * 252
        Sig_hat = np.cov(R.T) * 252

        for container, m, S in [
            (W_both,    mu_hat, Sig_hat),   # both estimated
            (W_true_mu, mu,     Sig_hat),   # true μ, estimated Σ̂
            (W_true_sig, mu_hat, Sigma),    # estimated μ̂, true Σ
        ]:
            try:
                w = tangency_weights(m, S, rf=rf)
                container.append(w)
            except Exception:
                pass

    return {
        "W_both":     np.array(W_both),
        "W_true_mu":  np.array(W_true_mu),
        "W_true_sig": np.array(W_true_sig),
    }
