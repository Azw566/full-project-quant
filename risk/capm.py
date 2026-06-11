"""
risk/capm.py — CAPM single-factor analysis and rolling-beta estimation.

Extracted from portfolio-risk-quant/module_2_factor_models.py (Session 5).

PUBLIC INTERFACE
────────────────
    capm_betas(returns, market_returns, rf_daily)                → dict
    rolling_betas(returns, market_returns, window, rf_daily)     → np.ndarray
"""

import numpy as np
from scipy import stats as _stats


def capm_betas(
    returns,
    market_returns: np.ndarray,
    rf_daily: float = 0.04 / 252,
) -> dict:
    """
    CAPM single-factor regression for each asset.

    Regress: r_i − rf = α_i + β_i (r_m − rf) + ε_i

    Risk decomposition (annualised):
        Systematic  = β_i² σ²_m
        Idiosyncratic = Var(ε_i)

    Parameters
    ----------
    returns        : (T, N) asset daily returns (DataFrame or ndarray)
    market_returns : (T,) market daily returns (e.g. SPY close-to-close)
    rf_daily       : daily risk-free rate (default: 4% / 252)

    Returns
    -------
    dict with keys:
        betas     : (N,) OLS beta coefficients
        alphas    : (N,) annualised Jensen's alpha
        r2s       : (N,) R-squared (fraction of variance explained by market)
        sys_vars  : (N,) annualised systematic variance  β²σ²_m
        idio_vars : (N,) annualised idiosyncratic variance Var(ε)
    """
    R = returns.values if hasattr(returns, "values") else np.asarray(returns)
    rm = np.asarray(market_returns)
    N = R.shape[1]

    excess_m = rm - rf_daily
    var_m = float(np.var(excess_m, ddof=1))

    betas, alphas, r2s, sys_vars, idio_vars = [], [], [], [], []
    for i in range(N):
        excess_i = R[:, i] - rf_daily
        slope, intercept, r_val, _, _ = _stats.linregress(excess_m, excess_i)
        betas.append(float(slope))
        alphas.append(float(intercept * 252))
        r2s.append(float(r_val ** 2))
        sys_vars.append(float(slope ** 2 * var_m * 252))
        residuals = excess_i - slope * excess_m - intercept
        idio_vars.append(float(np.var(residuals, ddof=1) * 252))

    return {
        "betas":     np.array(betas),
        "alphas":    np.array(alphas),
        "r2s":       np.array(r2s),
        "sys_vars":  np.array(sys_vars),
        "idio_vars": np.array(idio_vars),
    }


def rolling_betas(
    returns,
    market_returns: np.ndarray,
    window: int = 126,
    rf_daily: float = 0.04 / 252,
) -> np.ndarray:
    """
    Rolling CAPM beta for each asset.

    At each bar t (from `window` onward) regresses the previous `window`
    bars of excess returns on the market excess return.

    Parameters
    ----------
    returns        : (T, N) asset daily returns (DataFrame or ndarray)
    market_returns : (T,) market daily returns
    window         : rolling window length in trading days
    rf_daily       : daily risk-free rate

    Returns
    -------
    (T, N) array; NaN for bars 0 .. window-1.
    """
    R = returns.values if hasattr(returns, "values") else np.asarray(returns)
    rm = np.asarray(market_returns)
    T, N = R.shape

    rolling = np.full((T, N), np.nan)
    excess_m = rm - rf_daily

    for t in range(window, T):
        rm_w = excess_m[t - window: t]
        for i in range(N):
            ri_w = R[t - window: t, i] - rf_daily
            cov = np.cov(ri_w, rm_w)
            rolling[t, i] = cov[0, 1] / cov[1, 1]

    return rolling
