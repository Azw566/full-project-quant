"""
alloc/mean_variance.py — Constrained mean-variance optimisation via cvxpy.

Extracted from portfolio-risk-quant/module_1_mean_variance.py.
"""

import numpy as np
import cvxpy as cp


def mvp_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target: float,
    long_only: bool = False,
    max_w: float = 1.0,
) -> np.ndarray | None:
    """
    Minimum-variance portfolio at a target return level (QP via cvxpy).

    Respects the Allocator protocol: first positional arg is mu, second is Sigma.

    Parameters
    ----------
    mu        : (N,) expected returns
    Sigma     : (N, N) covariance matrix
    target    : minimum expected return constraint
    long_only : if True, enforce w >= 0
    max_w     : per-asset weight cap (applies when < 1.0)

    Returns
    -------
    (N,) weight vector, or None if the problem is infeasible.
    """
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
    cons = [cp.sum(w) == 1, mu @ w >= target]
    if long_only:
        cons.append(w >= 0)
    if max_w < 1.0:
        cons.append(w <= max_w)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)
    return w.value if w.value is not None else None
