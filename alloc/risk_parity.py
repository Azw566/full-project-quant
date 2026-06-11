"""
alloc/risk_parity.py — Equal Risk Contribution (ERC / risk parity) allocation.

Extracted from portfolio-risk-quant/module_4_modern_allocation.py.
"""

import numpy as np
import cvxpy as cp


def erc_weights(mu: np.ndarray | None, Sigma: np.ndarray, **_) -> np.ndarray | None:
    """
    Equal Risk Contribution portfolio via the Maillard-Roncalli-Teïletche
    log-barrier convex formulation.

    Conforms to the Allocator protocol (mu is accepted but ignored — ERC
    does not require a return estimate, which makes it more robust OOS).

    Validation: max |RC_i - RC_j| < 1e-4 and sum(RC_i) = sigma_p.

    Returns
    -------
    (N,) normalised weight vector, or None if the solver does not converge.
    """
    n = Sigma.shape[0]
    w = cp.Variable(n, pos=True)
    obj = cp.Minimize(
        0.5 * cp.quad_form(w, cp.psd_wrap(Sigma)) - (1.0 / n) * cp.sum(cp.log(w))
    )
    prob = cp.Problem(obj)
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-8)
    if w.value is None:
        return None
    w_val = np.maximum(w.value, 1e-10)
    return w_val / w_val.sum()
