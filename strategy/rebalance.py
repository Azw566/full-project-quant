"""
strategy/rebalance.py — Rebalancing strategy adapter.

Bridges allocation functions and the multi-asset event-driven engine.
Produces a (dates × N) weight matrix that is strictly causal: the weight
at bar T is computed from returns up to bar T-1 only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_weights(
    panel: pd.DataFrame,
    allocator,
    cov_estimator,
    lookback: int,
    rebalance_every: int,
) -> pd.DataFrame:
    """
    Produce target weights by periodically re-estimating the covariance and
    calling the allocator.

    Parameters
    ----------
    panel           : Close prices (dates × N).
    allocator       : Callable matching the Allocator protocol:
                      (mu, Sigma, **params) -> (N,) weights.
                      Pass mu=None for risk-only methods (GMV, ERC, min-CVaR).
    cov_estimator   : Callable matching the CovEstimator protocol:
                      (returns_window) -> (N, N) Sigma.
    lookback        : Number of bars in the estimation window.
    rebalance_every : Re-estimate every this many bars.

    Returns
    -------
    pd.DataFrame of shape (dates × N) with the same index as panel.
    Bars before the first estimation have 0 weight (warmup).
    Between rebalance events, the last computed weight is carried forward.

    CAUSAL GUARANTEE
    ────────────────
    At bar i (0-indexed), the estimation window is rets.iloc[i-lookback : i],
    which covers bars 0 .. i-1.  The weight at bar i is therefore determined
    by strictly past data.
    """
    weights = pd.DataFrame(
        np.nan, index=panel.index, columns=panel.columns, dtype=float
    )
    rets = np.log(panel / panel.shift(1))

    for i, ts in enumerate(panel.index):
        if i < lookback:
            continue
        if (i - lookback) % rebalance_every == 0:
            window = rets.iloc[i - lookback : i].dropna()   # strictly up to T-1
            if len(window) < 2:
                continue
            Sigma = cov_estimator(window)
            w = allocator(None, Sigma)
            if w is not None:
                weights.loc[ts] = w

    return weights.ffill().fillna(0.0)
