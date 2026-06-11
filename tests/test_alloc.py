"""
tests/test_alloc.py — Unit tests for alloc/ package.

Each test reproduces the validation cell from the corresponding notebook module.
"""

import numpy as np
import pytest

from alloc.base import (
    equal_weight, gmv_weights, tangency_weights, risk_contributions,
    port_vol, port_return, log_returns, sample_moments,
)
from alloc.risk_parity import erc_weights
from alloc.black_litterman import reverse_optimize, posterior_mean
from risk.covariance import is_psd, nearest_psd, condition_number, sample_cov


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sigma_2x2():
    """Small 2×2 PSD covariance matrix."""
    return np.array([[0.04, 0.01], [0.01, 0.09]])


@pytest.fixture
def sigma_5x5():
    """5×5 covariance from synthetic returns (reproducible)."""
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (500, 5))
    return np.cov(R.T)


@pytest.fixture
def mu_5():
    """Synthetic expected returns for 5 assets."""
    return np.array([0.08, 0.10, 0.12, 0.07, 0.09])


# ── alloc.base ─────────────────────────────────────────────────────────────────

def test_equal_weight_sums_to_one():
    for n in [1, 3, 10]:
        assert np.sum(equal_weight(n)) == pytest.approx(1.0)


def test_gmv_sums_to_one(sigma_5x5):
    w = gmv_weights(sigma_5x5)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_gmv_minimises_variance(sigma_5x5):
    w_gmv = gmv_weights(sigma_5x5)
    vol_gmv = port_vol(w_gmv, sigma_5x5)
    rng = np.random.default_rng(1)
    for _ in range(200):
        w_rand = rng.dirichlet(np.ones(5))
        assert port_vol(w_rand, sigma_5x5) >= vol_gmv - 1e-10


def test_tangency_sums_to_one(mu_5, sigma_5x5):
    w = tangency_weights(mu_5, sigma_5x5)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_risk_contributions_sum_to_port_vol(sigma_5x5):
    w = equal_weight(5)
    rc = risk_contributions(w, sigma_5x5)
    sigma_p = port_vol(w, sigma_5x5)
    assert rc.sum() == pytest.approx(sigma_p, rel=1e-8)


def test_sample_moments_shape():
    import pandas as pd
    rng = np.random.default_rng(2)
    R = pd.DataFrame(rng.normal(0, 0.01, (252, 5)))
    mu, Sigma = sample_moments(R)
    assert mu.shape == (5,)
    assert Sigma.shape == (5, 5)


# ── risk.covariance ────────────────────────────────────────────────────────────

def test_is_psd_true_for_cov(sigma_5x5):
    assert is_psd(sigma_5x5)


def test_is_psd_false_for_non_psd():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
    assert not is_psd(bad)


def test_nearest_psd_makes_psd():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    psd = nearest_psd(bad)
    assert is_psd(psd)


def test_condition_number_identity():
    assert condition_number(np.eye(4)) == pytest.approx(1.0, rel=1e-9)


def test_sample_cov_is_psd():
    rng = np.random.default_rng(3)
    R = rng.normal(0, 0.01, (300, 6))
    S = sample_cov(R)
    assert is_psd(S)


# ── alloc.risk_parity ─────────────────────────────────────────────────────────

def test_erc_weights_sum_to_one(sigma_5x5):
    w = erc_weights(None, sigma_5x5)
    assert w is not None
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_erc_equal_risk_contributions(sigma_5x5):
    """Max deviation from equal RC must be < 1e-4 (notebook validation cell)."""
    w = erc_weights(None, sigma_5x5)
    assert w is not None
    rc = risk_contributions(w, sigma_5x5)
    assert float(np.max(np.abs(rc - rc.mean()))) < 1e-4


def test_erc_rc_sum_equals_port_vol(sigma_5x5):
    """Sum of RC must equal portfolio vol (Euler decomposition)."""
    w = erc_weights(None, sigma_5x5)
    assert w is not None
    rc = risk_contributions(w, sigma_5x5)
    sigma_p = port_vol(w, sigma_5x5)
    assert rc.sum() == pytest.approx(sigma_p, rel=1e-6)


# ── alloc.black_litterman ─────────────────────────────────────────────────────

def test_bl_no_view_limit(sigma_2x2):
    """When Omega → ∞ (no confidence), mu_bar should converge to Pi."""
    w_mkt = np.array([0.5, 0.5])
    risk_aversion = 2.5
    Pi = reverse_optimize(sigma_2x2, w_mkt, risk_aversion)

    # One relative view, near-zero confidence
    P = np.array([[1.0, -1.0]])
    q = np.array([0.01])
    Omega = np.array([[1e6]])   # very uncertain → view should be ignored

    mu_bar = posterior_mean(sigma_2x2, Pi, P, q, Omega, tau=0.05)
    assert np.allclose(mu_bar, Pi, atol=1e-3), (
        f"With no-confidence view, mu_bar should ≈ Pi; got {mu_bar} vs {Pi}"
    )
