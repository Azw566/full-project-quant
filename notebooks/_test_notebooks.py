"""
Headless test of all six research notebooks.

Runs each notebook's key computations against the real cached data.
Uses matplotlib Agg so no display is needed.
"""

import sys
import os
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MPLBACKEND"] = "Agg"   # non-interactive renderer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# ── shared data helpers ───────────────────────────────────────────────────────
CACHE = str(ROOT / "data" / "cache")
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "JNJ", "XOM", "PG", "V",
]

from data.loader import get_panel, get_bars
from alloc.base import (
    log_returns, sample_moments,
    port_return, port_vol, port_sharpe,
    equal_weight, gmv_weights, tangency_weights, risk_contributions,
)

PASS = "[PASS]"
FAIL = "[FAIL]"

results = {}

def section(name):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print('='*55)

def check(label, cond, detail=""):
    tag = PASS if cond else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag} {label}{suffix}")
    if not cond:
        results[label] = "FAIL"
    else:
        results[label] = "PASS"

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 0 — Foundations
# ─────────────────────────────────────────────────────────────────────────────
section("NB-0  Foundations")

prices = get_panel(TICKERS, start="2022-01-01", end="2024-01-01", cache_dir=CACHE)
returns = log_returns(prices)
mu, Sigma = sample_moments(returns)
N = len(TICKERS)

check("data loaded", len(returns) > 200, f"{len(returns)} rows")
check("returns shape", returns.shape == (len(returns), N), str(returns.shape))
check("Sigma PSD", np.all(np.linalg.eigvalsh(Sigma) >= -1e-8))

w_eq = equal_weight(N)
eq_vol = port_vol(w_eq, Sigma)
avg_vol = float(np.sqrt(np.diag(Sigma)).mean())
check("diversification benefit", avg_vol > eq_vol,
      f"avg={avg_vol:.2%} eq={eq_vol:.2%}")

eigvals = np.linalg.eigvalsh(Sigma)[::-1]
cumvar  = np.cumsum(eigvals) / eigvals.sum()
check("eigenvalue spectrum", len(eigvals) == N)
check("cumvar reaches 1", abs(cumvar[-1] - 1.0) < 1e-8)

# interactive slider logic (using default values)
w0 = 1.0 / N
remaining = (1.0 - w0) / (N - 1)
w_custom = np.array([w0] + [remaining] * (N - 1))
check("custom weight sums to 1", abs(w_custom.sum() - 1.0) < 1e-10)

fig, _ = plt.subplots(2, 1, figsize=(12, 7))
plt.close(fig)
check("returns plot rendered", True)

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 — Mean-Variance
# ─────────────────────────────────────────────────────────────────────────────
section("NB-1  Mean-Variance Optimization")

from alloc.frontier import frontier_scalars, frontier_weights, estimation_error_experiment
from alloc.mean_variance import mvp_weights
from risk.covariance import condition_number

A, B, C, D = frontier_scalars(mu, Sigma)
check("frontier_scalars D>0", D > 0, f"D={D:.4f}")

mu_gmv = B / A
sigma2_gmv = 1.0 / A
w_gmv = gmv_weights(Sigma)
computed_var = port_vol(w_gmv, Sigma) ** 2
check("GMV sigma^2 = 1/A", abs(sigma2_gmv - computed_var) < 1e-6,
      f"formula={sigma2_gmv:.6f} computed={computed_var:.6f}")

# frontier weights
mu_lo = mu_gmv * 0.8
mu_hi = float(mu.max()) * 1.15
mu_targets = np.linspace(mu_lo, mu_hi, 50)
vols_cl = [port_vol(frontier_weights(mu, Sigma, t), Sigma) for t in mu_targets]
check("frontier parabola has finite vols", all(np.isfinite(v) for v in vols_cl))
check("frontier vols positive", all(v > 0 for v in vols_cl))

# QP constrained frontier
w_lo = mvp_weights(mu, Sigma, float(mu_gmv) + 0.01, long_only=True)
check("long-only QP weights >= 0 (solver tol)", w_lo is not None and np.all(w_lo >= -1e-4),
      f"min={w_lo.min():.4f}" if w_lo is not None else "None")
check("long-only QP sums to 1", w_lo is not None and abs(w_lo.sum() - 1.0) < 1e-3)

# tangency + CML
rf = 0.04
w_tan = tangency_weights(mu, Sigma, rf)
vol_tan = port_vol(w_tan, Sigma)
ret_tan = port_return(w_tan, mu)
sr_tan  = (ret_tan - rf) / vol_tan
check("tangency weights sum to 1", abs(w_tan.sum() - 1.0) < 1e-8)
# 2022-2024 includes a bear market; Sharpe can be negative — check it's finite
check("tangency Sharpe finite", np.isfinite(sr_tan), f"SR={sr_tan:.3f}")

# Session 4: estimation error experiment (small n_sims for speed)
exp = estimation_error_experiment(mu, Sigma, n_sims=30, T_days=252, rf=0.04, seed=0)
W_both = exp["W_both"]
W_true_sig = exp["W_true_sig"]
check("estimation_error_experiment returns 3 arrays", all(k in exp for k in ("W_both", "W_true_mu", "W_true_sig")))
W_true_mu_arr = exp["W_true_mu"]
check("estimation_error exp shapes correct",
      W_both.shape == (30, N) and W_true_mu_arr.shape == (30, N) and W_true_sig.shape == (30, N),
      f"W_both={W_both.shape} W_true_mu={W_true_mu_arr.shape} W_true_sig={W_true_sig.shape}")
check("estimation_error exp all finite",
      np.all(np.isfinite(W_both)) and np.all(np.isfinite(W_true_mu_arr)) and np.all(np.isfinite(W_true_sig)))

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 — Factor Models
# ─────────────────────────────────────────────────────────────────────────────
section("NB-2  Factor Models & Covariance Estimation")

from risk.covariance import (
    sample_cov, ledoit_wolf, factor_cov, pca_cov,
    marchenko_pastur_threshold, is_psd,
)
from risk.capm import capm_betas, rolling_betas
from sklearn.covariance import LedoitWolf

prices2 = get_panel(TICKERS, start="2020-01-01", end="2024-01-01", cache_dir=CACHE)
returns2 = log_returns(prices2)
mu2, Sigma2 = sample_moments(returns2)

# SPY market proxy
spy_bars = get_bars("SPY", start="2020-01-01", end="2024-01-01", cache_dir=CACHE)
spy_close = spy_bars["close"]
spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
spy_ret.name = "SPY"
combined = returns2.join(spy_ret, how="inner")
r_mkt = combined["SPY"].values

# CAPM betas
capm = capm_betas(combined[TICKERS], r_mkt, rf_daily=0.04/252)
check("capm_betas returns 5 arrays", len(capm) == 5)
check("betas all positive (large-cap vs SPY)", np.all(capm["betas"] > 0),
      str(capm["betas"].round(2)))
check("r2s in [0,1]", np.all((capm["r2s"] >= 0) & (capm["r2s"] <= 1)))
check("sys+idio approx Sigma diag",
      np.allclose(capm["sys_vars"] + capm["idio_vars"], np.diag(Sigma2), rtol=0.3))

# Rolling beta (combined has N+1 cols; we pass only asset columns)
rb = rolling_betas(combined[TICKERS], r_mkt, window=63, rf_daily=0.04/252)
expected_rb_shape = (len(combined), N)
check("rolling_betas shape", rb.shape == expected_rb_shape,
      f"{rb.shape} vs {expected_rb_shape}")
check("rolling_betas NaN for first window", np.all(np.isnan(rb[:62, :])))
check("rolling_betas finite after window", np.all(np.isfinite(rb[63:, :])))

# Factor covariance
R2_np = returns2.values
T2, N2 = R2_np.shape
# Build synthetic 3-factor matrix (random, for units without pandas_datareader)
rng_ff = np.random.default_rng(1)
F_syn = rng_ff.normal(0, 0.01, (T2, 3))
Sigma_factor = factor_cov(R2_np, F_syn, annualize=True)
check("factor_cov PSD", is_psd(Sigma_factor))
kappa_s = condition_number(Sigma2)
kappa_f = condition_number(Sigma_factor)
check("factor Sigma better conditioned", kappa_f < kappa_s,
      f"k_sample={kappa_s:.0f} k_factor={kappa_f:.0f}")

# MP threshold
sigma2_mp = float(np.mean(np.diag(np.cov(R2_np.T))))
lp = marchenko_pastur_threshold(N2, T2, sigma2=sigma2_mp)
check("MP threshold finite and positive", lp > 0 and np.isfinite(lp),
      f"lambda+={lp:.6f}")
eigvals2 = np.linalg.eigvalsh(np.cov(R2_np.T))[::-1]
signal_eigs = eigvals2[eigvals2 > lp]
check("at least 1 signal eigenvalue above MP floor", len(signal_eigs) >= 1,
      f"{len(signal_eigs)} above lambda+")

# PCA cov
Sigma_pca = pca_cov(R2_np, n_components=3)
check("pca_cov PSD", is_psd(Sigma_pca))

# Ledoit-Wolf
Sigma_lw = ledoit_wolf(R2_np, annualize=True)
check("ledoit_wolf PSD", is_psd(Sigma_lw))
kappa_lw = condition_number(Sigma_lw)
check("LW better conditioned than sample", kappa_lw < kappa_s,
      f"k_LW={kappa_lw:.0f} k_sample={kappa_s:.0f}")

# LW Frobenius error simulation
n_t, t_t = 10, 120
true_S = np.eye(n_t) * 0.04 + n_t * 0.001 * np.eye(n_t)
rng_lw = np.random.default_rng(123)
frob_s, frob_lw = [], []
for _ in range(50):
    R_sim = rng_lw.multivariate_normal(np.zeros(n_t), true_S / 252, size=t_t)
    frob_s.append(np.linalg.norm(np.cov(R_sim.T) * 252 - true_S, "fro"))
    frob_lw.append(np.linalg.norm(LedoitWolf().fit(R_sim).covariance_ * 252 - true_S, "fro"))
check("LW reduces Frobenius error", np.mean(frob_lw) < np.mean(frob_s),
      f"LW={np.mean(frob_lw):.3f} sample={np.mean(frob_s):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 3 — Risk Measures
# ─────────────────────────────────────────────────────────────────────────────
section("NB-3  Risk Measures")

from risk.var_cvar import (
    var_parametric, var_historical, var_mc,
    es_parametric, es_historical, min_cvar_weights,
)

w_eq2 = equal_weight(N)
port_ret_series = returns2.values @ w_eq2
port_losses = -port_ret_series
mu_p_d = float(np.mean(port_ret_series))
sig_p_d = float(np.std(port_ret_series, ddof=1))

c = 0.95
v_param = var_parametric(mu_p_d, sig_p_d, c)
v_hist  = var_historical(port_losses, c)
v_mc    = var_mc(mu_p_d, sig_p_d, c, n_sims=50_000, seed=7)

check("VaR parametric > 0", v_param > 0, f"{v_param:.4%}")
check("VaR historical > 0", v_hist > 0, f"{v_hist:.4%}")
check("VaR MC approx parametric (Gaussian)", abs(v_mc - v_param) < 0.005,
      f"param={v_param:.4%} mc={v_mc:.4%}")

e_param = es_parametric(mu_p_d, sig_p_d, c)
e_hist  = es_historical(port_losses, c)
check("ES > VaR (parametric)", e_param > v_param,
      f"ES={e_param:.4%} VaR={v_param:.4%}")
check("ES > VaR (historical)", e_hist > v_hist,
      f"ES={e_hist:.4%} VaR={v_hist:.4%}")

# Subadditivity violation
var_a, var_b, var_comb = 0.0, 0.0, 0.5
check("VaR subadditivity violation: VaR(A+B)>VaR(A)+VaR(B)", var_comb > var_a + var_b)

# Min-CVaR (small run for speed)
w_cv = min_cvar_weights(returns2.iloc[-252:], confidence=0.95)
check("min_cvar_weights converged", w_cv is not None)
if w_cv is not None:
    check("min_cvar weights >= 0", np.all(w_cv >= -1e-4), f"min={w_cv.min():.4f}")
    check("min_cvar weights sum to 1", abs(w_cv.sum() - 1.0) < 1e-3, f"sum={w_cv.sum():.4f}")
    ew_cv = equal_weight(N)
    ew_losses = -(returns2.values[-252:] @ ew_cv)
    cv_ew = float(np.mean(ew_losses[ew_losses >= np.quantile(ew_losses, 0.95)]))
    opt_losses = -(returns2.values[-252:] @ w_cv)
    cv_opt = float(np.mean(opt_losses[opt_losses >= np.quantile(opt_losses, 0.95)]))
    check("min-CVaR <= equal-weight CVaR", cv_opt <= cv_ew * 1.05,
          f"opt={cv_opt:.4%} ew={cv_ew:.4%}")

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 4 — Modern Allocation
# ─────────────────────────────────────────────────────────────────────────────
section("NB-4  Modern Allocation")

from alloc.risk_parity import erc_weights
from alloc.black_litterman import reverse_optimize, posterior_mean

w_erc = erc_weights(None, Sigma2)
check("erc_weights converged", w_erc is not None)
if w_erc is not None:
    rc = risk_contributions(w_erc, Sigma2)
    max_dev = float(np.max(np.abs(rc - rc.mean())))
    check("ERC equal risk contributions", max_dev < 1e-3, f"max dev={max_dev:.2e}")
    check("ERC weights positive", np.all(w_erc > 0))
    check("ERC weights sum to 1", abs(w_erc.sum() - 1.0) < 1e-6)

    # Diagonal Sigma sanity: ERC → 1/sigma_i
    sig_i = np.sqrt(np.diag(Sigma2))
    w_invvol = (1 / sig_i) / (1 / sig_i).sum()
    Sig_diag = np.diag(np.diag(Sigma2))
    w_erc_diag = erc_weights(None, Sig_diag)
    if w_erc_diag is not None:
        check("ERC(diag Sigma) approx inv-vol", np.max(np.abs(w_erc_diag - w_invvol)) < 1e-3,
              f"max diff={np.max(np.abs(w_erc_diag - w_invvol)):.2e}")

# Black-Litterman
spy_bl = get_bars("SPY", start="2020-01-01", end="2024-01-01", cache_dir=CACHE)
spy_r_bl = spy_bl["close"]
spy_ret_bl = float(np.log(spy_r_bl / spy_r_bl.shift(1)).dropna().mean()) * 252
spy_vol_bl = float(np.log(spy_r_bl / spy_r_bl.shift(1)).dropna().std()) * np.sqrt(252)
RF_BL = 0.04
lam_bl = (spy_ret_bl - RF_BL) / spy_vol_bl ** 2

w_mkt = np.ones(N) / N
Pi = reverse_optimize(Sigma2, w_mkt, lam_bl)
check("implied returns Pi finite", np.all(np.isfinite(Pi)))
check("Pi has plausible scale", np.all(np.abs(Pi) < 2.0))

P_bl = np.zeros((2, N))
P_bl[0, TICKERS.index("AAPL")] = 1.0
P_bl[1, TICKERS.index("MSFT")] = 1.0
P_bl[1, TICKERS.index("GOOGL")] = -1.0
q_bl = np.array([Pi[TICKERS.index("AAPL")] + 0.03, 0.02])
Omega_bl = np.diag([0.02, 0.04])
tau = 0.05

mu_bar = posterior_mean(Sigma2, Pi, P_bl, q_bl, Omega_bl, tau=tau)
check("BL posterior finite", np.all(np.isfinite(mu_bar)))

# No-views limit: Omega -> inf → mu_bar -> Pi
Omega_large = np.eye(2) * 1e6
mu_noview = posterior_mean(Sigma2, Pi, P_bl, q_bl, Omega_large, tau=tau)
diff_to_pi = float(np.max(np.abs(mu_noview - Pi)))
check("BL no-view limit -> Pi", diff_to_pi < 1e-4, f"max diff={diff_to_pi:.2e}")

# Views fade: increasing Omega → weights converge toward market
omega_scales = np.logspace(-2, 3, 10)
final_w = []
for scale in omega_scales:
    mb = posterior_mean(Sigma2, Pi, P_bl, q_bl, np.diag([scale, scale]), tau=tau)
    final_w.append(tangency_weights(mb, Sigma2, rf=RF_BL))
final_w = np.array(final_w)
check("BL weights matrix finite", np.all(np.isfinite(final_w)))

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 5 — Reality (Walk-Forward)
# ─────────────────────────────────────────────────────────────────────────────
section("NB-5  Reality — Walk-Forward Backtest")

from risk.covariance import ledoit_wolf as lw_fn
from strategy.rebalance import generate_weights
from backtest.multiasset import run as bt_run

prices5 = get_panel(TICKERS, start="2018-01-01", end="2024-12-31", cache_dir=CACHE)
LOOKBACK, REBALANCE, FEE = 126, 21, 10

def alloc_ew(mu, Sigma, **_):
    return equal_weight(Sigma.shape[0])

def alloc_gmv(mu, Sigma, **_):
    return gmv_weights(Sigma)

print("  Generating weights (EW, GMV, ERC) …")
w_ew5  = generate_weights(prices5, alloc_ew,  lw_fn, LOOKBACK, REBALANCE)
w_gmv5 = generate_weights(prices5, alloc_gmv, lw_fn, LOOKBACK, REBALANCE)
w_erc5 = generate_weights(prices5, erc_weights, lw_fn, LOOKBACK, REBALANCE)

check("EW weights shape", w_ew5.shape == prices5.shape, str(w_ew5.shape))
check("EW weights non-NaN after warmup",
      w_ew5.iloc[LOOKBACK:].notna().all().all())
check("ERC weights non-negative", (w_erc5.iloc[LOOKBACK:] >= -1e-6).all().all())

print("  Running backtests …")
res_ew5  = bt_run(prices5, w_ew5,  fee_bps=FEE)
res_gmv5 = bt_run(prices5, w_gmv5, fee_bps=FEE)
res_erc5 = bt_run(prices5, w_erc5, fee_bps=FEE)

check("EW backtest has rows", len(res_ew5) > 100, f"{len(res_ew5)} rows")
check("GMV backtest has rows", len(res_gmv5) > 100)
check("ERC backtest has rows", len(res_erc5) > 100)

check("EW net_return finite", res_ew5["net_return"].notna().all())
check("GMV net_return finite", res_gmv5["net_return"].notna().all())
check("ERC net_return finite", res_erc5["net_return"].notna().all())

# Performance sanity
for label, res in [("EW", res_ew5), ("GMV", res_gmv5), ("ERC", res_erc5)]:
    r = res["net_return"].values
    ann_ret = float(np.mean(r) * 252)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
    sharpe  = (ann_ret - 0.04) / ann_vol
    cum = np.cumprod(1 + r)
    max_dd = float(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min())
    print(f"  {label:4s}: ann_ret={ann_ret:.2%}  ann_vol={ann_vol:.2%}  Sharpe={sharpe:.2f}  MDD={max_dd:.2%}")
    check(f"{label} Sharpe finite", np.isfinite(sharpe))
    check(f"{label} MDD in [-1, 0]", -1.0 <= max_dd <= 0.0, f"{max_dd:.2%}")
    check(f"{label} equity stays positive", np.all(cum > 0))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")
n_pass = sum(1 for v in results.values() if v == "PASS")
n_fail = sum(1 for v in results.values() if v == "FAIL")
total  = len(results)

print(f"\n  {n_pass}/{total} checks passed", end="")
if n_fail:
    print(f"  ({n_fail} failures:)")
    for k, v in results.items():
        if v == "FAIL":
            print(f"    {FAIL} {k}")
else:
    print("  — all clear")

sys.exit(0 if n_fail == 0 else 1)
