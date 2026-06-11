import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Notebook 1 — Mean-Variance Optimization
    ## Sessions 1–4: Efficient Frontier → Why Markowitz Fails

    This notebook derives Markowitz from scratch, then deliberately breaks it.
    Sessions 1–3 build the machinery; Session 4 shows why it fails in practice
    — which motivates everything in Notebooks 2–4.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cvxpy as cp
    from scipy.optimize import minimize

    from data.loader import get_panel
    from alloc.base import (
        log_returns, sample_moments,
        port_return, port_vol, equal_weight,
        gmv_weights, tangency_weights,
    )
    from alloc.frontier import frontier_scalars, frontier_weights, estimation_error_experiment
    from alloc.mean_variance import mvp_weights
    from risk.covariance import condition_number
    return (
        np, pd, plt, cp, minimize, Path,
        get_panel, log_returns, sample_moments,
        port_return, port_vol, equal_weight,
        gmv_weights, tangency_weights,
        frontier_scalars, frontier_weights, estimation_error_experiment,
        mvp_weights, condition_number,
    )


@app.cell
def _(get_panel, log_returns, sample_moments, Path):
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "JPM", "JNJ", "XOM", "PG", "V",
    ]
    CACHE = str(Path(__file__).parent.parent / "data" / "cache")
    prices = get_panel(TICKERS, start="2022-01-01", end="2024-01-01", cache_dir=CACHE)
    returns = log_returns(prices)
    mu, Sigma = sample_moments(returns)
    tickers = list(prices.columns)
    N = len(tickers)
    return prices, returns, mu, Sigma, tickers, N, TICKERS, CACHE


# ═══════════════════════════════════════════════════════════════
# SESSION 1 — Efficient Frontier (Closed Form)
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 1 — The Efficient Frontier (Closed Form)

    Minimise variance for a target return $\mu^*$ using Lagrange multipliers:

    $$\min_w \tfrac12 w^T\Sigma w \quad\text{s.t.}\quad w^T\mu=\mu^*,\;\; w^T\mathbf{1}=1$$

    Define the four scalars:

    $$A = \mathbf{1}^T\Sigma^{-1}\mathbf{1},\quad B = \mathbf{1}^T\Sigma^{-1}\mu,\quad
      C = \mu^T\Sigma^{-1}\mu,\quad D = AC - B^2$$

    The frontier is the **parabola** $\sigma^2(\mu^*) = (A\mu^{*2} - 2B\mu^* + C) / D$.

    **Global minimum-variance portfolio** (no dependence on μ!):
    $w_{\text{gmv}} = \Sigma^{-1}\mathbf{1} / (\mathbf{1}^T\Sigma^{-1}\mathbf{1})$,
    $\sigma^2_{\text{gmv}} = 1/A$, $\mu_{\text{gmv}} = B/A$.
    """)
    return


@app.cell
def _(np, mu, Sigma, frontier_scalars, gmv_weights, port_vol, port_return):
    A, B, C, D = frontier_scalars(mu, Sigma)
    mu_gmv = B / A
    sigma2_gmv = 1.0 / A
    w_gmv = gmv_weights(Sigma)
    vol_gmv = port_vol(w_gmv, Sigma)
    ret_gmv = port_return(w_gmv, mu)
    print(f"A={A:.4f}  B={B:.4f}  C={C:.4f}  D={D:.6f}")
    print(f"GMV: μ={mu_gmv:.2%}  σ={sigma2_gmv**0.5:.2%}")
    return A, B, C, D, mu_gmv, sigma2_gmv, w_gmv, vol_gmv, ret_gmv


@app.cell
def _(np, plt, mu, Sigma, frontier_weights, w_gmv, vol_gmv, ret_gmv, mu_gmv, port_vol, port_return, tickers):
    mu_lo = mu_gmv * 0.8
    mu_hi = float(mu.max()) * 1.15
    mu_targets = np.linspace(mu_lo, mu_hi, 200)

    vols_cl = [port_vol(frontier_weights(mu, Sigma, t), Sigma) for t in mu_targets]
    vols_ind = np.sqrt(np.diag(Sigma))

    fig_s1, ax_s1 = plt.subplots(figsize=(9, 6))
    ax_s1.plot(vols_cl, mu_targets, "b-", linewidth=2.5, label="Efficient Frontier")
    ax_s1.scatter(vol_gmv, ret_gmv, s=180, color="red", zorder=6,
                  label=f"GMV  (σ={vol_gmv:.1%}, μ={ret_gmv:.1%})")
    ax_s1.scatter(vols_ind, mu, s=60, color="gray", alpha=0.7, label="Individual assets")
    for i, t in enumerate(tickers):
        ax_s1.annotate(t, (vols_ind[i], mu[i]), textcoords="offset points",
                       xytext=(4, 3), fontsize=7)
    ax_s1.set_xlabel("Annualised Volatility σ")
    ax_s1.set_ylabel("Annualised Expected Return μ")
    ax_s1.set_title("Markowitz Efficient Frontier — Unconstrained (Closed Form)")
    ax_s1.legend()
    ax_s1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_s1
    return fig_s1, ax_s1, mu_lo, mu_hi, mu_targets, vols_cl, vols_ind


@app.cell
def _(mo, np, sigma2_gmv, w_gmv, port_vol, Sigma, tickers):
    computed_var = port_vol(w_gmv, Sigma) ** 2
    mo.md(f"""
    **Validation:**
    - σ²_gmv = 1/A: expected `{sigma2_gmv:.6f}`, computed `{computed_var:.6f}` — {'✅' if abs(sigma2_gmv - computed_var) < 1e-6 else '❌'}
    - GMV weights sum: `{float(np.sum(w_gmv)):.6f}` — {'✅' if abs(float(np.sum(w_gmv)) - 1) < 1e-8 else '❌'}
    - Weights: {', '.join(f'{tickers[i]}: {w_gmv[i]:.1%}' for i in range(len(tickers)))}
    """)
    return computed_var


# ═══════════════════════════════════════════════════════════════
# SESSION 2 — Constrained QP with cvxpy
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 2 — Constraints and the QP Formulation

    Adding $w_i \geq 0$ (no-short) eliminates the closed form.
    The problem is still a convex **QP** — PSD $\Sigma$ guarantees a global optimum.

    $$\min_w \; w^T\Sigma w \quad\text{s.t.}\quad w^T\mathbf{1}=1,\; w^T\mu\ge\mu^*,\; w\ge0$$

    **Key:** adding constraints can only *shrink* the feasible set, so the constrained
    frontier sits *inside* (higher vol for same return) the unconstrained one.
    """)
    return


@app.cell
def _(np, mvp_weights, mu, Sigma, port_vol, mu_gmv):
    targets_qp = np.linspace(mu_gmv + 0.005, float(mu.max()) * 0.95, 40)
    vols_unc, vols_lo, vols_cap = [], [], []

    for t in targets_qp:
        w_u = mvp_weights(mu, Sigma, t, long_only=False)
        w_l = mvp_weights(mu, Sigma, t, long_only=True)
        w_c = mvp_weights(mu, Sigma, t, long_only=True, max_w=0.3)
        if w_u is not None: vols_unc.append(port_vol(w_u, Sigma))
        if w_l is not None: vols_lo.append(port_vol(w_l, Sigma))
        if w_c is not None: vols_cap.append(port_vol(w_c, Sigma))

    return targets_qp, vols_unc, vols_lo, vols_cap


@app.cell
def _(plt, targets_qp, vols_unc, vols_lo, vols_cap):
    fig_s2, ax_s2 = plt.subplots(figsize=(9, 6))
    ax_s2.plot(vols_unc, targets_qp[:len(vols_unc)], "b-",  linewidth=2, label="Unconstrained")
    ax_s2.plot(vols_lo,  targets_qp[:len(vols_lo)],  "g--", linewidth=2, label="Long-only ($w \\geq 0$)")
    ax_s2.plot(vols_cap, targets_qp[:len(vols_cap)], "r:",  linewidth=2, label="Long-only + max weight 30%")
    ax_s2.set_xlabel("Annualised Volatility σ")
    ax_s2.set_ylabel("Annualised Expected Return μ")
    ax_s2.set_title("Frontier Shrinks as Constraints Are Added")
    ax_s2.legend()
    ax_s2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_s2
    return fig_s2, ax_s2


# ═══════════════════════════════════════════════════════════════
# SESSION 3 — Tangency Portfolio & Capital Market Line
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 3 — Sharpe Ratio, Tangency Portfolio, and the CML

    With a risk-free rate $r_f$, the **Sharpe ratio** is:
    $\mathrm{SR} = (w^T\mu - r_f) / \sqrt{w^T\Sigma w}$

    The **tangency portfolio** (max-Sharpe):
    $w_{\text{tan}} = \Sigma^{-1}(\mu - r_f\mathbf{1}) / (\mathbf{1}^T\Sigma^{-1}(\mu - r_f\mathbf{1}))$

    **Two-fund separation:** every efficient portfolio is a mix of risk-free asset
    and $w_{\text{tan}}$.  The **Capital Market Line** is the new efficient frontier.
    """)
    return


@app.cell
def _(mo):
    rf_slider = mo.ui.slider(0.0, 0.06, step=0.005, value=0.04,
                              label="Risk-free rate rf")
    rf_slider
    return (rf_slider,)


@app.cell
def _(np, plt, mu, Sigma, rf_slider, tangency_weights, port_vol, port_return, vols_cl, mu_targets, vols_ind, tickers):
    rf = rf_slider.value
    w_tan = tangency_weights(mu, Sigma, rf)
    vol_tan = port_vol(w_tan, Sigma)
    ret_tan = port_return(w_tan, mu)
    sr_tan = (ret_tan - rf) / vol_tan

    cml_vols = np.linspace(0, float(max(vols_cl)) * 1.15, 200)
    cml_rets = rf + sr_tan * cml_vols

    fig_s3, ax_s3 = plt.subplots(figsize=(9, 6))
    ax_s3.plot(vols_cl, mu_targets, "b-", linewidth=2, alpha=0.7, label="Efficient Frontier")
    ax_s3.plot(cml_vols, cml_rets, "g-", linewidth=2,
               label=f"CML  (Sharpe = {sr_tan:.2f})")
    ax_s3.scatter(vol_tan, ret_tan, s=220, color="orange", zorder=6,
                  marker="*", label=f"Tangency  rf={rf:.2%}")
    ax_s3.scatter(0, rf, s=100, color="green", zorder=6, label="Risk-free asset")
    ax_s3.scatter(vols_ind, mu, s=50, color="gray", alpha=0.6)
    ax_s3.set_xlim(left=0)
    ax_s3.set_xlabel("Annualised Volatility σ")
    ax_s3.set_ylabel("Annualised Expected Return μ")
    ax_s3.set_title("Tangency Portfolio and Capital Market Line")
    ax_s3.legend()
    ax_s3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_s3
    return rf, w_tan, vol_tan, ret_tan, sr_tan, cml_vols, cml_rets, fig_s3, ax_s3


# ═══════════════════════════════════════════════════════════════
# SESSION 4 — Why Markowitz Fails (Estimation Error)  ⚠️ pivotal
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 4 — Why Markowitz Fails in Practice ⚠️

    The standard error of a mean return scales like $\sigma/\sqrt{T}$.
    With $\sigma \approx 30\%$/yr you need **decades** to reliably rank two assets.

    **Experiment:** simulate many samples from a known true $\mu, \Sigma$ and compute
    the tangency portfolio from each.  Three conditions reveal the source of instability.
    """)
    return


@app.cell
def _(mo):
    n_sims_slider = mo.ui.slider(50, 500, step=50, value=200, label="Simulations")
    T_months_slider = mo.ui.slider(24, 120, step=12, value=60, label="History (months)")
    mo.hstack([n_sims_slider, T_months_slider])
    return n_sims_slider, T_months_slider


@app.cell
def _(np, plt, mu, Sigma, n_sims_slider, T_months_slider,
      estimation_error_experiment, tickers, N):
    n_sims = n_sims_slider.value
    T_days_sim = T_months_slider.value * 21

    result = estimation_error_experiment(
        mu, Sigma,
        n_sims=n_sims,
        T_days=T_days_sim,
        rf=0.04,
        seed=42,
    )
    W_both    = result["W_both"]
    W_true_mu = result["W_true_mu"]
    W_true_sig = result["W_true_sig"]

    labels = [t[:4] for t in tickers]
    fig_s4, axes_s4 = plt.subplots(1, 3, figsize=(15, 5))

    for ax, W, title, color in zip(
        axes_s4,
        [W_both, W_true_mu, W_true_sig],
        ["Both estimated\n(μ̂, Σ̂)",
         "True μ, estimated Σ̂\n(only Σ noisy)",
         "Estimated μ̂, true Σ\n(only μ noisy)"],
        ["tomato", "steelblue", "darkorange"],
    ):
        if len(W) > 0:
            ax.boxplot(W, labels=labels, showfliers=False,
                       medianprops={"color": color, "linewidth": 2},
                       boxprops={"linewidth": 1.2})
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Weight")
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle(
        f"Tangency Weight Instability — {n_sims} simulations, T={T_months_slider.value} months",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig_s4
    return (fig_s4, axes_s4, n_sims, T_days_sim, result,
            W_both, W_true_mu, W_true_sig, labels)


@app.cell
def _(mo, np, W_both, W_true_mu, W_true_sig):
    def _summary(W):
        if len(W) == 0:
            return "n/a", "n/a"
        return f"{np.std(W, axis=0).mean():.3f}", f"{np.abs(W).max():.1f}"

    std_both, max_both   = _summary(W_both)
    std_mu,   max_mu     = _summary(W_true_mu)
    std_sig,  max_sig    = _summary(W_true_sig)

    mo.md(f"""
    ### Key takeaway

    | Scenario | Avg weight std | Max |w| |
    |----------|---------------|---------|
    | Both estimated (μ̂, Σ̂) | **{std_both}** | {max_both} |
    | True μ, estimated Σ̂   | {std_mu} | {max_mu} |
    | Estimated μ̂, true Σ   | **{std_sig}** | {max_sig} |

    **Conclusion:** the weight explosion comes overwhelmingly from $\\hat\\mu$, not $\\hat\\Sigma$.

    Two repairs follow:
    1. **Regularise Σ** — Notebook 2 (factor models, shrinkage, PCA)
    2. **Avoid or replace μ** — Notebook 4 (risk parity, Black–Litterman)
    """)
    return std_both, max_both, std_mu, max_mu, std_sig, max_sig


if __name__ == "__main__":
    app.run()
