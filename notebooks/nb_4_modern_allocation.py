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
    # Notebook 4 — Modern Allocation
    ## Sessions 11–12: Risk Parity → Black–Litterman

    **The core problem:** raw $\hat\mu$ is too noisy to use directly (Session 4).

    Two canonical solutions:
    1. **Risk Parity** — ignore $\mu$ entirely; use only $\Sigma$
    2. **Black–Litterman** — replace raw $\hat\mu$ with a Bayesian posterior anchored to the market
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

    from data.loader import get_panel, get_bars
    from alloc.base import (
        log_returns, sample_moments,
        port_return, port_vol, port_sharpe, equal_weight,
        gmv_weights, tangency_weights, risk_contributions,
    )
    from alloc.risk_parity import erc_weights
    from alloc.black_litterman import reverse_optimize, posterior_mean
    return (
        np, pd, plt, Path,
        get_panel, get_bars, log_returns, sample_moments,
        port_return, port_vol, port_sharpe, equal_weight,
        gmv_weights, tangency_weights, risk_contributions,
        erc_weights, reverse_optimize, posterior_mean,
    )


@app.cell
def _(get_panel, log_returns, sample_moments, Path):
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "JPM", "JNJ", "XOM", "PG", "V",
    ]
    CACHE = str(Path(__file__).parent.parent / "data" / "cache")
    prices = get_panel(TICKERS, start="2020-01-01", end="2024-01-01", cache_dir=CACHE)
    returns = log_returns(prices)
    mu, Sigma = sample_moments(returns)
    tickers = list(prices.columns)
    N = len(tickers)
    return prices, returns, mu, Sigma, tickers, N, TICKERS, CACHE


# ═══════════════════════════════════════════════════════════════
# SESSION 11 — Risk Parity / Equal Risk Contribution (ERC)
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 11 — Risk Parity (Equal Risk Contribution)

    Portfolio volatility decomposes via **Euler's theorem**:

    $$\sigma_p = \sum_i \mathrm{RC}_i, \qquad
      \mathrm{RC}_i = w_i \frac{(\Sigma w)_i}{\sigma_p}$$

    **ERC:** find $w > 0$ such that all $\mathrm{RC}_i$ are equal ($= \sigma_p / N$).

    Convex reformulation (Maillard–Roncalli–Teïletche / Spinu):

    $$\min_{w>0}\; \tfrac12 w^T\Sigma w - \frac{1}{N}\sum_i \ln w_i$$

    then rescale.  The log-barrier forces strictly positive weights.

    **Sanity check:** when $\Sigma$ is diagonal, ERC reduces to $w_i \propto 1/\sigma_i$.
    """)
    return


@app.cell
def _(np, Sigma, N, erc_weights, risk_contributions):
    w_erc = erc_weights(None, Sigma)
    if w_erc is not None:
        rc_erc = risk_contributions(w_erc, Sigma)
        sigma_p_erc = float(np.sqrt(w_erc @ Sigma @ w_erc))
        max_rc_diff = float(np.max(np.abs(rc_erc - rc_erc.mean())))
        print(f"ERC vol: {sigma_p_erc:.2%}  max RC deviation: {max_rc_diff:.2e}")
    else:
        rc_erc = np.ones(N) / N
        sigma_p_erc = 0.0
        max_rc_diff = float("nan")
    return w_erc, rc_erc, sigma_p_erc, max_rc_diff


@app.cell
def _(mo, max_rc_diff, rc_erc, sigma_p_erc, N):
    mo.md(f"""
    **Validation:**
    - All RC equal? Max deviation = `{max_rc_diff:.2e}` — {'✅' if max_rc_diff < 1e-4 else '⚠️ not converged'}
    - Sum of RC = σ_p: `{sum(rc_erc):.4%}` vs `{sigma_p_erc:.4%}` — {'✅' if abs(sum(rc_erc) - sigma_p_erc) < 1e-8 else '❌'}
    - Expected each RC = σ_p/{N} = `{sigma_p_erc/N:.4%}`, actual mean = `{rc_erc.mean():.4%}`
    """)
    return


@app.cell
def _(np, Sigma, tickers, N, risk_contributions, equal_weight, gmv_weights,
      port_return, port_vol, port_sharpe, w_erc):
    sigma_i = np.sqrt(np.diag(Sigma))
    w_invvol = (1 / sigma_i) / (1 / sigma_i).sum()
    w_eq  = equal_weight(N)
    w_gmv = gmv_weights(Sigma)

    portfolios = {}
    if w_erc is not None:
        portfolios = {
            "Equal Weight":     w_eq,
            "Risk Parity (ERC)": w_erc,
            "Inv-Volatility":   w_invvol,
            "Global Min-Var":   w_gmv,
        }
    else:
        portfolios = {
            "Equal Weight":   w_eq,
            "Inv-Volatility": w_invvol,
            "Global Min-Var": w_gmv,
        }
    return sigma_i, w_invvol, w_eq, w_gmv, portfolios


@app.cell
def _(plt, np, mu, Sigma, tickers, N, portfolios, risk_contributions,
      port_return, port_vol, port_sharpe):
    fig_s11, axes11 = plt.subplots(2, 2, figsize=(14, 10))
    x11 = np.arange(N)
    width11 = 0.2

    ax = axes11[0, 0]
    for i, (name, w) in enumerate(portfolios.items()):
        ax.bar(x11 + i * width11, w, width11, label=name, alpha=0.85)
    ax.set_xticks(x11 + width11 * (len(portfolios) - 1) / 2)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights Comparison")
    ax.legend(fontsize=8)

    ax = axes11[0, 1]
    for i, (name, w) in enumerate(portfolios.items()):
        rc = risk_contributions(w, Sigma)
        ax.bar(x11 + i * width11, rc, width11, label=name, alpha=0.85)
    ax.set_xticks(x11 + width11 * (len(portfolios) - 1) / 2)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Risk Contribution RC_i")
    ax.set_title("Risk Contributions")
    ax.legend(fontsize=8)

    ax = axes11[1, 0]
    vols_ind = np.sqrt(np.diag(Sigma))
    for name, w in portfolios.items():
        r = port_return(w, mu)
        v = port_vol(w, Sigma)
        ax.scatter(v, r, s=120, zorder=5)
        ax.annotate(name, (v, r), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.scatter(vols_ind, mu, s=40, color="gray", alpha=0.5)
    ax.set_xlabel("Vol σ")
    ax.set_ylabel("Expected Return μ")
    ax.set_title("Risk-Return Map")
    ax.grid(True, alpha=0.3)

    ax = axes11[1, 1]
    sharpes = {name: port_sharpe(w, mu, Sigma, rf=0.04) for name, w in portfolios.items()}
    bars11 = ax.bar(list(sharpes.keys()), list(sharpes.values()), color="steelblue", alpha=0.8)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratios (rf=4%)")
    ax.tick_params(axis="x", rotation=20)
    for bar11, v11 in zip(bars11, sharpes.values()):
        ax.text(bar11.get_x() + bar11.get_width()/2, v11 + 0.01, f"{v11:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_s11
    return fig_s11, axes11, x11, width11, sharpes, vols_ind


# ═══════════════════════════════════════════════════════════════
# SESSION 12 — Black–Litterman
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 12 — Black–Litterman

    ### Step 1 — Reverse-optimise the market
    Implied equilibrium returns: $\Pi = \lambda\,\Sigma\,w_{\text{mkt}}$,
    where $\lambda = (\mathbb{E}[r_m] - r_f) / \sigma_m^2$

    ### Step 2 — Views
    Prior: $\mu \sim N(\Pi, \tau\Sigma)$.  Views: $P\mu = q + \varepsilon$, $\varepsilon \sim N(0,\Omega)$

    ### Step 3 — Posterior mean
    $$\bar\mu = \big[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\big]^{-1}
      \big[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}q\big]$$
    """)
    return


@app.cell
def _(np, get_bars, Path):
    CACHE_BL = str(Path(__file__).parent.parent / "data" / "cache")
    spy_bl = get_bars("SPY", start="2020-01-01", end="2024-01-01", cache_dir=CACHE_BL)
    spy_r = spy_bl["close"]
    spy_ret_bl = float(np.log(spy_r / spy_r.shift(1)).dropna().mean()) * 252
    spy_vol_bl = float(np.log(spy_r / spy_r.shift(1)).dropna().std()) * np.sqrt(252)
    RF_BL = 0.04
    risk_aversion = (spy_ret_bl - RF_BL) / spy_vol_bl ** 2
    print(f"Market: E[rm]={spy_ret_bl:.2%}  σ_m={spy_vol_bl:.2%}  λ={risk_aversion:.2f}")
    return spy_bl, spy_r, spy_ret_bl, spy_vol_bl, RF_BL, risk_aversion, CACHE_BL


@app.cell
def _(np, Sigma, tickers, N, risk_aversion, reverse_optimize):
    w_mkt = np.ones(N) / N
    Pi = reverse_optimize(Sigma, w_mkt, risk_aversion)
    print("Implied equilibrium returns Π:")
    for t, p in zip(tickers, Pi):
        print(f"  {t}: {p:.2%}")
    return w_mkt, Pi


@app.cell
def _(mo):
    tau_slider = mo.ui.slider(0.01, 0.5, step=0.01, value=0.05,
                               label="τ (prior uncertainty scale)")
    view1_conf_slider = mo.ui.slider(0.001, 0.1, step=0.001, value=0.02,
                               label="View 1 uncertainty Ω₁ (lower = more confident)")
    view2_conf_slider = mo.ui.slider(0.001, 0.1, step=0.001, value=0.04,
                               label="View 2 uncertainty Ω₂")
    mo.vstack([tau_slider, view1_conf_slider, view2_conf_slider])
    return tau_slider, view1_conf_slider, view2_conf_slider


@app.cell
def _(np, Sigma, Pi, mu, tickers, N, w_mkt, RF_BL,
      tau_slider, view1_conf_slider, view2_conf_slider,
      tangency_weights, posterior_mean):
    tau = tau_slider.value

    aapl_idx  = tickers.index("AAPL")
    msft_idx  = tickers.index("MSFT")
    googl_idx = tickers.index("GOOGL")

    P_views = np.zeros((2, N))
    P_views[0, aapl_idx]  =  1.0
    P_views[1, msft_idx]  =  1.0
    P_views[1, googl_idx] = -1.0

    q_views = np.array([Pi[aapl_idx] + 0.03, 0.02])
    Omega_v = np.diag([view1_conf_slider.value, view2_conf_slider.value])

    mu_bar = posterior_mean(Sigma, Pi, P_views, q_views, Omega_v, tau=tau)

    w_bl  = tangency_weights(mu_bar, Sigma, rf=RF_BL)
    w_raw = tangency_weights(mu,     Sigma, rf=RF_BL)

    print("μ̄ (BL) vs μ (sample) vs Π (implied):")
    for t, mb, ms, pi in zip(tickers, mu_bar, mu, Pi):
        print(f"  {t:6s}: BL={mb:.2%}  Sample={ms:.2%}  Π={pi:.2%}")
    return tau, aapl_idx, msft_idx, googl_idx, P_views, q_views, Omega_v, mu_bar, w_bl, w_raw


@app.cell
def _(plt, np, tickers, N, mu, mu_bar, Pi, w_bl, w_raw, w_mkt,
      port_return, port_vol, Sigma, RF_BL, tangency_weights, posterior_mean, P_views, q_views):
    fig_s12, axes12 = plt.subplots(1, 3, figsize=(16, 5))
    x12 = np.arange(N)
    width12 = 0.28

    axes12[0].bar(x12 - width12, Pi,     width12, label="Π (implied)", color="gray",     alpha=0.8)
    axes12[0].bar(x12,           mu_bar, width12, label="μ̄ (BL)",      color="steelblue", alpha=0.8)
    axes12[0].bar(x12 + width12, mu,     width12, label="μ̂ (sample)",  color="tomato",   alpha=0.8)
    axes12[0].set_xticks(x12)
    axes12[0].set_xticklabels(tickers, rotation=45, ha="right")
    axes12[0].set_ylabel("Annualised return")
    axes12[0].set_title("Return Forecasts: Implied vs BL vs Sample")
    axes12[0].legend(fontsize=8)

    axes12[1].bar(x12 - width12, w_mkt, width12, label="Market",      color="gray",     alpha=0.8)
    axes12[1].bar(x12,           w_bl,  width12, label="BL tangency", color="steelblue", alpha=0.8)
    axes12[1].bar(x12 + width12, w_raw, width12, label="Raw tangency", color="tomato",   alpha=0.8)
    axes12[1].axhline(0, color="black", linewidth=0.7)
    axes12[1].set_xticks(x12)
    axes12[1].set_xticklabels(tickers, rotation=45, ha="right")
    axes12[1].set_ylabel("Portfolio weight")
    axes12[1].set_title("Weights: Market vs BL vs Raw Tangency")
    axes12[1].legend(fontsize=8)

    # Views fade-out: as Omega -> inf, BL -> market
    omega_scales = np.logspace(-3, 3, 40)
    bl_weights_track = []
    for scale in omega_scales:
        Omega_sc = np.diag([float(scale), float(scale)])
        try:
            mb_sc = posterior_mean(Sigma, Pi, P_views, q_views, Omega_sc, tau=0.05)
            w_sc  = tangency_weights(mb_sc, Sigma, rf=RF_BL)
            bl_weights_track.append(w_sc)
        except Exception:
            bl_weights_track.append(np.full(N, np.nan))

    bl_weights_track = np.array(bl_weights_track)
    for i in range(N):
        axes12[2].semilogx(omega_scales, bl_weights_track[:, i], linewidth=1.2,
                            label=tickers[i] if i < 5 else "_")
    axes12[2].axhline(0, color="black", linewidth=0.5)
    axes12[2].set_xlabel("Ω scale  (→ ∞ means no views → implied market)")
    axes12[2].set_ylabel("BL tangency weight")
    axes12[2].set_title("Views Fade Out: BL → Market Portfolio")
    axes12[2].legend(fontsize=7)
    axes12[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_s12
    return (fig_s12, axes12, x12, width12, omega_scales, bl_weights_track)


@app.cell
def _(mo, np, Sigma, Pi, P_views, q_views, posterior_mean):
    Omega_large = np.eye(2) * 1e6
    mu_noview = posterior_mean(Sigma, Pi, P_views, q_views, Omega_large, tau=0.05)
    diff_to_pi = float(np.max(np.abs(mu_noview - Pi)))
    mo.md(f"""
    **Validation — no-views limit (Ω → ∞):**
    max|μ̄ − Π| = `{diff_to_pi:.2e}` — {'✅ converges to Π' if diff_to_pi < 1e-4 else '⚠️ not converging'}

    As $\\Omega \\to \\infty$ (zero confidence in views), the posterior mean collapses to Π
    and the optimiser returns approximately the market portfolio.
    """)
    return Omega_large, mu_noview, diff_to_pi


if __name__ == "__main__":
    app.run()
