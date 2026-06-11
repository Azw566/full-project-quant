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
    # Notebook 3 — Risk Measures
    ## Sessions 9–10: Value-at-Risk → Expected Shortfall

    **The question:** given a portfolio, what is the distribution of losses?

    VaR answers a single quantile question; ES answers the harder question of
    *how bad* the tail is.  The difference matters enormously in practice —
    both theoretically (coherence) and practically (tail risk under fat tails).
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
    from scipy import stats

    from data.loader import get_panel
    from alloc.base import log_returns, sample_moments, port_return, port_vol, equal_weight
    from risk.var_cvar import (
        var_parametric, var_historical, var_mc,
        es_parametric, es_historical,
        min_cvar_weights,
    )
    return (
        np, pd, plt, stats, Path,
        get_panel, log_returns, sample_moments, port_return, port_vol, equal_weight,
        var_parametric, var_historical, var_mc,
        es_parametric, es_historical, min_cvar_weights,
    )


@app.cell
def _(get_panel, log_returns, sample_moments, equal_weight, Path):
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
    w_eq = equal_weight(N)
    return prices, returns, mu, Sigma, tickers, N, w_eq, TICKERS, CACHE


@app.cell
def _(np, returns, w_eq, port_return, port_vol, mu, Sigma):
    port_ret_series = (returns.values @ w_eq)
    port_losses = -port_ret_series

    mu_p_daily  = float(np.mean(port_ret_series))
    sig_p_daily = float(np.std(port_ret_series, ddof=1))
    mu_p_ann    = port_return(w_eq, mu)
    sig_p_ann   = port_vol(w_eq, Sigma)

    print(f"Portfolio: μ={mu_p_ann:.2%}/yr  σ={sig_p_ann:.2%}/yr")
    print(f"Daily:     μ={mu_p_daily:.4%}    σ={sig_p_daily:.4%}")
    return port_ret_series, port_losses, mu_p_daily, sig_p_daily, mu_p_ann, sig_p_ann


# ═══════════════════════════════════════════════════════════════
# SESSION 9 — Value-at-Risk: Three Methods
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 9 — Value-at-Risk: Three Methods

    Fix a confidence level $c$, tail probability $\alpha = 1 - c$.
    **VaR** is the loss threshold exceeded with probability $\alpha$:
    $P(L > \mathrm{VaR}_c) = \alpha$

    Three estimators:
    1. **Parametric (Gaussian):** $\mathrm{VaR}_c = -\mu_p + \sigma_p \Phi^{-1}(c)$
    2. **Historical:** empirical $\alpha$-quantile of realised losses
    3. **Monte Carlo:** simulate returns, take empirical quantile

    The three methods **agree under Gaussian assumptions** and **diverge under fat tails**.
    """)
    return


@app.cell
def _(mo):
    conf_slider = mo.ui.slider(0.90, 0.99, step=0.01, value=0.95,
                                label="Confidence level c")
    conf_slider
    return (conf_slider,)


@app.cell
def _(np, stats, mu_p_daily, sig_p_daily, port_losses, conf_slider,
      var_parametric, var_historical, var_mc):
    c = conf_slider.value
    alpha_tail = 1.0 - c

    v_param = var_parametric(mu_p_daily, sig_p_daily, c)
    v_hist  = var_historical(port_losses, c)
    v_mc    = var_mc(mu_p_daily, sig_p_daily, c, n_sims=200_000, seed=7)

    # Student-t parametric (fat tails)
    dof = 4
    t_scale = sig_p_daily * np.sqrt((dof - 2) / dof)
    v_t = float(-mu_p_daily + t_scale * stats.t.ppf(c, df=dof))

    print(f"c={c:.0%}  Gaussian:{v_param:.4%}  Historical:{v_hist:.4%}  MC:{v_mc:.4%}  t(4):{v_t:.4%}")
    return c, alpha_tail, v_param, v_hist, v_mc, v_t, dof, t_scale


@app.cell
def _(plt, port_losses, v_param, v_hist, v_mc, v_t, c):
    fig_s9, ax_s9 = plt.subplots(figsize=(10, 5))
    ax_s9.hist(port_losses, bins=80, density=True, alpha=0.55,
               color="steelblue", label="Historical daily losses")

    for var_val, label_v, color_v in [
        (v_param, f"Parametric (Gaussian) {v_param:.3%}", "red"),
        (v_hist,  f"Historical {v_hist:.3%}",              "green"),
        (v_mc,    f"Monte Carlo {v_mc:.3%}",                "orange"),
        (v_t,     f"Parametric (Student-t ν=4) {v_t:.3%}", "purple"),
    ]:
        ax_s9.axvline(var_val, linewidth=2, color=color_v, label=label_v)

    ax_s9.set_xlabel("Daily Loss (negative return)")
    ax_s9.set_ylabel("Density")
    ax_s9.set_title(f"VaR at {c:.0%} confidence — equal-weight portfolio")
    ax_s9.legend(fontsize=8)
    ax_s9.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_s9
    return fig_s9, ax_s9


@app.cell
def _(mo, stats, port_losses):
    kurt_val = float(stats.kurtosis(port_losses))
    skew_val = float(stats.skew(port_losses))
    _, p_jb  = stats.jarque_bera(port_losses)
    mo.md(f"""
    ### Are the tails Gaussian?

    | Statistic | Value | Interpretation |
    |-----------|-------|----------------|
    | Kurtosis (excess) | {kurt_val:.2f} | >0 means fatter tails than Gaussian |
    | Skewness | {skew_val:.2f} | Negative = left tail heavier |
    | Jarque–Bera p-value | {p_jb:.2e} | <0.05 rejects Gaussian |

    **Conclusion:** {'❌ Non-Gaussian — parametric VaR understates tail risk.' if p_jb < 0.05 else '✅ Cannot reject Gaussian.'}
    """)
    return kurt_val, skew_val, p_jb


# ═══════════════════════════════════════════════════════════════
# SESSION 10 — Expected Shortfall & Coherence
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 10 — Expected Shortfall (CVaR) and Coherence

    **Expected Shortfall:** $\mathrm{ES}_c = \mathbb{E}[L \mid L \geq \mathrm{VaR}_c]$

    Gaussian closed form: $\mathrm{ES}_c = -\mu_p + \sigma_p \dfrac{\phi(\Phi^{-1}(c))}{1-c}$

    ### Why ES is preferred: coherence

    - **VaR violates subadditivity** — it can reward concentration (dangerous!)
    - **ES is coherent** — diversification never increases ES

    ### ES is optimisable — Rockafellar–Uryasev LP

    $$\min_{w, \zeta}\; \zeta + \frac{1}{(1-c)T}\sum_t \max(L_t(w) - \zeta, 0)
    \quad\text{s.t.}\quad w\ge0,\; \mathbf{1}^Tw=1$$
    """)
    return


@app.cell
def _(np, mu_p_daily, sig_p_daily, port_losses, c,
      es_parametric, es_historical, var_historical):
    z_c = float(__import__("scipy").stats.norm.ppf(c))
    e_param = es_parametric(mu_p_daily, sig_p_daily, c)
    e_hist  = es_historical(port_losses, c)
    var_for_mc = var_historical(port_losses, c)

    rng_mc = __import__("numpy").random.default_rng(7)
    mc_l = -rng_mc.normal(mu_p_daily, sig_p_daily, 200_000)
    e_mc  = float(np.mean(mc_l[mc_l >= np.quantile(mc_l, c)]))

    print(f"ES at {c:.0%}: Param={e_param:.4%}  Historical={e_hist:.4%}  MC={e_mc:.4%}")
    return z_c, e_param, e_hist, e_mc, var_for_mc, rng_mc, mc_l


@app.cell
def _(plt, port_losses, var_for_mc, e_hist, e_param, e_mc, c):
    fig_es, ax_es = plt.subplots(figsize=(10, 5))
    ax_es.hist(port_losses, bins=80, density=True, alpha=0.5,
               color="steelblue", label="Daily losses")

    ax_es.axvline(var_for_mc, color="red",       linewidth=2, linestyle="--",
                   label=f"VaR (hist) = {var_for_mc:.3%}")
    ax_es.axvline(e_hist,     color="darkred",   linewidth=2,
                   label=f"ES (hist) = {e_hist:.3%}")
    ax_es.axvline(e_param,    color="orange",    linewidth=2, linestyle=":",
                   label=f"ES (param) = {e_param:.3%}")
    ax_es.axvline(e_mc,       color="darkorange", linewidth=1.5, linestyle="-.",
                   label=f"ES (MC) = {e_mc:.3%}")

    tail_mask = port_losses >= var_for_mc
    ax_es.hist(port_losses[tail_mask], bins=30, density=True, alpha=0.6,
               color="red", label=f"Tail ({tail_mask.mean():.1%} of losses)")

    ax_es.set_xlabel("Daily Loss")
    ax_es.set_ylabel("Density")
    ax_es.set_title(f"VaR vs ES at {c:.0%} confidence\n(ES = mean of the red tail)")
    ax_es.legend(fontsize=8)
    ax_es.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_es
    return fig_es, ax_es, tail_mask


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### VaR Subadditivity Violation — Two Defaultable Bonds

    Each bond: 4% default probability, loss = 1 on default.  Independent defaults.

    At 95% confidence (α = 5%):
    - P(bond A defaults) = 4% < 5% → VaR(A) = 0
    - P(bond B defaults) = 4% < 5% → VaR(B) = 0
    - P(at least one defaults) = 7.84% > 5% → VaR(combined) = 0.5
    """)
    return


@app.cell
def _(mo):
    p_none  = 0.96 ** 2
    p_one   = 2 * 0.04 * 0.96
    p_both  = 0.04 ** 2
    var_a_bond = 0.0
    var_b_bond = 0.0
    var_comb   = 0.5

    mo.md(f"""
    | Portfolio | VaR at 95% |
    |-----------|-----------|
    | Bond A alone | {var_a_bond:.1f} |
    | Bond B alone | {var_b_bond:.1f} |
    | Combined (equal weight) | **{var_comb:.1f}** |

    VaR(A+B) = **{var_comb}** > VaR(A) + VaR(B) = **{var_a_bond + var_b_bond}**

    **Subadditivity violated!**  ES does not have this problem:
    ES(A+B) ≤ ES(A) + ES(B) always holds.
    """)
    return p_none, p_one, p_both, var_a_bond, var_b_bond, var_comb


@app.cell
def _(np, returns, min_cvar_weights, N, c):
    w_mincvar = min_cvar_weights(returns, confidence=c)
    if w_mincvar is not None:
        R_sc = returns.values
        ew_losses = -(R_sc @ (np.ones(N) / N))
        v_ew = float(np.quantile(ew_losses, c))
        cvar_ew_val  = float(np.mean(ew_losses[ew_losses >= v_ew]))

        opt_losses = -(R_sc @ w_mincvar)
        v_opt = float(np.quantile(opt_losses, c))
        cvar_opt_val = float(np.mean(opt_losses[opt_losses >= v_opt]))
        print(f"Min-CVaR portfolio CVaR: {cvar_opt_val:.4%}  (vs EW: {cvar_ew_val:.4%})")
    else:
        cvar_opt_val = float("nan")
        cvar_ew_val = float("nan")
    return w_mincvar, cvar_opt_val, cvar_ew_val


@app.cell
def _(plt, np, w_mincvar, tickers, N, cvar_opt_val, cvar_ew_val):
    if w_mincvar is None:
        import marimo as _mo
        _mo.md("⚠️ CVaR optimisation did not converge.")
    else:
        fig_cv, axes_cv = plt.subplots(1, 2, figsize=(12, 5))

        axes_cv[0].barh(tickers, w_mincvar, color="steelblue")
        axes_cv[0].axvline(1/N, color="red", linestyle="--", label="Equal weight")
        axes_cv[0].set_xlabel("Portfolio weight")
        axes_cv[0].set_title(f"Min-CVaR Weights  (CVaR = {cvar_opt_val:.3%})")
        axes_cv[0].legend()

        bars_cv = axes_cv[1].bar(
            ["Equal-weight", "Min-CVaR"],
            [cvar_ew_val, cvar_opt_val],
            color=["tomato", "steelblue"],
        )
        axes_cv[1].set_ylabel("CVaR (daily)")
        axes_cv[1].set_title("CVaR: Equal-Weight vs Optimised")
        for bar_cv, v_cv in zip(bars_cv, [cvar_ew_val, cvar_opt_val]):
            axes_cv[1].text(bar_cv.get_x() + bar_cv.get_width()/2, v_cv + 0.0001,
                             f"{v_cv:.4%}", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        fig_cv
    return


if __name__ == "__main__":
    app.run()
