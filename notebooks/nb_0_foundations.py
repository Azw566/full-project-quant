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
    # Notebook 0 — Foundations
    ## The two objects that drive everything: μ and Σ

    **The single organizing idea:** almost every portfolio construction method reduces
    to two objects — expected returns $\mu$ and the covariance matrix $\Sigma$.

    ---

    - A portfolio is a weight vector $w \in \mathbb{R}^N$ with $w^T\mathbf{1} = 1$.
    - Asset returns $r$ have mean $\mu = \mathbb{E}[r]$ and covariance $\Sigma = \mathrm{Cov}(r)$.
    - **Expected return:** $\mathbb{E}[R_p] = w^T\mu$ (linear in $w$)
    - **Variance:** $\mathrm{Var}(R_p) = w^T\Sigma w$ (quadratic form in $w$)

    ### The asymmetry that drives everything

    $\mu$ is nearly impossible to estimate reliably; $\Sigma$ is hard but tractable.
    Every algorithm in these notebooks is a reaction to that asymmetry.
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

    from data.loader import get_panel
    from alloc.base import (
        log_returns, sample_moments,
        port_return, port_vol, equal_weight,
    )
    from risk.covariance import is_psd, condition_number
    return (
        np, pd, plt, Path,
        get_panel, log_returns, sample_moments,
        port_return, port_vol, equal_weight,
        is_psd, condition_number,
    )


@app.cell
def _(get_panel, log_returns, Path):
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "JPM", "JNJ", "XOM", "PG", "V",
    ]
    CACHE = str(Path(__file__).parent.parent / "data" / "cache")
    prices = get_panel(TICKERS, start="2022-01-01", end="2024-01-01", cache_dir=CACHE)
    returns = log_returns(prices)
    tickers = list(prices.columns)
    n_assets = len(tickers)
    return prices, returns, tickers, n_assets, TICKERS, CACHE


@app.cell
def _(mo, prices, returns, tickers):
    mo.md(f"""
    **Universe:** {', '.join(tickers)}

    **Period:** {prices.index[0].date()} → {prices.index[-1].date()}
    **Trading days:** {len(returns)}
    **Assets:** {len(tickers)}
    """)
    return


@app.cell
def _(plt, returns, tickers):
    fig0, axes0 = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    returns.iloc[:, :5].plot(ax=axes0[0], alpha=0.75, linewidth=0.7)
    axes0[0].set_title("Daily Log Returns — first 5 assets")
    axes0[0].set_ylabel("Log return")
    axes0[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes0[0].legend(tickers[:5], fontsize=8)

    ((returns.iloc[:, :5] + 1).cumprod()).plot(ax=axes0[1], alpha=0.8)
    axes0[1].set_title("Cumulative Returns (growth of $1)")
    axes0[1].set_ylabel("Portfolio value")
    axes0[1].legend(tickers[:5], fontsize=8)

    plt.tight_layout()
    fig0
    return fig0, axes0


@app.cell
def _(mo):
    mo.md("## Estimate $\\hat{\\mu}$ and $\\hat{\\Sigma}$ (annualised)")
    return


@app.cell
def _(sample_moments, returns):
    mu, Sigma = sample_moments(returns, annualize=True)
    return mu, Sigma


@app.cell
def _(mo, np, mu, Sigma, tickers, is_psd, condition_number, port_vol, equal_weight, n_assets):
    eigvals_S = np.linalg.eigvalsh(Sigma)
    kappa = condition_number(Sigma)
    psd_ok = is_psd(Sigma)
    sym_ok = bool(np.allclose(Sigma, Sigma.T))

    w_eq = equal_weight(n_assets)
    eq_vol = port_vol(w_eq, Sigma)
    avg_vol = float(np.sqrt(np.diag(Sigma)).mean())
    div_benefit = avg_vol - eq_vol

    mo.md(f"""
    ## Validate $\\hat{{\\Sigma}}$

    | Property | Value |
    |----------|-------|
    | Symmetric | {'✅' if sym_ok else '❌'} |
    | PSD (all eigenvalues ≥ 0) | {'✅' if psd_ok else '❌'} |
    | Smallest eigenvalue | `{eigvals_S.min():.6f}` |
    | Largest eigenvalue | `{eigvals_S.max():.4f}` |
    | Condition number κ(Σ) | `{kappa:.1f}` |

    **Diversification check:**
    - Equal-weight portfolio vol: **{eq_vol:.2%}**
    - Average individual vol:     **{avg_vol:.2%}**
    - Diversification benefit:    **{div_benefit:.2%}**

    The gap *is* diversification — the cross-terms $w_i w_j \\Sigma_{{ij}}$ reduce
    total risk when assets are not perfectly correlated.
    """)
    return eigvals_S, kappa, psd_ok, sym_ok, w_eq, eq_vol, avg_vol, div_benefit


@app.cell
def _(plt, np, mu, Sigma, tickers):
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    vols_ind = np.sqrt(np.diag(Sigma))
    ax1.scatter(vols_ind, mu, s=90, zorder=5, color="steelblue")
    for i, t in enumerate(tickers):
        ax1.annotate(t, (vols_ind[i], mu[i]), textcoords="offset points",
                     xytext=(5, 3), fontsize=9)
    ax1.set_xlabel("Annualised Volatility σ")
    ax1.set_ylabel("Annualised Expected Return μ")
    ax1.set_title("Individual Assets — Risk vs Return")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1
    return fig1, ax1, vols_ind


@app.cell
def _(mo):
    mo.md(r"""
    ## Eigendecomposition of $\Sigma$

    $\Sigma = Q\Lambda Q^T$ where columns of $Q$ are eigenvectors (principal directions of risk).

    The **condition number** $\kappa(\Sigma) = \lambda_{\max}/\lambda_{\min}$ tells you how much
    $\Sigma^{-1}$ amplifies noise — a large $\kappa$ warns of instability in all algorithms that
    invert the covariance matrix (Markowitz, CAPM, Black–Litterman, …).
    """)
    return


@app.cell
def _(plt, np, Sigma):
    eigvals_desc = np.linalg.eigvalsh(Sigma)[::-1]
    cumvar = np.cumsum(eigvals_desc) / eigvals_desc.sum()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    axes2[0].bar(range(1, len(eigvals_desc) + 1), eigvals_desc,
                 color="steelblue", alpha=0.85, edgecolor="white")
    axes2[0].set_xlabel("Eigenvalue rank")
    axes2[0].set_ylabel("Eigenvalue λ")
    axes2[0].set_title("Eigenvalue Spectrum of Σ")

    axes2[1].plot(range(1, len(cumvar) + 1), cumvar, "o-", color="darkorange", linewidth=2)
    axes2[1].axhline(0.9, color="red", linestyle="--", alpha=0.6, label="90% threshold")
    axes2[1].set_xlabel("Number of components")
    axes2[1].set_ylabel("Cumulative variance explained")
    axes2[1].set_title("How Many Factors Explain the Variance?")
    axes2[1].set_ylim(0, 1.05)
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2
    return fig2, axes2, eigvals_desc, cumvar


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive — Weight a portfolio and see its risk/return

    Move the slider to change how much of the budget goes to the first asset.
    The rest is split equally across the remaining assets.
    """)
    return


@app.cell
def _(mo, n_assets, tickers):
    w0_slider = mo.ui.slider(
        0.0, 1.0, step=0.02, value=round(1.0 / n_assets, 2),
        label=f"Weight on {tickers[0]}"
    )
    w0_slider
    return (w0_slider,)


@app.cell
def _(mo, np, mu, Sigma, tickers, n_assets, w0_slider, port_return, port_vol):
    w0 = w0_slider.value
    remaining = (1.0 - w0) / (n_assets - 1)
    w_custom = np.array([w0] + [remaining] * (n_assets - 1))

    ret_custom = port_return(w_custom, mu)
    vol_custom = port_vol(w_custom, Sigma)

    rows_tbl = [f"| {tickers[i]} | {w_custom[i]:.2%} |" for i in range(n_assets)]

    mo.md(f"""
    | Asset | Weight |
    |-------|--------|
    {chr(10).join(rows_tbl)}

    **Portfolio expected return:** {ret_custom:.2%}
    **Portfolio volatility:** {vol_custom:.2%}
    """)
    return w0, remaining, w_custom, ret_custom, vol_custom


if __name__ == "__main__":
    app.run()
