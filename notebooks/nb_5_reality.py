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
    # Notebook 5 — Reality
    ## Session 13: Walk-Forward Backtesting Without Fooling Yourself

    **This is where careers are made or broken.**
    The math is easy. Not lying to yourself is the decisive, hard skill.

    ---

    | Bias | Fix |
    |------|-----|
    | Look-ahead | Rolling window — estimation window never touches future data |
    | Overfitting | Out-of-sample validation; multiple-hypothesis discount |
    | Transaction costs | Penalise turnover; `fee_bps` charged on every rebalance |
    | Regime dependence | Test across market regimes (multiple decades) |

    ---

    This notebook uses **fullproject's production infrastructure** directly:
    - `strategy.rebalance.generate_weights` — causal weight generation
    - `backtest.multiasset.run` — event-driven multi-asset backtester

    The same code path that powers live trading powers this research.
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
    from alloc.base import log_returns, equal_weight, gmv_weights
    from alloc.risk_parity import erc_weights
    from risk.covariance import ledoit_wolf, sample_cov
    from strategy.rebalance import generate_weights
    from backtest.multiasset import run as bt_run
    return (
        np, pd, plt, Path,
        get_panel, log_returns, equal_weight, gmv_weights, erc_weights,
        ledoit_wolf, sample_cov, generate_weights, bt_run,
    )


@app.cell
def _(get_panel, Path):
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "JPM", "JNJ", "XOM", "PG", "V",
    ]
    CACHE = str(Path(__file__).parent.parent / "data" / "cache")
    prices = get_panel(TICKERS, start="2015-01-01", end="2024-12-31", cache_dir=CACHE)
    tickers = list(prices.columns)
    N = len(tickers)
    print(f"{len(prices)} trading days, {N} assets, {prices.index[0].date()} – {prices.index[-1].date()}")
    return prices, tickers, N, TICKERS, CACHE


@app.cell
def _(mo):
    lookback_slider  = mo.ui.slider(63, 504, step=21, value=252,
                                     label="Lookback window (trading days)")
    rebalance_slider = mo.ui.slider(5, 63, step=5, value=21,
                                     label="Rebalance frequency (trading days)")
    fee_slider = mo.ui.slider(0, 30, step=5, value=10,
                               label="Fee (basis points)")
    mo.hstack([lookback_slider, rebalance_slider, fee_slider])
    return lookback_slider, rebalance_slider, fee_slider


@app.cell
def _(np, prices, equal_weight, gmv_weights, erc_weights,
      ledoit_wolf, generate_weights, bt_run, N,
      lookback_slider, rebalance_slider, fee_slider):
    LOOKBACK  = lookback_slider.value
    REBALANCE = rebalance_slider.value
    FEE       = fee_slider.value

    # Allocator wrappers (all follow the Allocator protocol: mu, Sigma -> w)
    def alloc_ew(mu, Sigma, **_):
        return equal_weight(Sigma.shape[0])

    def alloc_gmv(mu, Sigma, **_):
        return gmv_weights(Sigma)

    # erc_weights already conforms: erc_weights(mu, Sigma, **_)

    print("Computing weights (this may take a minute for risk parity) …")

    w_ew  = generate_weights(prices, alloc_ew,  ledoit_wolf, LOOKBACK, REBALANCE)
    w_gmv = generate_weights(prices, alloc_gmv, ledoit_wolf, LOOKBACK, REBALANCE)
    w_erc = generate_weights(prices, erc_weights, ledoit_wolf, LOOKBACK, REBALANCE)

    print("Running backtests …")
    res_ew  = bt_run(prices, w_ew,  fee_bps=FEE)
    res_gmv = bt_run(prices, w_gmv, fee_bps=FEE)
    res_erc = bt_run(prices, w_erc, fee_bps=FEE)
    print("Done.")

    return (LOOKBACK, REBALANCE, FEE, alloc_ew, alloc_gmv,
            w_ew, w_gmv, w_erc, res_ew, res_gmv, res_erc)


@app.cell
def _(mo, np, res_ew, res_gmv, res_erc):
    import pandas as _pd

    def _perf(res_df):
        r = res_df["net_return"].values
        ann_ret = float(np.mean(r) * 252)
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
        sharpe  = (ann_ret - 0.04) / ann_vol if ann_vol > 0 else float("nan")
        cum = np.cumprod(1 + r)
        rolling_max = np.maximum.accumulate(cum)
        max_dd = float(((cum - rolling_max) / rolling_max).min())
        to = float(res_df["turnover"].mean())
        return {
            "Ann. Return":  f"{ann_ret:.2%}",
            "Ann. Vol":     f"{ann_vol:.2%}",
            "Sharpe":       f"{sharpe:.3f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Avg Turnover": f"{to:.3f}",
        }

    rows_perf = [
        {"Strategy": "Equal Weight",  **_perf(res_ew)},
        {"Strategy": "Min-Variance",  **_perf(res_gmv)},
        {"Strategy": "Risk Parity",   **_perf(res_erc)},
    ]
    df_perf = _pd.DataFrame(rows_perf).set_index("Strategy")

    mo.md(f"""
    ## Out-of-Sample Performance Summary

    {df_perf.to_markdown()}

    ---

    **Honest result:** Equal-weight or risk parity often beats "optimised" min-variance OOS.
    The culprit is exactly what Session 4 showed — estimation error turns optimisation into
    error-maximisation.  Understanding *why* is the synthesis of the whole course.
    """)
    return df_perf, rows_perf, _perf


@app.cell
def _(plt, np, pd, res_ew, res_gmv, res_erc):
    oos_dates_ew  = res_ew.index
    oos_dates_gmv = res_gmv.index
    oos_dates_erc = res_erc.index

    cum_ew  = np.cumprod(1 + res_ew["net_return"].values)
    cum_gmv = np.cumprod(1 + res_gmv["net_return"].values)
    cum_erc = np.cumprod(1 + res_erc["net_return"].values)

    fig_bt, axes_bt = plt.subplots(3, 1, figsize=(13, 12))

    # Cumulative returns
    ax = axes_bt[0]
    ax.plot(oos_dates_ew,  cum_ew,  label="Equal Weight", linewidth=1.5, color="gray")
    ax.plot(oos_dates_gmv, cum_gmv, label="Min-Variance",  linewidth=1.5, color="steelblue")
    ax.plot(oos_dates_erc, cum_erc, label="Risk Parity",   linewidth=1.5, color="darkorange")
    ax.set_ylabel("Cumulative Return (growth of $1)")
    ax.set_title("Walk-Forward Backtest — Cumulative Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drawdowns
    ax = axes_bt[1]
    for rets, dates, label, color in [
        (res_ew["net_return"].values,  oos_dates_ew,  "Equal Weight", "gray"),
        (res_gmv["net_return"].values, oos_dates_gmv, "Min-Variance",  "steelblue"),
        (res_erc["net_return"].values, oos_dates_erc, "Risk Parity",   "darkorange"),
    ]:
        cum = np.cumprod(1 + rets)
        rolling_max = np.maximum.accumulate(cum)
        dd = (cum - rolling_max) / rolling_max
        ax.fill_between(dates, dd, 0, alpha=0.4, color=color, label=label)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rolling 1-year Sharpe
    ax = axes_bt[2]
    roll_w = 252
    for rets, dates, label, color in [
        (res_ew["net_return"].values,  oos_dates_ew,  "Equal Weight", "gray"),
        (res_gmv["net_return"].values, oos_dates_gmv, "Min-Variance",  "steelblue"),
        (res_erc["net_return"].values, oos_dates_erc, "Risk Parity",   "darkorange"),
    ]:
        r_s = pd.Series(rets, index=dates)
        roll_ret = r_s.rolling(roll_w).mean() * 252
        roll_vol = r_s.rolling(roll_w).std() * np.sqrt(252)
        roll_sharpe = (roll_ret - 0.04) / roll_vol
        ax.plot(dates, roll_sharpe, label=label, linewidth=1.2, color=color)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Rolling 1-yr Sharpe")
    ax.set_title("Rolling Sharpe Ratio (1-year window)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_bt
    return (fig_bt, axes_bt, oos_dates_ew, oos_dates_gmv, oos_dates_erc,
            cum_ew, cum_gmv, cum_erc)


@app.cell
def _(plt, np, w_gmv, w_erc, prices, LOOKBACK, tickers, N):
    fig_wt, axes_wt = plt.subplots(1, 2, figsize=(14, 5))

    for ax_wt, weights_df, title_wt in zip(
        axes_wt,
        [w_gmv, w_erc],
        ["Min-Variance Weights Over Time", "Risk Parity Weights Over Time"],
    ):
        # Sample every ~20 rows to keep the plot readable
        stride = max(1, len(weights_df) // 300)
        wdf = weights_df.iloc[::stride]

        for i, tick in enumerate(tickers):
            ax_wt.plot(wdf.index, wdf[tick].values, linewidth=1.0,
                       label=tick if i < 6 else "_", alpha=0.85)
        ax_wt.axhline(1/N, color="black", linewidth=0.8, linestyle="--",
                       label="Equal weight")
        ax_wt.set_ylabel("Portfolio weight")
        ax_wt.set_title(title_wt)
        ax_wt.legend(fontsize=7, ncol=2)
        ax_wt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_wt
    return fig_wt, axes_wt


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Synthesis — The Argument of the Course

    1. **Notebooks 0–1:** Portfolio construction reduces to $\mu$ and $\Sigma$.
    2. **Notebook 1 (Session 4):** Raw Markowitz is an *error-maximiser*.  $\hat\mu$ is the culprit.
    3. **Notebook 2:** Fix $\Sigma$ via factor models or shrinkage — well-conditioned covariance
       estimates produce sane weights.
    4. **Notebook 3:** Risk is a tail phenomenon.  ES is coherent; VaR is not.
       ES is optimisable via the Rockafellar–Uryasev reformulation.
    5. **Notebook 4:** Skip $\mu$ (risk parity) or replace it with a Bayesian posterior
       anchored to the market (Black–Litterman).
    6. **Notebook 5 (this one):** Out-of-sample, simple methods often win.
       The reason is Session 4.  Knowing *why* is the insight that matters.

    ---

    **Production link:** the weights computed above were generated by `strategy/rebalance.py`
    and run through `backtest/multiasset.py` — the same modules used in live trading.
    """)
    return


if __name__ == "__main__":
    app.run()
