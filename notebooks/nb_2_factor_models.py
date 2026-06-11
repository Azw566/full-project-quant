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
    # Notebook 2 — Factor Models & Covariance Estimation
    ## Sessions 5–8: CAPM → Multi-Factor → PCA → Ledoit–Wolf

    **The goal:** cure the ill-conditioned $\hat\Sigma$ identified in Session 4.
    Factor models decompose risk into structured drivers; shrinkage regularises
    statistically.  Both repairs are complementary and used together in practice.
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
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    from data.loader import get_panel, get_bars
    from alloc.base import log_returns, sample_moments, gmv_weights, port_vol
    from risk.covariance import (
        condition_number, is_psd,
        sample_cov, ledoit_wolf, factor_cov, pca_cov,
        marchenko_pastur_threshold,
    )
    from risk.capm import capm_betas, rolling_betas
    return (
        np, pd, plt, LedoitWolf, PCA, Path,
        get_panel, get_bars, log_returns, sample_moments, gmv_weights, port_vol,
        condition_number, is_psd, sample_cov, ledoit_wolf, factor_cov, pca_cov,
        marchenko_pastur_threshold, capm_betas, rolling_betas,
    )


@app.cell
def _(get_panel, get_bars, log_returns, sample_moments, Path):
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

    # SPY as market proxy
    spy_bars = get_bars("SPY", start="2020-01-01", end="2024-01-01", cache_dir=CACHE)
    import numpy as _np
    spy_close = spy_bars["close"]
    spy_ret = _np.log(spy_close / spy_close.shift(1)).dropna()
    spy_ret.name = "SPY"
    combined = returns.join(spy_ret, how="inner")
    r_mkt = combined["SPY"].values
    return (prices, returns, mu, Sigma, tickers, N, TICKERS, CACHE,
            spy_bars, spy_ret, combined, r_mkt)


# ═══════════════════════════════════════════════════════════════
# SESSION 5 — CAPM and the Single-Factor Beta
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 5 — CAPM and the Single-Factor Regression

    Regress each asset's excess return on the market's:

    $$r_i - r_f = \alpha_i + \beta_i(r_m - r_f) + \varepsilon_i, \qquad
      \beta_i = \frac{\mathrm{Cov}(r_i, r_m)}{\mathrm{Var}(r_m)}$$

    Risk decomposes into **systematic** ($\beta_i^2\sigma_m^2$) and **idiosyncratic** ($\mathrm{Var}(\varepsilon_i)$).
    """)
    return


@app.cell
def _(capm_betas, combined, tickers, r_mkt):
    capm = capm_betas(combined[tickers], r_mkt, rf_daily=0.04 / 252)
    betas     = capm["betas"]
    alphas    = capm["alphas"]
    r2s       = capm["r2s"]
    sys_vars  = capm["sys_vars"]
    idio_vars = capm["idio_vars"]
    return betas, alphas, r2s, sys_vars, idio_vars, capm


@app.cell
def _(plt, betas, r2s, sys_vars, idio_vars, tickers):
    fig_s5, axes5 = plt.subplots(1, 3, figsize=(15, 5))

    axes5[0].barh(tickers, betas, color=["red" if b < 1 else "steelblue" for b in betas])
    axes5[0].axvline(1, color="black", linewidth=1, linestyle="--", label="β=1")
    axes5[0].set_xlabel("Beta β")
    axes5[0].set_title("CAPM Beta (vs SPY)")
    axes5[0].legend()

    axes5[1].barh(tickers, r2s, color="mediumseagreen")
    axes5[1].set_xlabel("R² (systematic fraction of variance)")
    axes5[1].set_title("Fraction of Variance Explained by Market")
    axes5[1].set_xlim(0, 1)

    axes5[2].barh(tickers, sys_vars, label="Systematic $\\beta^2\\sigma_m^2$", color="steelblue")
    axes5[2].barh(tickers, idio_vars, left=sys_vars, label="Idiosyncratic $\\sigma_\\varepsilon^2$", color="tomato")
    axes5[2].set_xlabel("Annualised variance")
    axes5[2].set_title("Risk Decomposition")
    axes5[2].legend()

    plt.tight_layout()
    fig_s5
    return fig_s5, axes5


@app.cell
def _(mo, np, betas, alphas, r2s, sys_vars, idio_vars, tickers, Sigma):
    import pandas as _pd
    rows_capm = []
    for i, t in enumerate(tickers):
        rows_capm.append({
            "Ticker": t,
            "β": f"{betas[i]:.3f}",
            "α (ann.)": f"{alphas[i]:.2%}",
            "R²": f"{r2s[i]:.2%}",
            "Systematic": f"{sys_vars[i]:.4f}",
            "Idiosyncratic": f"{idio_vars[i]:.4f}",
            "Σ diag": f"{np.diag(Sigma)[i]:.4f}",
        })
    df_capm = _pd.DataFrame(rows_capm)
    mo.md(f"""
    **CAPM decomposition — Systematic + Idiosyncratic ≈ Σ diagonal:**

    {df_capm.to_markdown(index=False)}
    """)
    return df_capm, rows_capm


@app.cell
def _(mo):
    mo.md("### Rolling Beta — how β drifts over time")
    return


@app.cell
def _(mo):
    roll_window_slider = mo.ui.slider(60, 252, step=21, value=126,
                                       label="Rolling window (trading days)")
    roll_window_slider
    return (roll_window_slider,)


@app.cell
def _(plt, combined, tickers, r_mkt, rolling_betas, roll_window_slider):
    window = roll_window_slider.value
    roll_b = rolling_betas(combined[tickers], r_mkt, window=window, rf_daily=0.04 / 252)

    fig_roll, ax_roll = plt.subplots(figsize=(12, 5))
    for i, tick in enumerate(tickers[:5]):
        ax_roll.plot(combined.index, roll_b[:, i], linewidth=1.2, label=tick)
    ax_roll.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax_roll.set_ylabel("Rolling β")
    ax_roll.set_title(f"Rolling Beta vs SPY  (window = {window} days)")
    ax_roll.legend(fontsize=9)
    ax_roll.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_roll
    return fig_roll, ax_roll, window, roll_b


# ═══════════════════════════════════════════════════════════════
# SESSION 6 — Multi-Factor Model Σ = B F B^T + D
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 6 — Multi-Factor Models & the Key Formula ⚠️

    $$\boxed{\Sigma = BFB^T + D}$$

    - $B$ is $N\times k$ (factor loadings), $F$ is $k\times k$, $D$ is diagonal
    - Parameter count drops from $O(N^2)$ to $O(Nk)$
    - $BFB^T + D$ is **well-conditioned by construction**

    We use Fama–French 3 factors (Mkt-RF, SMB, HML) when available,
    or fall back to synthetic factors.
    """)
    return


@app.cell
def _(pd, returns, Path):
    try:
        import pandas_datareader.data as web
        ff_raw = web.DataReader(
            "F-F_Research_Data_Factors_daily", "famafrench",
            start="2020-01-01", end="2024-01-01",
        )
        ff = ff_raw[0] / 100.0
        ff.index = pd.to_datetime(ff.index)
        ff_ok = True
    except Exception as e:
        print(f"Fama-French download failed ({e}); using synthetic factors for demo")
        import numpy as _np
        ff = pd.DataFrame(
            _np.random.default_rng(0).normal(0, 0.01, (len(returns), 4)),
            index=returns.index,
            columns=["Mkt-RF", "SMB", "HML", "RF"],
        )
        ff_ok = False
    return ff, ff_ok


@app.cell
def _(np, returns, ff, tickers, N, factor_cov, condition_number,
      gmv_weights, port_vol, Sigma):
    common_idx = returns.index.intersection(ff.index)
    R_aligned  = returns.loc[common_idx].values
    F_mat      = ff.loc[common_idx, ["Mkt-RF", "SMB", "HML"]].values

    Sigma_factor = factor_cov(R_aligned, F_mat, annualize=True)

    kappa_sample = condition_number(Sigma)
    kappa_factor = condition_number(Sigma_factor)

    w_gmv_sample = gmv_weights(Sigma)
    w_gmv_factor = gmv_weights(Sigma_factor)

    print(f"Sample Σ  κ={kappa_sample:>10.1f}   GMV vol={port_vol(w_gmv_sample, Sigma):.2%}")
    print(f"Factor Σ  κ={kappa_factor:>10.1f}   GMV vol={port_vol(w_gmv_factor, Sigma_factor):.2%}")
    print(f"Factor model is {kappa_sample/kappa_factor:.0f}× better conditioned")
    return (common_idx, R_aligned, F_mat, Sigma_factor, kappa_sample, kappa_factor,
            w_gmv_sample, w_gmv_factor)


@app.cell
def _(np, plt, tickers, N, w_gmv_sample, w_gmv_factor,
      kappa_sample, kappa_factor, R_aligned, F_mat):
    # Compute B for heatmap
    T_ff = R_aligned.shape[0]
    k_ff = F_mat.shape[1]
    ones_col = np.ones((T_ff, 1))
    X_ff = np.hstack([ones_col, F_mat])
    coefs_ff = np.linalg.inv(X_ff.T @ X_ff) @ X_ff.T @ R_aligned
    B_loadings = coefs_ff[1:].T  # (N, k)

    fig_s6, axes6 = plt.subplots(1, 3, figsize=(16, 5))

    im = axes6[0].imshow(B_loadings, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axes6[0].set_xticks(range(3))
    axes6[0].set_xticklabels(["Mkt-RF", "SMB", "HML"])
    axes6[0].set_yticks(range(len(tickers)))
    axes6[0].set_yticklabels(tickers)
    axes6[0].set_title("Factor Loadings B")
    plt.colorbar(im, ax=axes6[0])

    axes6[1].bar(["Sample Σ", "Factor Σ"], [kappa_sample, kappa_factor],
                  color=["tomato", "steelblue"])
    axes6[1].set_ylabel("Condition number κ(Σ)")
    axes6[1].set_title("Condition Number Comparison\n(lower = better)")
    axes6[1].set_yscale("log")

    x6 = np.arange(N)
    width6 = 0.35
    axes6[2].bar(x6 - width6/2, w_gmv_sample, width6, label="Sample GMV", color="tomato", alpha=0.8)
    axes6[2].bar(x6 + width6/2, w_gmv_factor, width6, label="Factor GMV", color="steelblue", alpha=0.8)
    axes6[2].set_xticks(x6)
    axes6[2].set_xticklabels(tickers, rotation=45, ha="right")
    axes6[2].axhline(0, color="black", linewidth=0.7)
    axes6[2].set_ylabel("Portfolio weight")
    axes6[2].set_title("GMV Weights: Sample vs Factor Σ")
    axes6[2].legend()

    plt.tight_layout()
    fig_s6
    return fig_s6, axes6, B_loadings


# ═══════════════════════════════════════════════════════════════
# SESSION 7 — PCA + Marchenko-Pastur
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 7 — Statistical Factors via PCA

    Eigendecompose $\Sigma = Q\Lambda Q^T$.  Top eigenvectors are statistical factors.

    **Marchenko–Pastur** tells you which eigenvalues are noise. For a random matrix
    with $N$ assets and $T$ observations:

    $$\lambda_+ = \sigma^2\!\left(1 + \sqrt{N/T}\right)^2$$

    Eigenvalues above $\lambda_+$ carry signal; below it is noise.
    """)
    return


@app.cell
def _(mo):
    k_pca_slider = mo.ui.slider(1, 9, step=1, value=3, label="PCA components k")
    k_pca_slider
    return (k_pca_slider,)


@app.cell
def _(np, plt, returns, Sigma, N, k_pca_slider, condition_number,
      marchenko_pastur_threshold, pca_cov):
    k_pca = k_pca_slider.value
    R_pca = returns.values
    T_pca = R_pca.shape[0]

    S_daily = np.cov(R_pca.T)
    eigvals_all, eigvecs_all = np.linalg.eigh(S_daily)
    idx_desc = np.argsort(eigvals_all)[::-1]
    eigvals_desc = eigvals_all[idx_desc]
    eigvecs_desc = eigvecs_all[:, idx_desc]
    explained_ratio = eigvals_desc / eigvals_desc.sum()
    eigenvalues_ann = eigvals_desc * 252

    # MP threshold (on annualised scale)
    sigma2_mp = float(np.mean(np.diag(S_daily)))
    lambda_plus = marchenko_pastur_threshold(N, T_pca, sigma2=sigma2_mp) * 252

    # Denoised: replace noise bulk eigenvalues with their mean
    bulk_mean = float(np.mean(eigenvalues_ann[eigenvalues_ann < lambda_plus])) \
                if any(eigenvalues_ann < lambda_plus) else float(lambda_plus)
    lam_clean = np.where(eigenvalues_ann >= lambda_plus, eigenvalues_ann, bulk_mean)
    Sigma_cleaned = eigvecs_desc @ np.diag(lam_clean) @ eigvecs_desc.T

    Sigma_pca = pca_cov(R_pca, n_components=k_pca)

    kappa_sample_p = condition_number(Sigma)
    kappa_pca      = condition_number(Sigma_pca)
    kappa_clean    = condition_number(Sigma_cleaned)

    fig_s7, axes7 = plt.subplots(1, 3, figsize=(16, 5))

    axes7[0].bar(range(1, N + 1), explained_ratio * 100, color="steelblue", alpha=0.8)
    axes7[0].set_xlabel("Principal Component")
    axes7[0].set_ylabel("Variance explained (%)")
    axes7[0].set_title("PCA Scree Plot")

    axes7[1].bar(range(1, N + 1), eigenvalues_ann, color="steelblue", alpha=0.7, label="Eigenvalues")
    axes7[1].axhline(lambda_plus, color="red", linewidth=2, linestyle="--",
                      label=f"MP upper edge λ₊={lambda_plus:.3f}")
    axes7[1].axvline(k_pca + 0.5, color="orange", linewidth=1.5, linestyle=":",
                      label=f"k={k_pca} kept")
    axes7[1].set_xlabel("Eigenvalue rank")
    axes7[1].set_ylabel("Annualised eigenvalue")
    axes7[1].set_title("Marchenko–Pastur Noise Floor")
    axes7[1].legend(fontsize=8)

    axes7[2].bar(
        ["Sample Σ", f"PCA-{k_pca}", "MP-cleaned"],
        [kappa_sample_p, kappa_pca, kappa_clean],
        color=["tomato", "steelblue", "mediumseagreen"],
    )
    axes7[2].set_ylabel("Condition number κ(Σ)")
    axes7[2].set_title("Condition Number After Denoising")
    axes7[2].set_yscale("log")

    plt.tight_layout()
    fig_s7
    return (fig_s7, axes7, k_pca, R_pca, T_pca, S_daily,
            eigvals_desc, eigvecs_desc, explained_ratio, eigenvalues_ann,
            sigma2_mp, lambda_plus, bulk_mean, lam_clean, Sigma_cleaned,
            Sigma_pca, kappa_sample_p, kappa_pca, kappa_clean)


# ═══════════════════════════════════════════════════════════════
# SESSION 8 — Ledoit–Wolf Shrinkage
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Session 8 — Ledoit–Wolf Shrinkage

    Shrink the sample covariance $S$ toward a structured target $F^*$ (scaled identity):

    $$\hat\Sigma = \delta F^* + (1-\delta)S$$

    Ledoit–Wolf analytically derives $\delta^*$ that minimises expected Frobenius distance
    to the true $\Sigma$ — **no cross-validation needed**.
    """)
    return


@app.cell
def _(np, LedoitWolf, returns, Sigma, condition_number, gmv_weights, ledoit_wolf):
    R_lw = returns.values
    N_lw = R_lw.shape[1]
    T_lw = R_lw.shape[0]

    lw_obj = LedoitWolf()
    lw_obj.fit(R_lw)
    delta_lw = float(lw_obj.shrinkage_)
    Sigma_lw  = lw_obj.covariance_ * 252

    kappa_lw = condition_number(Sigma_lw)
    kappa_s  = condition_number(Sigma)

    print(f"sklearn δ* = {delta_lw:.4f}   κ(sample)={kappa_s:.1f}   κ(LW)={kappa_lw:.1f}")
    return R_lw, N_lw, T_lw, lw_obj, delta_lw, Sigma_lw, kappa_lw, kappa_s


@app.cell
def _(mo, delta_lw, kappa_s, kappa_lw):
    mo.md(f"""
    | Estimator | δ* | κ(Σ) |
    |-----------|-----|------|
    | Sample S  | —      | {kappa_s:.1f}  |
    | Ledoit–Wolf | {delta_lw:.4f} | {kappa_lw:.1f} |

    A smaller condition number means $\\Sigma^{{-1}}$ amplifies noise less —
    weights are more stable and the optimiser concentrates less into spurious positions.
    """)
    return


@app.cell
def _(plt, np, returns, Sigma, Sigma_lw, LedoitWolf):
    rng_lw = np.random.default_rng(123)
    N_sim_lw, T_sim_lw = 10, 120
    true_Sig_sim = np.eye(N_sim_lw) * 0.04 + 0.005
    true_Sig_sim += N_sim_lw * 0.001 * np.eye(N_sim_lw)

    frob_sample_list, frob_lw_list = [], []
    for _ in range(200):
        R_s = rng_lw.multivariate_normal(np.zeros(N_sim_lw), true_Sig_sim / 252, size=T_sim_lw)
        S_s = np.cov(R_s.T) * 252
        S_lw_s = LedoitWolf().fit(R_s).covariance_ * 252
        frob_sample_list.append(np.linalg.norm(S_s - true_Sig_sim, "fro"))
        frob_lw_list.append(np.linalg.norm(S_lw_s - true_Sig_sim, "fro"))

    D_std = np.diag(1 / np.sqrt(np.diag(Sigma_lw)))
    corr_lw   = D_std @ Sigma_lw @ D_std
    corr_samp = returns.corr().values

    fig_s8, axes8 = plt.subplots(1, 2, figsize=(12, 5))

    axes8[0].hist(frob_sample_list, bins=30, alpha=0.6,
                   label=f"Sample S  (mean={np.mean(frob_sample_list):.3f})", color="tomato")
    axes8[0].hist(frob_lw_list, bins=30, alpha=0.6,
                   label=f"LW shrinkage  (mean={np.mean(frob_lw_list):.3f})", color="steelblue")
    axes8[0].set_xlabel("Frobenius error ‖Σ̂ − Σ_true‖_F")
    axes8[0].set_ylabel("Count")
    axes8[0].set_title("Ledoit–Wolf Reduces Estimation Error\n(200 simulated trials)")
    axes8[0].legend()

    im8 = axes8[1].imshow(corr_lw - corr_samp, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes8[1].set_title("Correlation difference: LW − Sample\n(positive = LW shrinks toward zero)")
    axes8[1].set_xticks([])
    axes8[1].set_yticks([])
    plt.colorbar(im8, ax=axes8[1])

    plt.tight_layout()
    fig_s8
    return (fig_s8, axes8, rng_lw, frob_sample_list, frob_lw_list,
            corr_lw, corr_samp, D_std)


if __name__ == "__main__":
    app.run()
