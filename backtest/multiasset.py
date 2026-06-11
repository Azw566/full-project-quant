"""
backtest/multiasset.py — Multi-asset event-driven backtester.

Generalises backtest/event_driven.py from a scalar position on one asset to a
continuous weight vector over N assets.

INVARIANT (see tests/test_multiasset_parity.py):
    When N = 1 and weights ∈ {0, 1}, this engine must reproduce the output of
    the legacy event_driven.run() to within 1e-9. If these numbers diverge,
    there is an accounting bug.

PUBLIC INTERFACE
────────────────
    run(panel, weights, fee_bps, ...) → pd.DataFrame
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


# ── Events ─────────────────────────────────────────────────────────────────────

@dataclass
class MarketEvent:
    timestamp: pd.Timestamp
    asset_returns: np.ndarray   # (N,) bar returns


@dataclass
class SignalEvent:
    timestamp: pd.Timestamp
    target_weights: np.ndarray  # (N,) target portfolio weights


@dataclass
class OrderEvent:
    timestamp: pd.Timestamp
    deltas: np.ndarray          # (N,) weight changes to execute


@dataclass
class FillEvent:
    timestamp: pd.Timestamp
    deltas: np.ndarray          # executed weight changes
    fee: float                  # fee_rate * Σ|Δwᵢ|  (turnover-based)
    slippage: float = 0.0


# ── Handlers ───────────────────────────────────────────────────────────────────

class _Portfolio:
    """
    Tracks positions (weight vector) and equity; converts SignalEvents into
    OrderEvents.  Risk overlays (vol targeting, drawdown circuit breaker) act
    on the aggregate weight vector and are compatible with the legacy overlays.
    """

    def __init__(
        self,
        n_assets: int,
        vol_target: float | None = None,
        vol_lookback: int = 20,
        periods_per_year: float = 252.0,
        max_drawdown: float | None = None,
        cooldown_bars: int = 20,
    ) -> None:
        self.positions: np.ndarray = np.zeros(n_assets)
        self.equity: float = 1.0

        self._vol_target = vol_target
        self._vol_lookback = vol_lookback
        self._periods_per_year = periods_per_year
        # Portfolio gross returns for rolling vol estimation
        self._port_returns: deque[float] = deque(maxlen=vol_lookback)

        self._max_drawdown = max_drawdown
        self._cooldown_bars = cooldown_bars
        self._peak_equity = 1.0
        self._halted = False
        self._bars_since_halt = 0

    def on_signal(self, event: SignalEvent) -> OrderEvent:
        target = event.target_weights.copy()

        if self._max_drawdown is not None:
            self._peak_equity = max(self._peak_equity, self.equity)
            drawdown = self.equity / self._peak_equity - 1.0

            if not self._halted and drawdown < -self._max_drawdown:
                self._halted = True
                self._bars_since_halt = 0
                _logger.warning(
                    "Circuit breaker fired: drawdown=%.2f%%  threshold=%.0f%%.",
                    drawdown * 100,
                    self._max_drawdown * 100,
                )
            elif self._halted:
                self._bars_since_halt += 1
                if self._bars_since_halt >= self._cooldown_bars:
                    self._halted = False
                    _logger.info("Circuit breaker reset after %d bars.", self._cooldown_bars)

            if self._halted:
                target = np.zeros_like(target)

        if not self._halted and self._vol_target is not None:
            if len(self._port_returns) == self._vol_lookback:
                realized_vol = (
                    statistics.stdev(self._port_returns)
                    * math.sqrt(self._periods_per_year)
                )
                if realized_vol > 0.0:
                    target = target * min(self._vol_target / realized_vol, 1.0)

        return OrderEvent(timestamp=event.timestamp, deltas=target - self.positions)

    def on_fill(self, event: FillEvent, asset_returns: np.ndarray) -> dict:
        """
        Apply the fill, earn the bar's return, compound equity.

        Position order: update first (to the new target), then earn the return.
        This mirrors event_driven._Portfolio.on_fill() and the vectorized engine:
        the weight decided at T-1 (= lagged[T]) is held during bar T.
        """
        self.positions = self.positions + event.deltas
        gross_return = float(self.positions @ asset_returns)
        net_return = gross_return - event.fee - event.slippage
        self.equity *= 1.0 + net_return
        self._port_returns.append(gross_return)
        return {
            "gross_return": gross_return,
            "fee": event.fee,
            "slippage": event.slippage,
            "net_return": net_return,
            "equity": self.equity,
            "turnover": float(np.abs(event.deltas).sum()),
        }


class _Broker:
    """
    Simulated execution: fills immediately at close, charges fees and slippage
    proportional to turnover (Σ|Δwᵢ|) rather than |Δposition|.
    """

    def __init__(self, fee_bps: float, slippage_bps: float = 0.0) -> None:
        self._fee_rate = fee_bps / 10_000.0
        self._slippage_rate = slippage_bps / 10_000.0

    def on_order(self, event: OrderEvent) -> FillEvent:
        turnover = float(np.abs(event.deltas).sum())
        return FillEvent(
            timestamp=event.timestamp,
            deltas=event.deltas,
            fee=turnover * self._fee_rate,
            slippage=turnover * self._slippage_rate,
        )


# ── Public run() ───────────────────────────────────────────────────────────────

def run(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    periods_per_year: float = 252.0,
    max_drawdown: float | None = None,
    cooldown_bars: int = 20,
) -> pd.DataFrame:
    """
    Run a multi-asset event-driven backtest and return a result DataFrame.

    Parameters
    ----------
    panel   : Close prices, shape (dates × N). Columns = asset names.
    weights : Target weights, shape (dates × N). Must share panel's index.
              Rows where any weight is NaN are treated as warmup and skipped.
    fee_bps : Exchange commission in basis points applied to Σ|Δwᵢ| per bar.

    1-BAR SIGNAL LAG
    ────────────────
    weights[T] determines the position held during bar T+1, identical to the
    legacy engine's shift(1) convention.

    Returns
    -------
    pd.DataFrame with columns: gross_return, fee, slippage, net_return,
        equity, turnover.
    """
    lagged = weights.shift(1)
    asset_returns = panel.pct_change()

    # Keep only rows where both lagged weights and asset returns are fully defined.
    mask = lagged.notna().all(axis=1) & asset_returns.notna().all(axis=1)
    lagged_clean = lagged[mask]
    returns_clean = asset_returns[mask]

    n_assets = panel.shape[1]
    portfolio = _Portfolio(
        n_assets=n_assets,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        periods_per_year=periods_per_year,
        max_drawdown=max_drawdown,
        cooldown_bars=cooldown_bars,
    )
    broker = _Broker(fee_bps, slippage_bps)

    rows: list[dict] = []
    index: list[pd.Timestamp] = []
    first = True

    for ts in lagged_clean.index:
        w_target = lagged_clean.loc[ts].values.astype(float)
        r_bar = returns_clean.loc[ts].values.astype(float)

        if first:
            # Mirror legacy: set initial position without charging a fee.
            portfolio.positions = w_target.copy()
            first = False
            continue

        market_evt = MarketEvent(timestamp=ts, asset_returns=r_bar)
        signal_evt = SignalEvent(timestamp=ts, target_weights=w_target)
        order_evt = portfolio.on_signal(signal_evt)
        fill_evt = broker.on_order(order_evt)
        row = portfolio.on_fill(fill_evt, market_evt.asset_returns)
        rows.append(row)
        index.append(ts)

    return pd.DataFrame(rows, index=index)
