"""
strategy/__init__.py — Strategy registry and loader.

To add a new strategy: create strategy/<name>.py with a generate_signals()
function, then add its name to _STRATEGIES below.
"""

import importlib
from functools import partial

_STRATEGIES = {
    "ma_crossover",
    "ema_crossover",
    "rsi",
    "bollinger_bands",
    "momentum",
    "macd",
    "mean_reversion",
}


def load_strategy(cfg: dict):
    """
    Return a generate_signals(bars) callable configured from config.yaml.

    Reads cfg["strategy"] for the strategy name and cfg["strategies"][name]
    for its parameters. Falls back to "ma_crossover" if strategy is unset.
    """
    name = cfg.get("strategy", "ma_crossover")
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {sorted(_STRATEGIES)}"
        )
    module = importlib.import_module(f"strategy.{name}")
    params = cfg.get("strategies", {}).get(name, {})
    return partial(module.generate_signals, **params)
