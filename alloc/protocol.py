from typing import Protocol

import numpy as np


class Allocator(Protocol):
    """Produces a target weight vector summing to 1."""
    def __call__(self, mu: np.ndarray | None, Sigma: np.ndarray, **params) -> np.ndarray: ...


class CovEstimator(Protocol):
    """Estimates Sigma (N, N) PSD from a returns array or DataFrame."""
    def __call__(self, returns) -> np.ndarray: ...
