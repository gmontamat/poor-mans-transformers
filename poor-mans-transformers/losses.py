import numpy as np

from typing import Tuple


class Loss:

    def __init__(self):
        pass

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return loss value and gradients w.r.t. y"""
        raise NotImplementedError()

# TODO: categorical cross-entropy & cross-entropy


class Metric:

    def __init__(self):
        pass

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Return metric value."""
        raise NotImplementedError
