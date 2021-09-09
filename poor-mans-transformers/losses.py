import numpy as np

from typing import Tuple


class Loss:
    """
    Loss base class. When called, it returns the sum of the
    cost function divided by the batch size and the gradients
    w.r.t. the predictions (y_hat).
    The shape of y and y_hat is (batch_size, output_dim),
    no reduction is implemented here, just sum divided by
    batch_size. y is expected to be provided in a one-hot
    representation.
    """

    def __init__(self):
        pass

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return loss value and gradients w.r.t. y_hat"""
        raise NotImplementedError()


class CategoricalCrossEntropy(Loss):

    def __init__(self, from_logits: bool = False):
        """Need to specify whether y_hat are logits or probabilities.
        """
        super().__init__()
        self.from_logits = from_logits

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.from_logits:
            # We can avoid having to use np.log by sending logits
            # LogSoftmax is also more numerically stable
            loss = np.mean(-y_hat[y == 1.])
            grad = None  # TODO
        else:
            y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)  # Avoid division by zero
            loss = np.mean(np.sum(-y * np.log(y_hat), axis=-1))
            grad = (y_hat - y) / y.shape[0]
        return float(loss), grad


class Metric:

    def __init__(self):
        pass

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Return metric value."""
        raise NotImplementedError
