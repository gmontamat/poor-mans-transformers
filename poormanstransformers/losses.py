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

    def __init__(self, from_logits: bool = False):
        """Need to specify whether y_hat are logits or probabilities."""
        self.from_logits = from_logits

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return loss value and gradients w.r.t. y_hat"""
        raise NotImplementedError()

    def __str__(self) -> str:
        """Print Loss metric name."""
        raise NotImplementedError()


class CategoricalCrossEntropy(Loss):

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.from_logits:
            # y_hat here are log probabilities (logits)
            # LogSoftmax is more numerically stable
            loss = -np.mean(np.sum(y * y_hat, axis=-1))
            grad = -y / y.shape[0]
        else:
            y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)  # Avoid overflow
            loss = -np.mean(np.sum(y * np.log(y_hat), axis=-1))
            grad = -y / y_hat / y.shape[0]
        return float(loss), grad

    def __str__(self) -> str:
        return "categorical-cross-entropy"


class Metric:

    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Return computed metric."""
        raise NotImplementedError()

    def __str__(self) -> str:
        """Print metric name."""
        raise NotImplementedError()


class Accuracy(Metric):

    def __call__(self, y: np.ndarray, y_hat: np.ndarray, threshold: float = 0.5) -> float:
        if self.from_logits:
            threshold = np.log(threshold)
        return float(np.mean(y == (y_hat > threshold).astype(y.dtype)))

    def __str__(self) -> str:
        return "accuracy"
