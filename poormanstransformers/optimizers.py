import numpy as np


class Optimizer:

    def __init__(self):
        pass

    def __call__(self, weights: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Compute updated weights and return them."""
        raise NotImplementedError()


class Adam(Optimizer):
    """
    Reference:
        - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    """

    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        super(Adam, self).__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        # Define momentum and velocity
        self.m = None
        self.v = None
        self.t = 0

    def __call__(self, weights: np.ndarray, grad: np.ndarray) -> np.ndarray:
        assert weights.shape == grad.shape
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        self.m = self.beta1 * self.m + (1. - self.beta1) * grad
        self.v = self.beta2 * self.v + (1. - self.beta2) * np.power(grad, 2)
        m_hat = self.m / (1. - self.beta1 ** self.t)
        v_hat = self.v / (1. - self.beta2 ** self.t)
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    """
    Reference:
        - https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        - https://paperswithcode.com/method/rmsprop
    """

    def __init__(self, learning_rate: float = 0.001,
                 gamma: float = 0.9, epsilon: float = 1e-8):
        super(RMSProp, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.mean_square = None
        self.eps = epsilon

    def __call__(self, weights: np.ndarray, grad: np.ndarray):
        if self.mean_square is None:
            self.mean_square = np.zeros_like(grad)
        self.mean_square = self.gamma * self.mean_square + (1 - self.gamma) * np.power(grad, 2)
        return weights - self.learning_rate * grad / np.sqrt(self.mean_square + self.eps)
