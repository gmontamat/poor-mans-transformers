import numpy as np

from typing import List, Optional, Tuple

from .optimizers import Optimizer


class Parameter:
    """
    Parameter base class. Only layers can instantiate these objects
    and they need to initialize the weights and set the optimizer
    before doing forward/backward propagation.
    """

    def __init__(self, weights: Optional[np.ndarray] = None, optimizer: Optional[Optimizer] = None):
        self.weights = weights
        self.optimizer = optimizer

    def __call__(self) -> np.ndarray:
        """Access the weights easily with
        p = Parameter(); p()
        """
        assert self.weights is not None
        return self.weights

    def update(self, grad: np.ndarray):
        """Update weights given d_loss / d_weights (grad)."""
        assert self.weights is not None and self.optimizer is not None
        self.weights = self.optimizer(self.weights, grad)


class Layer:
    """
    Layer building block.
    Each layer is capable of performing two things:
    - Process input to get output:           output = layer.forward(x)
    - Propagate gradients through itself:    grad_input = layer.backward(x, grad)
    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self,
                 input_shape: Optional[Tuple[Optional[int], ...]] = None,
                 output_shape: Optional[Tuple[Optional[int], ...]] = None):
        """Define layer's input and output shape, and parameters (if any)."""
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __str__(self):
        return f"{type(self).__name__.ljust(15)}: {self.output_shape}"

    def initialize(self):
        """Initialize parameters weights and optimizers."""
        pass

    def get_parameters(self) -> List[Parameter]:
        """Return all parameters used by the layer."""
        return [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), Parameter)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Take input data of shape `input_shape`, perform forward pass.
        """
        # Dummy layer just returns whatever it gets as input.
        return x

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform backpropagation step through the layer with respect to a given input (x).
        To compute loss gradients w.r.t x, you need to apply chain rule:
        d_loss / d_x  = (d_loss / d_layer) * (d_layer / d_x)
        Luckily, you already receive d_loss / d_layer as `grad`, so you only need to multiply it by d_layer / d_x.
        If your layer has parameters (e.g. dense layer), you also need to update them here using d_loss / d_layer
        """
        # The gradient of a dummy layer is precisely grad, but we'll write it more explicitly
        num_units = x.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad, d_layer_d_input)


class Activation(Layer):
    """
    Base layer whose input and output shape are the same.
    """

    def __init__(self, input_shape: Optional[Tuple[Optional[int], ...]] = None):
        super(Activation, self).__init__(input_shape=input_shape, output_shape=input_shape)

    def initialize(self):
        self.output_shape = self.input_shape


class Dense(Layer):
    """
    A fully-connected layer which performs a learned affine transformation:
    f(x) = <W*x> + b
    """

    def __init__(self, n_units: int, input_shape: Optional[Tuple[Optional[int], int]] = None):
        super(Dense, self).__init__(input_shape, (input_shape[0] if input_shape is not None else None, n_units))
        self.n_units = n_units
        self.W = Parameter()
        self.b = Parameter()

    def initialize(self):
        """Initialize parameters using Xavier initialization (aka. glorot_uniform).
        """
        assert self.input_shape is not None, "`input_shape` must be specified if it is the first layer in the network."
        input_units = self.input_shape[1]
        # Define parameters and perform Xavier initialization (aka. glorot_uniform)
        setattr(self.W, 'weights', np.random.normal(0., np.sqrt(1. / input_units), size=(input_units, self.n_units)))
        setattr(self.b, 'weights', np.zeros(self.n_units))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform an affine transformation:
        f(x) = <W*x> + b
        input shape: [batch, input_units]
        output shape: [batch, n_units]
        """
        W, b = self.W(), self.b()
        return np.dot(x, W) + b

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Compute d_loss / d_W and d_loss / d_b to update parameters
        and return d_loss / d_x = d_loss / d_dense * d_dense / d_x"""
        W, b = self.W(), self.b()
        grad_x = np.dot(grad, W.T)
        self.W.update(np.dot(x.T, grad))
        self.b.update(np.dot(np.ones((x.shape[0],)), grad))
        return grad_x


class ReLU(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, np.zeros_like(x))

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return (x > 0.).astype(grad.dtype) * grad


class Softmax(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Shift to negatives to avoid overflow
        # aka. stable softmax
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        p = self.forward(x)
        return p * (1. - p) * grad


class LogSoftmax(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x - np.sum(x, axis=-1, keepdims=True)

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # TODO: check your lazy math!
        softmax = np.exp(self.forward(x))
        return (1. - softmax(x)) * grad


class Dropout(Activation):

    def __init__(self, rate: float,
                 input_shape: Optional[Tuple[Optional[int], int]] = None,
                 mode: str = 'train'):
        super(Dropout, self).__init__(input_shape)
        self.rate = rate
        self.factor = 1. / (1. - rate)
        self.mode = mode
        self.last_forwards = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.mode == 'train':
            self.last_forwards = (np.random.random(x.shape) > self.rate).astype(np.float32)
            return np.multiply(self.last_forwards, x) * self.factor
        return x

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        assert self.last_forwards is not None, "Do forward pass before backward"
        return np.multiply(self.last_forwards, grad) * self.factor
