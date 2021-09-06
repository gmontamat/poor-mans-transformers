import numpy as np

from types import Tuple, Optional

from .optimizers import Optimizer, Adam


class Layer:
    """Layer building block.
    Each layer is capable of performing two things:
    - Process input to get output:           output = layer.forward(x)
    - Propagate gradients through itself:    grad_input = layer.backward(x, grad)
    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """Define layer's input and output shape and parameters (if any)."""
        self.input_shape = None
        self.output_shape = None

    def initialize_parameters(self):
        raise NotImplementedError()

    def update_parameters(self, **kwargs):
        raise NotImplementedError()

    def forward(self, x):
        """Take input data of shape `input_shape`, perform forward pass.
        """
        # Dummy layer just returns whatever it gets as input.
        return x

    def backward(self, x, grad):
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


class Dense(Layer):
    """
    A fully-connected layer which performs a learned affine transformation:
    f(x) = <W*x> + b
    """

    def __init__(self, n_units: int, input_shape: Optional[Tuple[Optional[int], int]] = None):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] if input_shape else None, n_units)
        self.n_units = n_units
        self.W = None
        self.b = None

    def initialize_parameters(self):
        """Initialize parameters using Xavier initialization (aka. glorot_uniform).
        """
        assert self.input_shape is not None, "`input_shape` must be specified if it is the first layer in the network."
        input_units = self.input_shape[1]
        # Define parameters and perform Xavier initialization (aka. glorot_uniform)
        self.W = np.random.normal(0., np.sqrt(1. / input_units), size=(input_units, self.n_units))
        self.b = np.zeros(self.n_units)

    def update_parameters(self, grad_W: np.ndarray, grad_b: np.ndarray):
        """Update layer's parameters using Adam"""
        assert grad_W.shape == self.W.shape and grad_b.shape == self.b.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform an affine transformation:
        f(x) = <W*x> + b
        input shape: [batch, input_units]
        output shape: [batch, n_units]
        """
        return np.dot(x, self.W) + self.b

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # compute d_loss / d_x = d_loss / d_dense * d_dense / d_x
        # where d_dense / d_x are the transposed weights
        grad_x = np.dot(grad, self.W.T)
        # compute gradient w.r.t. weights and biases (d_loss / d_weights)
        grad_W = np.dot(x.T, grad)
        grad_b = np.dot(np.ones((x.shape[0],)), grad)
        # Update parameters
        self.update_parameters(grad_W, grad_b)
        return grad_x
