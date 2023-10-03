import numpy as np

from typing import List, Optional, Tuple, Union

from .optimizers import Optimizer


class Parameter:
    """
    Parameter base class. Only layers can instantiate these objects,
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

    def forward(self, *args: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Take input data of shape `input_shape`, perform forward pass.
        """
        # Dummy layer just returns whatever it gets as input.
        if len(args) == 1:
            return args[0]
        return args

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform backpropagation step through the layer with respect to a given input (x).
        To compute loss gradients w.r.t x, you need to apply chain rule:
        d_loss / d_x  = (d_loss / d_layer) · (d_layer / d_x)
        Luckily, you already receive d_loss / d_layer as `grad`, so you only need to multiply it by d_layer / d_x.
        If your layer has parameters (e.g. dense layer), you also need to update them here using d_loss / d_layer
        """
        # The gradient of a dummy layer is precisely grad, but we'll write it more explicitly
        # In general, we need to compute the Jacobian of the forward function and return its dot
        # product with `grad`
        num_units = x.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad, d_layer_d_input)


class Activation(Layer):
    """
    Base layer whose input and output shape are the same.
    """

    def __init__(self, input_shape: Optional[Tuple[Optional[int], ...]] = None):
        super(Activation, self).__init__(input_shape=input_shape, output_shape=input_shape)
        self.last_output = None  # Activations need the last forwards to do backpropagation

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
        and return d_loss / d_x = d_loss / d_dense · d_dense / d_x"""
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


class Sigmoid(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = 1. / (1. + np.exp(-x))
        return self.last_output

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad * self.last_output * (1. - self.last_output)


class Softmax(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Shift x to [-inf., 0] to avoid overflow
        # aka. stable softmax
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.last_output = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.last_output

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # Compute Jacobian for each batch since full matrix is sparse
        grad_x = np.zeros_like(grad)
        for batch in range(self.last_output.shape[0]):
            # Jacobian for a single sample in batch is:
            # Si - SiSj if i==j or else -SiSj
            s = self.last_output[batch]
            jac = -np.outer(s, s) + np.diag(s)
            grad_x[batch] = np.dot(grad[batch], jac)
        return grad_x


class LogSoftmax(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Shift x to [-inf., 0] to avoid overflow
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        self.last_output = x_shifted - np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
        return self.last_output

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        softmax = np.exp(self.last_output)
        s_dim = softmax.shape[1]
        grad_x = np.zeros_like(grad)
        for batch in range(self.last_output.shape[0]):
            # Jacobian for a single sample in batch is:
            # 1 - Si if i==j or else -Si
            jac = np.eye(s_dim) - softmax[batch, np.newaxis]
            grad_x[batch] = np.dot(grad[batch], jac)
        return grad_x


class Dropout(Activation):

    def __init__(self, rate: float,
                 input_shape: Optional[Tuple[Optional[int], int]] = None,
                 mode: str = 'train'):
        super(Dropout, self).__init__(input_shape)
        self.rate = rate
        self.factor = 1. / (1. - rate)
        self.mode = mode

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.mode == 'train':
            self.last_output = (np.random.random(x.shape) > self.rate).astype(np.float32)
            return self.last_output * x * self.factor
        return x

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        assert self.last_output is not None, "Do forward pass before backward"
        return self.last_output * grad * self.factor


class Embedding(Layer):
    """
    Turn positive integers (indexes) into dense vectors of fixed size.
    e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    This layer can only be used as the first layer in a model.
    """

    def __init__(self, vocab_size: int, d_feature: int, input_length: Optional[int] = None):
        super(Embedding, self).__init__((None, input_length), (None, input_length, d_feature))
        self.weights_shape = (vocab_size, d_feature)
        self.W = Parameter()

    def initialize(self):
        setattr(self.W, 'weights', np.random.uniform(-0.05, 0.05, size=self.weights_shape))

    def save(self, file_name: str):
        """Save embeddings for future usage."""
        if file_name[-4:] != '.npy':
            file_name += '.npy'
        np.save(file_name, self.W())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Use inputs as indexes to avoid matrix-matrix multiplication."""
        return self.W()[x]

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        dW = np.zeros_like(self.W())
        np.add.at(dW, x, grad)  # Below we give the loop version
        # reshaped_grad = np.reshape(grad, (x.shape[0] * x.shape[1], -1))
        # for index, g in zip(x.flatten(), reshaped_grad):
        #     dW[index] += g
        self.W.update(dW)
        return np.empty_like(x)  # Should not be propagated


class AxisMean(Layer):
    """
    This layer is usually implemented as a Lambda layer in popular frameworks:
    e.g. model.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
    Here, we don't have a method to compute gradients, so we specify the forward
    and backwards step manually.
    """

    def __init__(self, axis: int, input_shape: Optional[Tuple[Optional[int], ...]] = None):
        super(AxisMean, self).__init__(input_shape, None)
        self.axis = axis

    def initialize(self):
        assert self.input_shape is not None, "Cannot initialize DimensionMean without `input_shape`."
        self.output_shape = self.input_shape[:self.axis] + self.input_shape[self.axis + 1:]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=self.axis)

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return np.repeat(np.expand_dims(grad / x.shape[self.axis], axis=self.axis), x.shape[self.axis], axis=self.axis)


class AxisDot(Layer):
    """
    Compute dot product of vectors along a certain axis of size 2
    """

    def __init__(self, axis: int, input_shape: Optional[Tuple[Optional[int], ...]] = None):
        super(AxisDot, self).__init__(input_shape, None)
        self.axis = axis
        self.last_input = None

    def initialize(self):
        assert self.input_shape is not None, "Cannot initialize DimensionMean without `input_shape`."
        assert self.input_shape[self.axis] == 2, "Selected axis must have size of 2."
        assert len(self.input_shape) == 3, "Only 3 dimensions allowed: batch, vector, and number of vectors"
        self.output_shape = (self.input_shape[0], 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        return np.diag(np.dot(x[:, :, 0], x[:, :, 1].T))[:, np.newaxis]

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.axis == 2 or self.axis == -1:
            return self.last_input[:, :, ::-1] * np.repeat(np.expand_dims(grad, axis=-1), 2, axis=-1)
        return self.last_input[:, ::-1, :] * np.expand_dims(np.repeat(grad, 2, axis=-1), axis=-1)
