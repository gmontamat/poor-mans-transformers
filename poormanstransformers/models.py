from copy import copy
from typing import List, Tuple, Optional, Dict, Union, Generator

import numpy as np

from .layers import Layer, Parameter
from .losses import Loss, Metric
from .optimizers import Optimizer

# A model is simply a list of layers
Model = List[Layer]
# ModelWeights = List[Dict[str, Parameter]]


class Trainer:
    """
    Defines a supervised training task with a model,
    optimizer, loss, training and evaluation data,
    learning rate schedule, early stopping, etc.
    """

    def __init__(self,
                 model: Model,
                 optimizer: Optimizer,
                 loss: Loss,
                 metrics: List[Union[Loss, Metric]],
                 mode: str = 'train'):
        self.layers = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.mode = mode

    @staticmethod
    def compare_shapes(shape1: Tuple[Optional[int], ...],
                       shape2: Tuple[Optional[int], ...]) -> bool:
        if len(shape1) != len(shape2):
            return False
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 is not None and dim2 is not None and dim1 != dim2:
                return False
        return True

    def initialize_layer(self, layer: Layer):
        if hasattr(layer, 'mode'):
            setattr(layer, 'mode', self.mode)
        for parameter in layer.get_parameters():
            setattr(parameter, 'optimizer', copy(self.optimizer))
        layer.initialize()

    def prepare_model(self):
        """Ensure input_shape and output_shape are set
        and valid then initialize the layer.
        """
        if not hasattr(self.layers[0], 'input_shape'):
            raise AttributeError("`input_shape` not defined for the first layer.")
        self.initialize_layer(self.layers[0])
        for i, layer in enumerate(self.layers[1:]):
            if hasattr(layer, 'input_shape'):
                if not self.compare_shapes(layer.input_shape, self.layers[i].input_shape):
                    raise AttributeError("`input_shape` mismatch.")
            else:
                setattr(layer, 'input_shape', self.layers[i].output_shape)
            self.initialize_layer(layer)

    def fit(self,
            train_generator: Generator[Tuple[np.ndarray, np.ndarray]],
            epochs: int,
            eval_generator: Optional[Generator[Tuple[np.ndarray, np.ndarray]]] = None):
        """Train model with data passed."""
        self.prepare_model()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}...")
            train_loss = []
            train_metrics = {str(metric): [] for metric in self.metrics}
            for x_train, y_train in train_generator:
                outputs = [x_train]
                # Forward propagation
                for layer in self.layers:
                    outputs.append(layer.forward(outputs[-1]))
                # Compute loss and grad w.r.t. final output
                loss, grad = self.loss(y_train, outputs[-1])
                train_loss.append(loss)
                # Compute metrics
                for metric in self.metrics:
                    if isinstance(metric, Loss):
                        value, _ = metric(y_train, outputs[-1])
                    else:
                        value = metric(y_train, outputs[-1])
                    train_metrics[str(metric)] = value
                # Backward propagation
                for layer, layer_input in zip(reversed(self.layers), reversed(outputs[:-1])):
                    grad = layer.backward(layer_input, grad)
            print(np.mean(train_loss))
            for metric in self.metrics:
                print(np.mean(train_metrics[str(metric)]))
