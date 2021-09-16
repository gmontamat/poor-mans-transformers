import numpy as np

from copy import copy
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

from .layers import Layer, Parameter
from .losses import Loss, Metric
from .optimizers import Optimizer

# A model is simply a list of layers
Model = List[Layer]
# ModelWeights = List[Dict[str, Parameter]]


class DataGeneratorWrapper:
    """
    Wrap a data generator with all of its parameters
    so that it can be "rewound" each epoch. Use the
    generator `.utils.split_in_batches` if the entire
    dataset is loaded in memory.
    """

    def __init__(self,
                 generator: Callable[..., Generator[Tuple[np.ndarray, np.ndarray], None, None]],
                 **kwargs):
        self.generator = generator
        self.kwargs = kwargs

    def __call__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        return self.generator(**self.kwargs)


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
                 metrics: Optional[List[Union[Loss, Metric]]] = None):
        self.layers = model
        self.optimizer = optimizer
        self.loss = loss
        if metrics is None:
            self.metrics = []
        else:
            self.metrics = metrics

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
        for parameter in layer.get_parameters():
            setattr(parameter, 'optimizer', copy(self.optimizer))
        layer.initialize()

    def set_mode(self, mode: str):
        for layer in self.layers:
            if hasattr(layer, 'mode'):
                setattr(layer, 'mode', mode)

    def compile(self):
        """Ensure input_shape and output_shape are set
        and valid then initialize the layer.
        """
        if not hasattr(self.layers[0], 'input_shape'):
            raise AttributeError("`input_shape` not defined for the first layer.")
        self.initialize_layer(self.layers[0])
        print(self.layers[0])
        for i, layer in enumerate(self.layers[1:]):
            if getattr(layer, 'input_shape'):
                if not self.compare_shapes(layer.input_shape, self.layers[i].output_shape):
                    raise AttributeError("`input_shape` mismatch.")
            else:
                setattr(layer, 'input_shape', self.layers[i].output_shape)
            self.initialize_layer(layer)
            print(layer)

    def validate_data(self, data_generator: DataGeneratorWrapper):
        """Ensure that input features and output targets
        match the network's input and output shapes
        respectively."""
        data = data_generator()
        features, targets = next(data)
        assert self.compare_shapes(features.shape, self.layers[0].input_shape), "Input shape is incompatible"
        assert self.compare_shapes(targets.shape, self.layers[-1].output_shape), "Output shape is incompatible"

    def fit(self,
            train_data: DataGeneratorWrapper,
            epochs: int,
            eval_data: Optional[DataGeneratorWrapper] = None):
        """Train model with data passed."""
        self.compile()
        self.validate_data(train_data)
        if eval_data:
            self.validate_data(eval_data)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}...")
            self.set_mode('train')
            train_loss = []
            train_metrics = {str(metric): [] for metric in self.metrics}
            for features, targets in train_data():
                outputs = [features]
                # Forward propagation
                for layer in self.layers:
                    outputs.append(layer.forward(outputs[-1]))
                # Compute loss and grad w.r.t. final output
                loss, grad = self.loss(targets, outputs[-1])
                train_loss.append(loss)
                # Compute metrics
                for metric in self.metrics:
                    if isinstance(metric, Loss):
                        value, _ = metric(targets, outputs[-1])
                    else:
                        value = metric(targets, outputs[-1])
                    train_metrics[str(metric)].append(value)
                # Backward propagation
                for layer, layer_input in zip(reversed(self.layers), reversed(outputs[:-1])):
                    grad = layer.backward(layer_input, grad)
            print(f"train-{self.loss}: {np.mean(train_loss):.5f}")
            for metric in self.metrics:
                print(f"train-{metric}: {np.mean(train_metrics[str(metric)]):.5f}")
            # Compute eval metrics
            if eval_data is None:
                continue
            self.set_mode('eval')
            eval_loss = []
            eval_metrics = {str(metric): [] for metric in self.metrics}
            for features, targets in eval_data():
                outputs = [features]
                # Forward propagation
                for layer in self.layers:
                    outputs.append(layer.forward(outputs[-1]))
                loss, _ = self.loss(targets, outputs[-1])
                eval_loss.append(loss)
                # Compute metrics
                for metric in self.metrics:
                    if isinstance(metric, Loss):
                        value, _ = metric(targets, outputs[-1])
                    else:
                        value = metric(targets, outputs[-1])
                    eval_metrics[str(metric)].append(value)
            print(f"eval-{self.loss}: {np.mean(eval_loss):.5f}")
            for metric in self.metrics:
                print(f"eval-{metric}: {np.mean(eval_metrics[str(metric)]):.5f}")
