from typing import List
from .layers import Layer


class Model:

    def __init__(self):
        pass


class Sequential(Model):

    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(self.layers):
            if i and getattr(layer, 'input_shape') is None:
                setattr(layer, 'input_shape', getattr(self.layers[i-1], 'output_shape'))
            layer.initialize_parameters()
