import inspect
import networkx as nx

from .layers import Layer, Dense, ReLU, Dropout, LogSoftmax, Embedding, Sigmoid


class NeuralNetwork:

    def __init__(self):
        self.callgraph = nx.MultiDiGraph()

    def compile(self):
        """Pass callgraph to all the Layers defined in __init__
        so they can add edges when called.
        """
        for element in inspect.getmembers(self):
            if isinstance(element, Layer):
                element.__setattr__('network', self.callgraph)
        # Build directed acyclic graph
        inputs = [x for x in inspect.getfullargspec(self.forward)[0]]
        # Run forward pass with pointers to build DAG
        self.forward(*inputs)
        # Validate that the callgraph is a DAG (can't backpropagate with cycles)
        assert nx.is_directed_acyclic_graph(self.callgraph), \
            "Check forward method, NeuralNetwork contains cycles."

    def forward(self, *args):
        """Define forward pass of Neural Network."""
        raise NotImplementedError


class MLP(NeuralNetwork):

    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = Dense(128)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(0.25)
        self.dense2 = Dense(64)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(0.25)
        self.dense3 = Dense(10)
        self.logsoftmax = LogSoftmax()

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return self.logsoftmax(x)


class SkipGram(NeuralNetwork):

    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()
        self.embedding = Embedding(vocab_size=vocab_size, d_feature=embedding_size)
        self.dot = Dot()  # TODO
        self.dense = Dense(1)
        self.sigmoid = Sigmoid()

    def forward(self, x1, x2):
        embedding1 = self.embedding(x1)
        embedding2 = self.embedding(x2)
        dot = self.dot(embedding1, embedding2)
        y = self.dense(dot)
        return self.sigmoid(y)
