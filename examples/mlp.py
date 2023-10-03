import numpy as np
import os
import sys

# Import module directly from root folder
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from poormanslayers.utils import to_one_hot, split_in_batches
from poormanslayers.layers import Dense, ReLU, Dropout, Softmax, LogSoftmax
from poormanslayers.losses import CategoricalCrossEntropy, Accuracy
from poormanslayers.optimizers import Adam
from poormanslayers.train import Trainer, DataGeneratorWrapper


def load_mnist(file_name: str):
    mnist = np.load(file_name)
    x_train = mnist["x_train"].reshape(mnist["x_train"].shape[:-2] + (-1,))
    y_train = mnist["y_train"]
    x_test = mnist["x_test"].reshape(mnist["x_test"].shape[:-2] + (-1,))
    y_test = mnist["y_test"]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    try:
        x_train, y_train, x_test, y_test = load_mnist('mnist.npz')
    except FileNotFoundError:
        print("Download MNIST dataset. Run ./download_mnist.sh")
        sys.exit(0)
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    mlp = [
        Dense(128, input_shape=(None, 28 * 28)),
        ReLU(),
        Dropout(0.25),
        Dense(64),
        ReLU(),
        Dropout(0.25),
        Dense(10),
        LogSoftmax()  # Softmax()
    ]
    batch_size = 32
    train_data = DataGeneratorWrapper(split_in_batches, batch_size=batch_size, features=x_train / 255, targets=y_train)
    test_data = DataGeneratorWrapper(split_in_batches, batch_size=batch_size, features=x_test / 255, targets=y_test)
    trainer = Trainer(
        model=mlp,
        optimizer=Adam(),
        loss=CategoricalCrossEntropy(from_logits=True),
        metrics=[Accuracy(from_logits=True)]
    )
    trainer.fit(train_data, epochs=5, eval_data=test_data, batches_per_epoch=len(x_train) // batch_size)
