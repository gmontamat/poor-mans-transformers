import csv
import numpy as np
import os
import random
import sys

# Import module directly from root folder
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from poormanstransformers.utils import to_one_hot, split_in_batches
from poormanstransformers.layers import Dense, ReLU, Dropout, Softmax, LogSoftmax
from poormanstransformers.losses import CategoricalCrossEntropy, Accuracy
from poormanstransformers.optimizers import Adam
from poormanstransformers.train import Trainer, DataGeneratorWrapper


def load_mnist(file_name):
    y = []
    X = []
    with open(file_name) as handler:
        csv_reader = csv.reader(handler)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            y.append(int(row[0]))
            assert int(row[0]) in range(10)
            features = [int(pixel) for pixel in row[1:]]
            assert len(features) == 28 * 28
            X.append(features)
    return np.array(X), np.array(y)


def split_mnist(file_name, split=.85, shuffle=True):
    X, y = load_mnist(file_name)
    train_size = round(X.shape[0] * split)
    indexes = list(range(X.shape[0]))
    if shuffle:
        random.shuffle(indexes)
    X_train, y_train = X[indexes[:train_size], :], y[indexes[:train_size]]
    X_eval, y_eval = X[indexes[train_size:], :], y[indexes[train_size:]]
    return X_train, y_train, X_eval, y_eval


if __name__ == '__main__':
    try:
        X_train, y_train, X_eval, y_eval = split_mnist('train.csv')
    except FileNotFoundError:
        print("Download MNIST dataset from: https://www.kaggle.com/c/digit-recognizer/data?select=train.csv")
        sys.exit(0)
    y_train = to_one_hot(y_train)
    y_eval = to_one_hot(y_eval)
    mlp = [
        Dense(128, input_shape=(None, 28 * 28)),
        ReLU(),
        Dropout(0.25),
        Dense(64),
        ReLU(),
        Dropout(0.25),
        Dense(10),
        LogSoftmax()
    ]
    batch_size = 32
    train_data = DataGeneratorWrapper(
        split_in_batches, batch_size=batch_size, features=X_train / 255, targets=y_train,
        total_batches=len(X_train) // batch_size
    )
    eval_data = DataGeneratorWrapper(split_in_batches, features=X_eval / 255, targets=y_eval)
    trainer = Trainer(
        model=mlp,
        optimizer=Adam(),
        loss=CategoricalCrossEntropy(from_logits=True),
        metrics=[Accuracy(from_logits=True)]
    )
    trainer.fit(train_data, epochs=5, eval_data=eval_data)
