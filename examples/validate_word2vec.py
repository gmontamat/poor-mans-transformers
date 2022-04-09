import numpy as np
import os
import sys

from matplotlib import pyplot as plt

# Import module directly from root folder
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from poormanstransformers.layers import Embedding, Sigmoid, Dense, AxisDot
from poormanstransformers.losses import BinaryCrossEntropy
from poormanstransformers.optimizers import RMSProp
from poormanstransformers.train import Trainer, DataGeneratorWrapper


VOCABULARY = [
    ('paris', 0),
    ('france', 1),
    ('berlin', 2),
    ('germany', 3)
]  # This is just a reference


def generate_word2vec_samples(batch_size: int, max_batches: int):
    all_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    all_labels = [[1], [1], [0], [0], [1], [1]]
    ctr, total_batches, pairs, labels = 0, 0, [], []
    idx = 0
    while True:
        pairs.append(all_pairs[idx])
        labels.append(all_labels[idx])
        idx += 1
        idx %= len(all_labels)
        ctr += 1
        if ctr == batch_size:
            yield np.array(pairs), np.array(labels)
            ctr, pairs, labels = 0, [], []
            total_batches += 1
        if total_batches == max_batches:
            return


if __name__ == '__main__':
    # A simple vocabulary and data generator to validate
    # that word2vec embeddings are being trained properly
    vocab_size = 4
    embedding_size = 2
    batch_size = 32
    max_batches = 1000
    train_data = DataGeneratorWrapper(generate_word2vec_samples, batch_size=batch_size, max_batches=max_batches)
    skipgram = [
        Embedding(vocab_size=vocab_size, d_feature=embedding_size, input_length=2),
        AxisDot(1),
        Dense(1),
        Sigmoid()
    ]
    trainer = Trainer(
        model=skipgram,
        optimizer=RMSProp(),
        loss=BinaryCrossEntropy()
    )
    trainer.fit(train_data, epochs=10, batches_per_epoch=max_batches)
    # Print and plot word embeddings
    x, y, names = [], [], []
    for (word, id_), embedding in zip(VOCABULARY, skipgram[0].W().tolist()):
        print(id_, word, embedding)
        x.append(embedding[0])
        y.append(embedding[1])
        names.append(word)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(names):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
