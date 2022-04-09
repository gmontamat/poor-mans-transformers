import numpy as np
import os
import pickle
import random
import sys

from collections import Counter
from typing import List, Dict, Tuple, Optional

# Import module directly from root folder
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from poormanstransformers.layers import Embedding, Sigmoid, Dense, AxisDot
from poormanstransformers.losses import BinaryCrossEntropy
from poormanstransformers.optimizers import RMSProp
from poormanstransformers.train import Trainer, DataGeneratorWrapper


def create_vocabulary(text_file: str, size: int = 10000, oov_token: str = '[OOV]',
                      vocabulary_file: str = 'vocabulary.pkl') -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create a dictionary that maps a word to a unique integer."""
    if os.path.isfile(vocabulary_file):
        with open(vocabulary_file, 'rb') as handler:
            return pickle.load(handler)
    print("Generating vocabulary dictionary...")
    with open(text_file) as handler:
        text = handler.read()
    word_counts = Counter(text.lower().split())
    oov_counter = sum(word_counts.values())
    vocabulary = {oov_token: 0}
    word_frequency = {oov_token: 0}
    for i, word_count in enumerate(word_counts.most_common(size - 1)):
        word, count = word_count
        vocabulary[word] = i + 1
        word_frequency[word] = count
        oov_counter -= count
    word_frequency[oov_token] = oov_counter
    with open(vocabulary_file, 'wb') as handler:
        pickle.dump((vocabulary, word_frequency), handler)
    print("Vocabulary generated!")
    return vocabulary, word_frequency


def skipgrams(text_sequence: List[str], word_counts: Dict[str, int], window_size: int = 10,
              negative_samples: int = 20, oov_token: str = '[OOV]', batch_size: int = 32):
    """Generate target -> context pairs (positive and negative)
    using subsampling and negative sampling.
    """
    ctr, pairs, labels = 0, [], []
    # Variables to compute subsampling
    total_words = sum(word_counts.values())
    oov_frequency = word_counts[oov_token]
    # Variables to compute negative sampling
    p_negative = {word: count ** .75 for word, count in word_counts.items()}
    negative_sum = sum(p_negative.values())
    p_negative = {word: count / negative_sum for word, count in p_negative.items()}
    for i, target in enumerate(text_sequence):
        freq = word_counts.get(target, oov_frequency) / total_words
        p_keep = min((np.sqrt(freq / .001) + 1.) * .001 / freq, 1.)
        if p_keep < random.random():
            continue  # Do not sample this one
        # Positive pairs
        for j in range(max(0, i - window_size), min(len(text_sequence), i + window_size + 1)):
            if i == j:
                continue
            pairs.append((target, text_sequence[j]))
            labels.append([1])
            ctr += 1
            if ctr == batch_size:
                yield pairs, labels
                ctr, pairs, labels = 0, [], []
        # Negative pairs
        negatives = random.choices(list(p_negative.keys()), weights=list(p_negative.values()), k=negative_samples)
        for context in negatives:
            pairs.append((target, context))
            labels.append([0])
            ctr += 1
            if ctr == batch_size:
                yield pairs, labels
                ctr, pairs, labels = 0, [], []


def generate_skipgrams(text_file: str, vocabulary: Dict[str, int], word_counts: Dict[str, int],
                       window_size: int = 10, negative_samples: int = 20, batch_size: int = 32,
                       oov_token: str = '[OOV]', max_batches: Optional[int] = None):
    """Generate Skip-gram model batches."""
    with open(text_file) as handler:
        text_sequence = handler.read().lower().split()
    total_batches = 0
    oov_index = vocabulary[oov_token]
    for pairs, labels in skipgrams(text_sequence, word_counts, window_size, negative_samples, oov_token, batch_size):
        X = np.array([
            [vocabulary.get(target, oov_index), vocabulary.get(context, oov_index)]
            for target, context in pairs
        ])
        y = np.array(labels)
        yield X, y
        total_batches += 1
        if total_batches == max_batches:
            break


if __name__ == '__main__':
    # word2vec uses a vocabulary of 3M words and embeddings of size 300
    # Google trained it on 100B words from Google News
    # GoogleNews-vectors-negative300.bin used negative sampling and
    # the skip-gram architecture in favor of CBOW with a window size of
    # around 10 and subsampling of frequent words.
    vocab_size = 10000
    oov_token = '[OOV]'
    embedding_size = 300
    window_size = 10
    negative_samples = 20
    batch_size = 4096
    corpus = 'text8'
    try:
        vocabulary, word_counts = create_vocabulary(corpus, size=vocab_size, oov_token=oov_token)
    except FileNotFoundError:
        print("Download Text8 dataset. Run ./download_text8.sh")
        print(f"File needed: {corpus}")
        sys.exit(0)
    max_batches = None
    # max_batches = batch_size * 100
    # assert max_batches < batch_size or max_batches % batch_size == 0
    train_data = DataGeneratorWrapper(
        generate_skipgrams, text_file=corpus, vocabulary=vocabulary, word_counts=word_counts,
        window_size=window_size, negative_samples=negative_samples, batch_size=batch_size,
        oov_token=oov_token, max_batches=max_batches
    )
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
    # Approximate batches per epoch
    batches = int(
        (sum(word_counts.values()) * (2 * window_size + negative_samples) - sum(range(window_size)) * 2) *
        0.658 / batch_size
    )
    if max_batches is not None:
        batches_per_epoch = min(batches, max_batches)
    else:
        batches_per_epoch = batches
    trainer.fit(train_data, epochs=2, batches_per_epoch=batches_per_epoch)
    # Save the trained weights (word embeddings)
    skipgram[0].save('embeddings_skipgram.npy')
