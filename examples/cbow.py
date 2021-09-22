import nltk
import numpy as np
import os
import pickle
import re
import sys

from collections import Counter

# Import module directly from root folder
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from poormanstransformers.utils import to_one_hot
from poormanstransformers.layers import Embedding, LogSoftmax, Dense, AxisMean
from poormanstransformers.losses import CategoricalCrossEntropy
from poormanstransformers.optimizers import Adam
from poormanstransformers.train import Trainer, DataGeneratorWrapper

STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', '.']


def create_vocabulary(text_file, size=6400, oov_token='[OOV]', save_file='vocabulary.pkl'):
    """Create a dictionary that maps a word to a unique integer."""
    if os.path.isfile(save_file):
        with open(save_file, 'rb') as handler:
            return pickle.load(handler)
    print("Generating vocabulary dictionary. This will take several minutes...")
    with open(text_file) as handler:
        data = handler.read()
    data = re.sub(r'[,!?;-]', '.', data)  # Replace all punctuation with "."
    data = nltk.word_tokenize(data)  # Tokenize string to words
    data = [ch.lower() for ch in data if ch.isalpha() and ch.lower() not in STOPWORDS]
    word_counts = Counter(data)
    vocabulary = {word_count[0]: i for i, word_count in enumerate(word_counts.most_common(size - 1))}
    vocabulary[oov_token] = size - 1
    with open(save_file, 'wb') as handler:
        pickle.dump(vocabulary, handler)
    print("Vocabulary generated!")
    return vocabulary


def generate_cbow_data(text_file, vocabulary, window_size=2, batch_size=64, oov_token='[OOV]', max_batches=None):
    """Generate CBOW model batches."""
    size = len(vocabulary)
    total_batches = 0
    with open(text_file) as handler:
        features = []
        targets = []
        batch_ctr = 0
        while True:
            data = handler.readline()
            data = re.sub(r'[,!?;-]', '.', data)
            data = nltk.word_tokenize(data)
            data = [ch.lower() for ch in data if ch.isalpha() and ch.lower() not in STOPWORDS]
            if len(data) >= window_size * 2 + 1:
                for target_index in range(window_size, len(data) - window_size):
                    features.append([
                        vocabulary.get(data[target_index + i], vocabulary[oov_token])
                        for i in range(-window_size, 0)
                    ] + [
                        vocabulary.get(data[target_index + i], vocabulary[oov_token])
                        for i in range(1, window_size + 1)
                    ])
                    targets.append(vocabulary.get(data[target_index], vocabulary[oov_token]))
                    batch_ctr += 1
                    if batch_ctr == batch_size:
                        total_batches += 1
                        yield np.array(features), to_one_hot(targets, size)
                        if max_batches and total_batches >= max_batches:
                            return
                        features = []
                        targets = []
                        batch_ctr = 0


if __name__ == '__main__':
    # word2vec uses a vocabulary of 3M words and embeddings of size 300
    # Google trained it on 100B words from Google News
    vocab_size = 6400
    oov_token = '[OOV]'
    embedding_size = 64
    window_size = 2
    batch_size = 512
    corpus = 'wikisent2.txt'
    try:
        vocabulary = create_vocabulary(corpus, size=vocab_size, oov_token=oov_token)
    except FileNotFoundError:
        print("Download Wikipedia Sentences dataset from: https://www.kaggle.com/mikeortman/wikipedia-sentences")
        print(f"File needed: {corpus}")
        sys.exit(0)
    batches = 2000
    train_data = DataGeneratorWrapper(
        generate_cbow_data, text_file=corpus, vocabulary=vocabulary,
        batch_size=batch_size, window_size=window_size, oov_token=oov_token,
        max_batches=batches
    )
    cbow = [
        Embedding(vocab_size=vocab_size, d_feature=embedding_size),
        AxisMean(1),  # Lambda(lambda x: np.mean(x, axis=1)),
        Dense(vocab_size),
        LogSoftmax()
    ]
    trainer = Trainer(
        model=cbow,
        optimizer=Adam(),
        loss=CategoricalCrossEntropy(from_logits=True),
    )
    trainer.fit(train_data, epochs=1, batches_per_epoch=batches)
    # Save the trained weights (word embeddings)
    cbow[0].save('embeddings.npy')
