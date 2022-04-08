import numpy as np
import random

from typing import Generator, List, Optional, Tuple, Union, Sequence


def to_one_hot(vector: Union[List[int], Tuple[int, ...], np.ndarray],
               n_columns: Optional[int] = None) -> np.ndarray:
    """Convert array of values into one-hot vector."""
    vector = np.array(vector).squeeze()
    assert len(vector.shape) == 1, "`vector` passed is multi-dimensional"
    if not n_columns:
        n_columns = np.max(vector) + 1
    one_hot = np.zeros((vector.shape[0], n_columns))
    one_hot[np.arange(vector.shape[0]), vector] = 1
    return one_hot


def split_in_batches(features: np.ndarray,
                     targets: np.ndarray,
                     batch_size: int = 32,
                     shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Ideally, you would create your own generator so as not to
    load the entire dataset in memory. Here, all the features
    and targets are passed. Wrap it using
    models.DataGeneratorWrapper.
    """
    assert features.shape[0] == targets.shape[0], "Incompatible feature and targets sizes"
    assert batch_size <= features.shape[0], "Not enough samples for batch size"
    indexes = list(range(features.shape[0]))
    if shuffle:
        random.shuffle(indexes)
    for batch in range(int(features.shape[0] / batch_size)):
        start = batch * batch_size
        yield features[indexes[start:start+batch_size], ...], targets[indexes[start:start+batch_size], ...]


def cosine_similarity(x: Union[List[int], Tuple[int, ...], np.ndarray],
                      y: Union[List[int], Tuple[int, ...], np.ndarray]) -> float:
    """Compute cosine similarity of 2 vectors."""
    x = np.array(x).squeeze()
    y = np.array(y).squeeze()
    assert len(x.shape) == 1, "First vector is multi-dimensional"
    assert len(y.shape) == 1, "Second vector is multi-dimensional"
    assert x.shape == y.shape, "Vector dimensions are not equal"
    cos = np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)
    return float(cos)


# def generate_projector_files(embeddings: np.ndarray, vocabulary: )