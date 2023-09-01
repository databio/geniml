from typing import List, Union

import numpy as np


def mean_pooling(region_vectors: List[Union[np.ndarray, None]]) -> np.ndarray:
    """
    Mean pooling of region vectors.

    The function first filters out None values and then computes the mean of the remaining vectors.

    :param region_vectors | None: region vectors
    :return: mean pooled vector
    """
    region_vectors = [rv for rv in region_vectors if rv is not None]
    if len(region_vectors) == 0:
        return None

    region_vectors = np.array(region_vectors)
    return np.mean(region_vectors, axis=0)


def max_pooling(region_vectors: List[Union[np.ndarray, None]]) -> np.ndarray:
    """
    Max pooling of region vectors.

    The function first filters out None values and then computes the max of the remaining vectors.

    :param region_vectors | None: region vectors
    :return: max pooled vector
    """
    region_vectors = [rv for rv in region_vectors if rv is not None]
    if len(region_vectors) == 0:
        return None

    region_vectors = np.array(region_vectors)
    return np.max(region_vectors, axis=0)
