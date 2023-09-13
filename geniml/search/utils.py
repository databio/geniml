from typing import Dict, List, Union

import numpy as np


def verify_load_inputs(vectors: np.ndarray, ids: List[Union[str, int]], payloads: List[Dict[str, str]]):
    n_ids = len(ids)
    n_vectors = vectors.shape[0]
    n_payloads = len(payloads)
    if n_ids != n_vectors or n_ids != n_payloads:
        raise ValueError(
            "The number of ids ({n_ids}), vectors ({n_vectors}), and payloads ({n_payloads}) must match"
        )
