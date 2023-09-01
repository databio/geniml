from typing import Dict, List

import numpy as np


def verify_load_inputs(embeddings: np.ndarray, labels: List[Dict[str, str]]):
    if embeddings.shape[0] != len(labels):
        raise KeyError("The number of embeddings does not match the number of labels")
