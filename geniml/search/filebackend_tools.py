import logging
import pickle
from typing import Dict, List, Set, Tuple, Union

import numpy as np

from ..const import PKG_NAME
from .backends.filebackend import HNSWBackend

_LOGGER = logging.getLogger(PKG_NAME)


def merge_backends(
    backends_to_merge: List[HNSWBackend], local_index_path: str, dim: int
) -> HNSWBackend:
    """
    Merge multiple backends into one

    :param backends_to_merge: a list of [HNSWBackend]
    :param local_index_path: the path to the local index file of the merged output HNSWBackend
    :param dim: the dimension of vectors stored in the HNSWBackend

    :return: a HNSWBackend that comes from merge all HNSWBackend in the input list backends_to_merge
    """

    result_backend = HNSWBackend(
        local_index_path=local_index_path,
        payloads={},
        dim=dim,
    )

    for backend in backends_to_merge:
        result_vecs = []
        result_payloads = []
        for j in range(len(backend)):
            result_vecs.append(backend.idx.get_items([j], return_type="numpy")[0])
            result_payloads.append(backend.payloads[j])

        result_backend.load(vectors=np.array(result_vecs), payloads=result_payloads)

    return result_backend
