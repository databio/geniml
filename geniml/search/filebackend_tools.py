import logging
from typing import List

import numpy as np

from ..const import *
from .backends.filebackend import HNSWBackend

_LOGGER = logging.getLogger(PKG_NAME)


def merge_backends(
    backends_to_merge: List[HNSWBackend], local_index_path: str, dim: int
) -> HNSWBackend:
    """
    merge multiple backends into one
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
