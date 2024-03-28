import logging
from typing import Dict, List, Tuple, Union

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


def sample_non_target_vec(
    max_id: int, matching_ids: List[Union[int, np.int64]], size: int
) -> List[Union[int, np.int64]]:
    """
    Sample non-matching vectors pairs for contrastive loss

    Args:
        max_id: maximum id = total number of vectors - 1
        matching_ids: ids of matching vectors (target pairs)
        size: number of samples

    Returns: a list of ids

    """

    # sample range
    if (size + len(matching_ids)) > max_id + 1:
        _LOGGER.error("IndexError: Sample size + matching size should below the maximum ID")

    full_range = np.arange(0, max_id + 1)

    # skipping ids of matching vectors
    eligible_integers = np.setdiff1d(full_range, matching_ids)
    # sample result
    sampled_integer = np.random.choice(eligible_integers, size=size, replace=False)

    return list(sampled_integer)


def reverse_payload(payload: Dict[np.int64, Dict], target_key: str) -> Dict[str, np.int64]:
    """
    The payload dictionary of a HNSWBackend is in this format:
    {
        <store id>: <metadata dictionary of that vector>,
        ...
    }
    This function will return a reversal dictionary, in which each key is one value in the metadata dict,
    and each value is storage id.

    For example, if the payload dictionary is:
    {
       1: {
           "name": "A0001.bed",
           "summary": <summary>,
           ...
       }
    }

    if target_key is "name", the output will be:
    {
        "A0001.bed": 1,
    }

    Args:
        payload: payload dictionary of a HNSWBackend
        target_key: a key in metadata dictionary

    Returns: the reversal payload dictionary

    """
    output_dict = dict()
    for i in payload.keys():
        output_dict[payload[i][target_key]] = i

    return output_dict


def vec_pairs(
    nl_backend: HNSWBackend,
    bed_backend: HNSWBackend,
    bed_payload_key: str = "name",
    nl_payload_key: str = "files",
    non_target_pairs: bool = False,
    non_target_pairs_prop: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract vector pairs for training / validating from file backends that store embedding vectors of BED and metadata.
    The payloads of BED backend must contain the file name of each embedding vector.
    The payloads of metadata backend must contain names of matching files of each metadata string.

    Args:
        nl_backend: backend where embedding vectors of natural language metadata are stored
        bed_backend: backend where embedding vectors of BED files are stored
        bed_payload_key: the key of BED file name in the payload of BED embedding backend
        nl_payload_key: the key of matching BED files in the payload of metadata embedding backend
        non_target_pairs: whether non-target pairs will be sampled
        non_target_pairs_prop: proportion of <number of non-target pairs> :  <number of target pairs>

    Returns:

    """
    # maximum id of metadata embedding vectors
    max_num_nl = nl_backend.idx.get_max_elements()

    # maximum id of BED embedding vectors
    max_num_bed = bed_backend.idx.get_max_elements()

    # List of embedding vectors
    X = []
    Y = []

    # list of 1 and -1, indicate whether the vector pair is target pair or not
    target = []

    # reverse the BED backend payload dictionary into {<file name>: store id}
    bed_reversal_payload = reverse_payload(bed_backend.payloads, bed_payload_key)

    # pair vectors
    for i in range(max_num_nl):
        nl_vec = nl_backend.idx.get_items([i])[0]
        bed_vec_ids = []
        # get target pairs
        for file_name in nl_backend.payloads[i][nl_payload_key]:
            try:
                bed_vec_ids.append(bed_reversal_payload[file_name])
            except:
                continue
        if len(bed_vec_ids) == 0:
            continue
        bed_vecs = bed_backend.idx.get_items(bed_vec_ids, return_type="numpy")
        for y_vec in bed_vecs:
            X.append(nl_vec)
            Y.append(y_vec)
            target.append(1)

        # sample non target pairs if needed for contrastive loss
        if non_target_pairs:
            non_match_ids = sample_non_target_vec(
                max_num_bed, bed_vec_ids, int(non_target_pairs_prop * len(bed_vec_ids))
            )
            print(f"Non match ids: {non_match_ids}")
            non_match_vecs = bed_backend.idx.get_items(non_match_ids, return_type="numpy")
            for y_vec in non_match_vecs:
                X.append(nl_vec)
                Y.append(y_vec)
                target.append(-1)

    return np.array(X), np.array(Y), np.array(target)
