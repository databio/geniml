import logging
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np

from geniml.const import *

from .backends.filebackend import HNSWBackend

_LOGGER = logging.getLogger(PKG_NAME)


def load_local_backend(bin_path: str, pkl_path: str, dim: int) -> HNSWBackend:
    """
    Load a HNSWBackend from the local index file and the saved payloads dictionary

    :param bin_path: path of the index file (.bin)
    :param pkl_path: path of saved payloads (.pkl)
    :param dim: the dimension of vectors stored in the hnsw index

    :return: the HNSWBackend from saved index and payloads
    """
    payloads = pickle.load(open(pkl_path, "rb"))
    return HNSWBackend(local_index_path=bin_path, payloads=payloads, dim=dim)


def merge_backends(
    backends_to_merge: List[HNSWBackend], local_index_path: str, dim: int
) -> HNSWBackend:
    """
    merge multiple backends into one

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


def sample_non_target_vec(
    max_id: int, matching_ids: List[Union[int, np.int64]], size: int
) -> List[Union[int, np.int64]]:
    """
    When the torch loss function for text2bednn training is CosineEmbeddingLoss,
    the goal is to maximize the cosine similarity when the input metadata (embedding) matches
    the input BED file (embedding), and minimize otherwise. Therefore, besides target pairs
    (matching metadata embedding and BED embedding), non-target pairs also need sampling.
    This function samples ids of non-matching vectors in the backend.

    :param max_id: maximum id = total number of vectors - 1
    :param matching_ids: ids of matching vectors (target pairs)
    :param size: number of vectors to sample

    :return: a list of ids in the backend.
    """

    # sample range
    if (size + len(matching_ids)) > max_id:
        _LOGGER.error("IndexError: Sample size + matching size should below the maximum ID")

    full_range = np.arange(0, max_id)

    # skipping ids of matching vectors
    eligible_integers = np.setdiff1d(full_range, matching_ids)
    # sample result
    sampled_integer = np.random.choice(eligible_integers, size=size, replace=False)

    return list(sampled_integer)


def reverse_payload(payload: Dict[np.int64, Dict], target_key: str) -> Dict[str, np.int64]:
    """

    :param payload: payload dictionary of a HNSWBackend, in this format:
        {
            <store id>: <metadata dictionary of that vector>,
            ...
        }
    :param target_key: a key in metadata dictionary
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


    :return: the reversal payload dictionary
    """
    output_dict = dict()
    for i in payload.keys():
        output_dict[payload[i][target_key]] = i

    return output_dict


def vec_pairs(
    nl_backend: HNSWBackend,
    bed_backend: HNSWBackend,
    nl_payload_key: str = "files",
    bed_payload_key: str = "name",
    non_target_pairs: bool = False,
    non_target_pairs_prop: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The training of geniml.text2bednn needs pairs of vectors (natural language embeddings & region set embeddings).
    This function extract vector pairs file backends that store embedding vectors of region set (BED) and metadata.
    The payloads of BED backend must contain the file name of each embedding vector.
    The payloads of metadata backend must contain names of matching files of each metadata string.

    :param nl_backend: backend where embedding vectors of natural language metadata are stored
    :param bed_backend: backend where embedding vectors of BED files are stored
    :param nl_payload_key: the key of matching BED files in the payload of metadata embedding backend
    :param bed_payload_key: the key of BED file name in the payload of BED embedding backend
    :param non_target_pairs: whether non-target pairs will be sampled, for details, see the docstring of sample_non_target_vec()
    :param non_target_pairs_prop: proportion of <number of non-target pairs> / <number of target pairs>

    :return: A tuple of 3 np.ndarrays:
        X: with shape of (n, <natural language embedding dimension>)
        Y: with shape of (n, <region set embedding dimension>)
        target: with shape of (n,), contain only 1 and -1, indicating if the X-Y vector pair is target or not
        (see the docstring of sample_non_target_vec())

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
            non_match_vecs = bed_backend.idx.get_items(non_match_ids, return_type="numpy")
            for y_vec in non_match_vecs:
                X.append(nl_vec)
                Y.append(y_vec)
                target.append(-1)

    return np.array(X), np.array(Y), np.array(target)
