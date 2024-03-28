import logging
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..search.backends.filebackend import HNSWBackend
from .const import *

_LOGGER = logging.getLogger(MODULE_NAME)


def arrays_to_torch_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    target: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = DEFAULT_DATALOADER_SHUFFLE,
) -> DataLoader:
    """
    Based on https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

    Store np.ndarray of X and Y into a torch.DataLoader

    Args:
        X: embedding vectors of input data (natural language embeddings)
        Y: embedding vectors of output data (BED file embeddings)
        target: vector of 1 and -1, indicating if each vector pair of (X, Y) are target pairs or not
        batch_size: size of small batch
        shuffle: shuffle dataset or not

    Returns:

    """
    tensor_X = torch.from_numpy(dtype_check(X))
    tensor_Y = torch.from_numpy(dtype_check(Y))
    tensor_target = torch.from_numpy(dtype_check(target))
    my_dataset = TensorDataset(tensor_X, tensor_Y, tensor_target)  # create your dataset

    return DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)


def dtype_check(vecs: np.ndarray) -> np.ndarray:
    """
    Since the default float in np is float64, but in pytorch tensor it's float32,
    to avoid errors, the dtype will be switched

    Args:
        vecs:

    Returns: np.ndarray with dtype of float32

    """
    if not isinstance(vecs.dtype, type(np.dtype("float32"))):
        vecs = vecs.astype(np.float32)

    return vecs


def metadata_dict_from_csv(
    csv_path: str,
    col_names: Set[str],
    file_key: str = DEFAULT_FILE_KEY,
    genomes: Union[Set[str], None] = None,
    genomes_key: Union[str, None] = DEFAULT_GENOME_KEY,
    series_key: Union[str, None] = DEFAULT_SERIES_KEY,
    chunk_size: Union[int, None] = None,
) -> Dict[str, Union[str, Dict[str, Union[str, List[str]]]]]:
    """
    Read selected columns from a metadata csv and return metadata dictionary,
    can filter genomes with given list of genomes and the column name of genome

    Args:
        csv_path: path to the csv file that contain metadata
        col_names: list of columns that contain informative metadata
        file_key: name of column of file names
        genomes: list of genomes
        genomes_key: name of column of sample genomes
        chunk_size: size of chunk to read when the csv file is large
        series_key: name of column of series

    Returns: a dictionary that contains metadata,

    if series information is in the csv, the dictionary format will be:
    {
        <series>:[
            {
                "name": <file name>
                "metadata": {
                    <csv column name>: <metadata string>,
                    ...
                }
            },
            ...
        ],
        ...
    }

    else, the dictionary format will be:
    {
        <file name>: {
            <csv column name>: <metadata string>,
            ...
        },
        ...
    }

    """
    # dictionary to store data
    output_dict = dict()
    # count number of series, files, and csv chunks
    series_count = 0
    bed_count = 0
    text_count = 0
    empty_count = 0
    read_chunk = True
    # read csv
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # if chunk size is None
        if isinstance(chunk, str):
            read_chunk = False
            rows_to_ite = pd.read_csv(csv_path)
        else:
            rows_to_ite = chunk

        for index, row in rows_to_ite.iterrows():
            genome_filter = True
            # select genome if list of genomes and genome key are given
            if genomes is not None and genomes_key is not None:
                if row[genomes_key].strip() not in genomes:
                    genome_filter = False

            if genome_filter:
                # collect metadata
                metadata_dict = dict()

                for col in col_names:
                    if isinstance(row[col], str):  #
                        text_count += 1
                        metadata_dict[col] = row[col]

                if len(metadata_dict) == 0:
                    empty_count += 1
                # add the metadata into output dictionary
                else:
                    if series_key is None or not series_key in rows_to_ite.columns:
                        output_dict[row[file_key]] = metadata_dict

                    else:
                        payload = {
                            "name": row[file_key],
                            "metadata": metadata_dict,
                        }
                        try:
                            output_dict[row[series_key]].append(payload)
                        except:
                            output_dict[row[series_key]] = [payload]
                            series_count += 1
                bed_count += 1
        if not read_chunk:
            break

    # output of summary statistics
    if series_key is not None:
        _LOGGER.info(f"Number of series: {series_count}")

    _LOGGER.info(f"Number of files: {bed_count}")
    _LOGGER.info(f"Number of metadata strings: {text_count}")
    _LOGGER.info(f"Number of files with 0 metadata strings: {empty_count}")

    return output_dict


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
