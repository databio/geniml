import logging
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .const import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATALOADER_SHUFFLE,
    DEFAULT_FILE_KEY,
    DEFAULT_GENOME_KEY,
    DEFAULT_SERIES_KEY,
    MODULE_NAME,
)

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

    :param X: embedding vectors of input data (natural language embeddings)
    :param Y: embedding vectors of output data (BED file embeddings)
    :param target: vector of 1 and -1, indicating if each vector pair of (X, Y) are target pairs or not
    :param batch_size: size of small batch
    :param shuffle: shuffle dataset or not
    :return: a Dataset for pytorch training in format of torch.DataLoader
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

    :param vecs: input np.ndarray

    :return: np.ndarray with dtype of float32
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

    :param csv_path: path to the csv file that contain metadata
    :param col_names: set of csv columns that contain informative metadata
    :param file_key: name of column of file names
    :param genomes: set of genomes
    :param genomes_key: name of column of sample genomes
    :param series_key: name of column of series
    :param chunk_size: size of chunk to read when the csv file is large

    :return: a metadata dictionary in this format:
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
