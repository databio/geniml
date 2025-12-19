import logging
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError(
        "Please install Machine Learning dependencies by running 'pip install geniml[ml]'"
    )

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
    """Store numpy arrays of X and Y into a torch.DataLoader.

    Based on https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

    Args:
        X (np.ndarray): Embedding vectors of input data (natural language embeddings).
        Y (np.ndarray): Embedding vectors of output data (BED file embeddings).
        target (np.ndarray): Vector of 1 and -1, indicating if each vector pair of (X, Y)
            are target pairs or not.
        batch_size (int): Size of small batch.
        shuffle (bool): Shuffle dataset or not.

    Returns:
        DataLoader: A Dataset for pytorch training in format of torch.DataLoader.
    """
    tensor_X = torch.from_numpy(dtype_check(X))
    tensor_Y = torch.from_numpy(dtype_check(Y))
    tensor_target = torch.from_numpy(dtype_check(target))
    my_dataset = TensorDataset(tensor_X, tensor_Y, tensor_target)  # create your dataset

    return DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)


def dtype_check(vecs: np.ndarray) -> np.ndarray:
    """Convert numpy array dtype from float64 to float32 for PyTorch compatibility.

    Since the default float in numpy is float64, but in pytorch tensor it's float32,
    the dtype will be switched to avoid errors.

    Args:
        vecs (np.ndarray): Input numpy array.

    Returns:
        np.ndarray: Numpy array with dtype of float32.
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
    """Read selected columns from a metadata CSV and return a metadata dictionary.

    Can filter genomes with a given list of genomes and the column name of genome.

    Args:
        csv_path (str): Path to the csv file that contain metadata.
        col_names (Set[str]): Set of csv columns that contain informative metadata.
        file_key (str): Name of column of file names.
        genomes (Union[Set[str], None]): Set of genomes.
        genomes_key (Union[str, None]): Name of column of sample genomes.
        series_key (Union[str, None]): Name of column of series.
        chunk_size (Union[int, None]): Size of chunk to read when the csv file is large.

    Returns:
        Dict[str, Union[str, Dict[str, Union[str, List[str]]]]]: A metadata dictionary
            in one of two formats:

            If series information is in the csv:
            {
                <series>: [
                    {
                        "name": <file name>,
                        "metadata": {
                            <csv column name>: <metadata string>,
                            ...
                        }
                    },
                    ...
                ],
                ...
            }

            Otherwise:
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
