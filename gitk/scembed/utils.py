from enum import Enum
from collections import Counter
from logging import getLogger
from random import shuffle
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Dict

import pandas as pd
import scanpy as sc
from tqdm import tqdm

from .const import *
from .scembed import SCEmbed

_LOGGER = getLogger(MODULE_NAME)


class ScheduleType(Enum):
    """Learning rate schedule types"""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class LearningRateScheduler:
    """
    Simple class to track learning rates of the training procedure

    Based off of: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
    """

    def __init__(
        self,
        init_lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        type: Union[str, ScheduleType] = ScheduleType.EXPONENTIAL,
        decay: float = None,
        n_epochs: int = None,
    ):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.n_epochs = n_epochs

        # convert type to learning rate if necessary
        if isinstance(type, str):
            try:
                self.type = ScheduleType[type.upper()]
            except KeyError:
                raise ValueError(
                    f"Unknown schedule type: {type}. Must be one of ['linear', 'exponential']."
                )

        # init the current lr and iteration
        self._current_lr = init_lr
        self._iter = 1

        # init decay rate
        if decay is None:
            _LOGGER.warning(
                "No decay rate provided. Calculating decay rate from init_lr and n_epochs."
            )
            self.decay = init_lr / n_epochs
        else:
            self.decay = decay

    def _update_linear(self, epoch: int):
        lr = self.init_lr - (self.decay * epoch)
        return max(lr, self.min_lr)

    def _update_exponential(self, epoch: int):
        lr = self.get_lr() * (1 / (1 + self.decay * epoch))
        return max(lr, self.min_lr)

    def update(self):
        # update the learning rate according to the type
        if self.type == ScheduleType.LINEAR:
            self._current_lr = self._update_linear(self._iter)
            self._iter += 1
        elif self.type == ScheduleType.EXPONENTIAL:
            self._current_lr = self._update_exponential(self._iter)
            self._iter += 1
        else:
            raise ValueError(f"Unknown schedule type: {self.type}")

    def get_lr(self):
        return self._current_lr


class AnnDataChunker:
    def __init__(self, adata: sc.AnnData, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Simple class to chunk an AnnData object into smaller pieces. Useful for
        training on large datasets.

        :param adata: AnnData object to chunk. Must be in backed mode. See: https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html
        :param chunk_size: Number of cells to include in each chunk
        """
        self.adata = adata
        self.chunk_size = chunk_size
        self.n_chunks = len(adata) // chunk_size + 1

    def __iter__(self):
        for i in range(self.n_chunks):
            chunk = self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :]
            yield chunk.to_memory()

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, item):
        return self.adata[item * self.chunk_size : (item + 1) * self.chunk_size, :]

    def __repr__(self):
        return f"<AnnDataChunker: {self.n_chunks} chunks of size {self.chunk_size}>"


def shuffle_documents(
    documents: List[List[str]],
    n_shuffles: int,
    threads: int = None,
) -> List[List[str]]:
    """
    Shuffle around the genomic regions for each cell to generate a "context".

    :param List[List[str]] documents: the document list to shuffle.
    :param int n_shuffles: The number of shuffles to conduct.
    :param int threads: The number of threads to use for shuffling.
    """

    def shuffle_list(l: List[str], n: int) -> List[str]:
        for _ in range(n):
            shuffle(l)
        return l

    _LOGGER.debug(f"Shuffling documents {n_shuffles} times.")
    shuffled_documents = documents.copy()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        shuffled_documents = list(
            tqdm(
                executor.map(
                    shuffle_list, shuffled_documents, [n_shuffles] * len(documents)
                ),
                total=len(documents),
            )
        )
    return shuffled_documents


def remove_regions_below_min_count(
    region_sets: List[List[str]], min_count: int
) -> List[List[str]]:
    """
    Remove regions that don't satisfy the min count.

    TODO - this is a bit slow, could be sped up using dataframe operations.

    :param List[List[str]] region_sets: the region sets to remove regions from
    :param int min_count: the min count to use
    """
    # get the counts for each region
    region_counts = Counter()
    for region_set in region_sets:
        region_counts.update(region_set)

    # remove any regions that dont satisfy the min count
    region_sets = [
        [r for r in region_set if region_counts[r] >= min_count]
        for region_set in region_sets
    ]

    return region_sets


def load_scanpy_data(path_to_h5ad: str) -> sc.AnnData:
    """
    Load in the h5ad file that holds all of the information
    for our single-cell data with scanpy.

    :param str path_to_h5ad: the path to the h5ad file made with scanpy
    """
    return sc.read_h5ad(path_to_h5ad)


def extract_region_list(region_df: pd.DataFrame) -> List[str]:
    """
    Parse the `var` attribute of the scanpy.AnnData object and
    return a list of regions from the matrix. Converts each
    region to a string of the form `chr_start_end`.

    :param pandas.DataFrame region_df: the regions dataframe to parse
    """
    _LOGGER.info("Extracting region list from matrix.")
    regions = region_df.apply(
        lambda x: f"{x[CHR_KEY]}_{x[START_KEY]}_{x[END_KEY]}", axis=1
    ).tolist()
    return regions


def remove_zero_regions(cell_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Removes any key-value pairs in a dictionary where the value (copy number)
    is equal to zero (no signal). This is done using dictionary comprehension
    as it is much faster.

    :param cell_dict Dict[str, int]: the cell dictionary with region index keys and copy number values
    """
    return {k: v for k, v in cell_dict.items() if v > 0}


def convert_anndata_to_documents(
    anndata: sc.AnnData,
    use_defaults: bool = True,
) -> List[List[str]]:
    """
    Parses the scanpy.AnnData object to create the required "documents" object for
    training the Word2Vec model. Each row (or cell) is treated as a "document". That
    is, each region is a "word", and the total collection of regions is the "document".

    :param scanpy.AnnData anndata: the AnnData object to parse.
    :use_defaults bool: whether or not to use the default column names for the regions. this
                        is most commonly used when the AnnData object was created without using
                        the var attribute.
    """
    if use_defaults:
        regions_parsed = [f"r{i}" for i in range(anndata.var.shape[0])]
        anndata.var["region"] = regions_parsed
    else:
        regions_parsed = extract_region_list(anndata.var)
    sc_df = anndata.to_df()
    docs = []
    _LOGGER.info("Generating documents.")

    def process_row(row):
        row_dict = row.to_dict()
        row_dict = remove_zero_regions(row_dict)
        new_doc = []
        for region_indx in row_dict:
            region_str = regions_parsed[int(region_indx)]
            new_doc.append(region_str)
        return new_doc

    docs = sc_df.apply(process_row, axis=1).tolist()

    return docs


def load_scembed_model(path: str) -> SCEmbed:
    """
    Load a scembed model from disk.

    :param str path: The path to the model.
    """
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)
