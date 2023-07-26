import os
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from logging import getLogger
from random import shuffle
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

if TYPE_CHECKING:
    from .main import SCEmbed

from .const import *

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
        """
        :param float init_lr: The initial learning rate
        :param float min_lr: The minimum learning rate
        :param str type: The type of learning rate schedule to use. Must be one of ['linear', 'exponential'].
        :param float decay: The decay rate to use. If None, this will be calculated from init_lr and n_epochs.
        :param int n_epochs: The number of epochs to train for. Only used if decay is None.
        """
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
        """
        Update the learning rate using a linear schedule.

        :param int epoch: The current epoch
        """

        lr = self.init_lr - (self.decay * epoch)
        return max(lr, self.min_lr)

    def _update_exponential(self, epoch: int):
        """
        Update the learning rate using an exponential schedule.

        :param int epoch: The current epoch
        """
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

        :param sc.AnnData adata: AnnData object to chunk. Must be in backed mode. See: https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html
        :param int chunk_size: Number of cells to include in each chunk
        """
        self.adata = adata
        self.chunk_size = chunk_size
        self.n_chunks = len(adata) // chunk_size + 1

    def __iter__(self):
        for i in range(self.n_chunks):
            # check for shape = 0
            if self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :].shape[0] == 0:
                return
            yield self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :]

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, item: int):
        """
        Get a chunk of the AnnData object.

        :param int item: The chunk index to get.
        """
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
                    shuffle_list,
                    shuffled_documents,
                    [n_shuffles] * len(documents),
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
        [r for r in region_set if region_counts[r] >= min_count] for region_set in region_sets
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
    regions = [
        f"{row[CHR_KEY]}_{row[START_KEY]}_{row[END_KEY]}" for _, row in region_df.iterrows()
    ]
    return regions


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
    # enable progress bar
    tqdm.pandas()

    if use_defaults:
        regions_parsed = [f"r{i}" for i in range(anndata.var.shape[0])]
        # drop var attribute since it messes with things
        anndata.var.drop(anndata.var.columns, axis=1, inplace=True)
        anndata.var.reset_index(inplace=True)

        # add the region column back in
        anndata.var["region"] = regions_parsed
    else:
        regions_parsed = extract_region_list(anndata.var)
    sc_df = anndata.to_df()
    _LOGGER.info("Generating documents.")

    docs = [
        [
            regions_parsed[int(region_indx)]
            for region_indx, value in row.to_dict().items()
            if value > 0
        ]
        for _, row in sc_df.iterrows()
    ]

    return docs


def load_scembed_model(path: str) -> "SCEmbed":
    """
    Load a scembed model from disk.

    :param str path: The path to the model.
    """
    import gzip
    import pickle

    with gzip.open(path, "rb") as f:
        model = pickle.load(f)
        return model


def load_scembed_model_deprecated(path: str) -> "SCEmbed":
    """
    Load a scembed model from disk. THIS IS DEPRECATED AND WILL BE REMOVED IN THE FUTURE.

    :param str path: The path to the model.
    """
    import pickle

    with open(path, "rb") as f:
        model = pickle.load(f)
        return model


def anndata_to_regionsets(adata: sc.AnnData) -> List[List[str]]:
    """
    Converts an AnnData object to a list of lists of regions. This
    is done by taking each cell and creating a list of all regions
    that have a value greater than 0.

    *Note: this method requires that the sc.AnnData object have
    chr, start, and end in `.var` attributes*

    :param sc.AnnData adata: the AnnData object to convert
    """
    # Extract the arrays for chr, start, and end
    chr_values = adata.var["chr"].values
    start_values = adata.var["start"].values
    end_values = adata.var["end"].values

    # Perform the comparison using numpy operations
    positive_values = adata.X > 0

    if not isinstance(positive_values, np.ndarray):
        positive_values = positive_values.toarray()

    regions = []
    for i in tqdm(range(adata.shape[0]), total=adata.shape[0]):
        regions.append(
            [
                f"{chr_values[j]}_{start_values[j]}_{end_values[j]}"
                for j in np.where(positive_values[i])[0]
            ]
        )
    return regions


def barcode_mtx_peaks_to_anndata(
    barcodes_path: str,
    mtx_path: str,
    peaks_path: str,
    transpose: bool = True,
    sparse: bool = True,
) -> sc.AnnData:
    """
    This function will take three files:

    1. a barcodes file (.tsv)
    2. a matrix file (.mtx)
    3. a peaks file (.bed)

    And turn them into an AnnData object. It will attach the peaks
    as a .var attribute (chr, start, end) and the barcodes as a .obs attribute (index)

    :param str barcodes_path: the path to the barcodes file
    :param str mtx_path: the path to the matrix file
    :param str peaks_path: the path to the peaks file
    :param bool transpose: whether or not to transpose the matrix
    :param bool sparse: whether or not the matrix is sparse

    :return sc.AnnData: an AnnData object
    """
    from scipy.io import mmread

    # load the barcodes
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)
    barcodes.columns = ["barcode"]
    barcodes.index = barcodes["barcode"]

    # load the mtx
    mtx = mmread(mtx_path)
    if transpose:
        mtx = mtx.T

    # load the peaks
    peaks = pd.read_csv(peaks_path, sep="\t", header=None)
    peaks.columns = ["chr", "start", "end"]

    # create the AnnData object
    adata = sc.AnnData(X=mtx, obs=barcodes, var=peaks)

    return adata
