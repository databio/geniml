from typing import Dict, Iterable, List

# ignore import errors for six
# see here: https://stackoverflow.com/questions/36213989/getting-six-and-six-moves-modules-to-autocomplete-in-pycharm
from six.moves import cPickle as pickle  # for performance
from gensim.models import Word2Vec
from numba import config

import numpy as np
import pandas as pd
import os
from logging import getLogger

from tqdm import tqdm

import scanpy as sc

from .const import *

_LOGGER = getLogger(PKG_NAME)

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"

# shuffle the document to generate data for word2vec
def shuffle_documents(
    documents: Dict[str, List[str]], shuffle_repeat: int
) -> List[List[str]]:
    """
    Shuffle around the genomic regions for each cell to generate a "context".

    :param Dict[str, List[str]] documents: the document dictionary to shuffle.
    :param int shuffle_repeat: The number of shuffles to conduct.
    """
    _LOGGER.debug(f"Shuffling documents {shuffle_repeat} times.")
    common_text = list(documents.values())
    training_samples = []
    training_samples.extend(common_text)
    for _ in range(shuffle_repeat):
        [(np.random.shuffle(l)) for l in common_text]
        training_samples.extend(common_text)
    return training_samples


def train(
    documents: Iterable[Iterable[str]],
    window_size: int = 100,
    dim: int = 128,
    min_count: int = 10,
    nothreads: int = 1,
) -> Word2Vec:
    """
    Train the Word2Vec algorithm on the region's

    :param Iterable[Iterable[str]] documents: this is the list of lists of regions that are to be shuffled
    :param int window_size: the context window size for the algorithm when training.
    :param int dim: the embeddings vector dimensionality.
    :param int min_count: Ignores all regions with total frequency lower than this.
    :param int nothreads: number of threads to train with.
    """
    # sg=0 Training algorithm: 1 for skip-gram; otherwise CBOW.
    message = (
        f"Training Word2Vec embeddings with {window_size} window size, "
        f"at {dim} dimensions, with a minimum 'word' count of "
        f"{min_count} and using {nothreads} threads."
    )
    _LOGGER.debug(message)
    model = Word2Vec(
        sentences=documents,
        window=window_size,
        size=dim,
        min_count=min_count,
        workers=nothreads,
    )
    return model


# preprocess the labels
def label_preprocessing(y: List[str], delim: str) -> List[str]:
    """
    Split the list of cell annotations on the delimiter.

    :param str y: List of cell annotations/names
    :param str delim: delimiter to split on
    """
    y_cell = []
    for y1 in y:
        y_cell.append(y1.split(delim)[0])
    return y_cell

def save_dict(di_: dict, filename_: str):
    """
    Using pickle, save the document dictionary to disk.

    :param dict di_: dictionary to save
    :param str filename_: the file to save to
    """
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_: str) -> dict:
    """
    Load the document dictionary saved to disk.

    :param str filename_: the file to load with pickle.
    """
    with open(filename_, "rb") as f:
        try:
            ret_di = pickle.load(f)
            return ret_di
        except EOFError:
            _LOGGER.error(
                "Size (In bytes) of '%s':" % filename_, os.path.getsize(filename_)
            )


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
    return a list of regions from the matrix

    :param pandas.DataFrame region_df: the regions dataframe to parse
    """
    regions_parsed = []
    for r in tqdm(region_df.iterrows()):
        r_dict = r[1].to_dict()
        regions_parsed.append(
            " ".join([r_dict["chr"], str(r_dict["start"]), str(r_dict["end"])])
        )
    return regions_parsed


def extract_cell_list(cell_df: pd.DataFrame) -> List[str]:
    """
    Parses the `obs` attribute of the scanpy.AnnData object and
    returns a list of the cell identifiers.

    :param cell_df pandas.DataFrame: the cell dataframe to parse
    """
    cells_parsed = []
    for c in tqdm(cell_df.iterrows()):
        c_dict = c[1].to_dict()
        cells_parsed.append(c_dict["cell-annotation"])
    return cells_parsed


def remove_zero_regions(cell_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Removes any key-value pairs in a dictionary where the value (copy number)
    is equal to zero (no signal). This is done using dictionary comprehension
    as it is much faster.

    :param cell_dict Dict[str, int]: the cell dictionary with region index keys and copy number values
    """
    return {k: v for k, v in cell_dict.items() if v > 0}


def convert_anndata_to_documents(anndata: sc.AnnData) -> Dict[str, List[str]]:
    """
    Parses the scanpy.AnnData object to create the required "documents" object for
    training the Word2Vec model.

    :param scanpy.AnnData anndata: the AnnData object to parse.
    """
    regions_parsed = extract_region_list(anndata.var)
    cells_parsed = extract_cell_list(anndata.obs)
    sc_df = anndata.to_df()
    _docs = {}
    _LOGGER.info("Generating documents.")

    for indx, row in tqdm(enumerate(sc_df.iterrows())):
        row_dict = row[1].to_dict()
        cell_label = cells_parsed[indx]
        _docs[cell_label] = []
        row_dict = remove_zero_regions(row_dict)
        for region_indx in row_dict:
            region_str = regions_parsed[int(region_indx)]
            _docs[cell_label].append(region_str)

    return _docs
