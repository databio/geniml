import scanpy as sc
import pandas as pd

from typing import Dict, Iterable, List
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from gensim.models import Word2Vec
from numba import config
from logging import getLogger
from tqdm import tqdm

from .const import *

_LOGGER = getLogger(PKG_NAME)

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"

# shuffle the document to generate data for word2vec
def shuffle_documents(
    documents: List[List[str]],
    n_shuffles: int,
    threads: int = 1,
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
                    [n_shuffles] * len(documents)
                ),
                total=len(documents),
            )
        )
    return shuffled_documents


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
    _LOGGER.info("Extracting region list from matrix.")
    regions_parsed = []
    for r in tqdm(region_df.iterrows(), total=region_df.shape[0]):
        r_dict = r[1].to_dict()
        regions_parsed.append(
            " ".join([r_dict["chr"], str(r_dict["start"]), str(r_dict["end"])])
        )
    return regions_parsed


def remove_zero_regions(cell_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Removes any key-value pairs in a dictionary where the value (copy number)
    is equal to zero (no signal). This is done using dictionary comprehension
    as it is much faster.

    :param cell_dict Dict[str, int]: the cell dictionary with region index keys and copy number values
    """
    return {k: v for k, v in cell_dict.items() if v > 0}


def convert_anndata_to_documents(anndata: sc.AnnData) -> List[List[str]]:
    """
    Parses the scanpy.AnnData object to create the required "documents" object for
    training the Word2Vec model. Each row (or cell) is treated as a "document". That
    is, each region is a "word", and the total collection of regions is the "document".

    :param scanpy.AnnData anndata: the AnnData object to parse.
    """
    regions_parsed = extract_region_list(anndata.var)
    sc_df = anndata.to_df()
    docs = []
    _LOGGER.info("Generating documents.")

    for row in tqdm(sc_df.iterrows(), total=sc_df.shape[0]):
        row_dict = row[1].to_dict()
        row_dict = remove_zero_regions(row_dict)
        new_doc = []
        for region_indx in row_dict:
            region_str = regions_parsed[int(region_indx)]
            new_doc.append(region_str)
        docs.append(new_doc)
    return docs
