from typing import Dict, Iterable, List
from six.moves import cPickle as pickle  # for performance
from gensim.models import Word2Vec
from numba import config
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure pyqt5 is upgraded (pip install --user --upgrade pyqt5
# import csv
import numpy as np
import pandas as pd
import os
from logging import getLogger
import umap
from tqdm import tqdm

import scanpy as sc

from .const import *

_LOGGER = getLogger(PKG_NAME)

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"


def embedding_avg(model, document):
    _LOGGER.debug(f"embedding_avg model: {model}")
    listOfWVs = []
    if type(document) is list:
        _LOGGER.debug(f"embedding_avg document is a list")
        for word in document:
            if word in model.wv.vocab:
                listOfWVs.append(model[word])
        if len(listOfWVs) == 0:
            return np.zeros([len(model[list(model.wv.vocab.keys())[0]])])
    else:
        _LOGGER.debug(f"embedding_avg document is NOT a list")
        for word in document.split(" "):
            if word in model.wv.vocab:
                listOfWVs.append(model[word])
        if len(listOfWVs) == 0:
            return np.zeros([len(model[list(model.wv.vocab.keys())[0]])])
    return np.mean(listOfWVs, axis=0)


def document_embedding_avg(document_Embedding, model):
    document_Embedding_avg = {}
    for file, doc in document_Embedding.items():
        document_Embedding_avg[file] = embedding_avg(model, doc)
    return document_Embedding_avg


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


def train_Word2Vec(
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


# This function reduces the dimension using umap and plot
def UMAP_plot(data_X, y, title, nn, filename, umet, rasterize=False):
    np.random.seed(42)
    # TODO: make low_memory a tool argument
    ump = umap.UMAP(
        a=None,
        angular_rp_forest=False,
        b=None,
        force_approximation_algorithm=False,
        init="spectral",
        learning_rate=1.0,
        local_connectivity=1.0,
        low_memory=False,
        metric=umet,
        metric_kwds=None,
        min_dist=0.1,
        n_components=2,
        n_epochs=1000,
        n_neighbors=nn,
        negative_sample_rate=5,
        output_metric=umet,
        output_metric_kwds=None,
        random_state=42,
        repulsion_strength=1.0,
        set_op_mix_ratio=1.0,
        spread=1.0,
        target_metric="categorical",
        target_metric_kwds=None,
        target_n_neighbors=-1,
        target_weight=0.5,
        transform_queue_size=4.0,
        transform_seed=42,
        unique=False,
        verbose=False,
    )
    _LOGGER.info(f"-- Fitting UMAP data --")
    # For UMAP,`pip install --user --upgrade pynndescent` for large data
    ump.fit(data_X)
    ump_data = pd.DataFrame(ump.transform(data_X))
    ump_data = pd.DataFrame({"UMAP 1": ump_data[0], "UMAP 2": ump_data[1], title: y})
    ump_data.to_csv(filename, index=False)
    _LOGGER.info(f"-- Saved UMAP data as {filename} --")
    fig, ax = plt.subplots(figsize=(8, 6.4))
    plt.rc("font", size=11)
    plate = sns.color_palette("husl", n_colors=len(set(y)))
    sns.scatterplot(
        x="UMAP 1",
        y="UMAP 2",
        hue=title,
        s=10,
        ax=ax,
        palette=plate,
        sizes=(10, 40),
        data=ump_data,  # .sort_values(by = title),
        rasterized=rasterize,
    )
    # TODO: only label a subset of the samples...
    # bbox_to_anchor(xpos, ypos)
    # See: https://stackoverflow.com/questions/30413789/matplotlib-automatic-legend-outside-plot
    lgd = plt.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=11,
        markerscale=2,
        edgecolor="black",
    )
    plt.gcf().canvas.draw()
    # Then define the transformation to go from pixel coordinates to Figure coordinates:
    invFigure = plt.gcf().transFigure.inverted()
    # Next, get the legend extents in pixels and convert to Figure coordinates.
    # Pull out the farthest extent in the x direction since that is the canvas
    # direction we need to adjust:
    lgd_pos = lgd.get_window_extent()
    lgd_coord = invFigure.transform(lgd_pos)
    lgd_xmax = lgd_coord[1, 0]
    # Do the same for the Axes:
    ax_pos = plt.gca().get_window_extent()
    ax_coord = invFigure.transform(ax_pos)
    ax_xmax = ax_coord[1, 0]
    # Finally, adjust the Figure canvas using tight_layout for the proportion
    # of the Axes that must move over to allow room for the legend to fit
    # within the canvas:
    shift = 1 - (lgd_xmax - ax_xmax)
    plt.gcf().tight_layout(rect=(0, 0, shift, 1))
    # plt.legend(bbox_to_anchor=(1.1,1), loc="upper right", fontsize =  11,
    #           markerscale=2, edgecolor = 'black')
    return fig


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
