from six.moves import cPickle as pickle #for performance
from gensim.models import Word2Vec
from numba import config, threading_layer
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# Ensure pyqt5 is upgraded (pip install --user --upgrade pyqt5
#import csv
import numpy as np
import pandas as pd
import os
from logging import getLogger
import sys
import umap
import vaex

from .const import *

_LOGGER = getLogger(PKG_NAME)

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'


def embedding_avg(model, document):
    _LOGGER.debug(f"embedding_avg model: {model}") 
    listOfWVs= []
    if type(document) is list:
        _LOGGER.debug(f"embedding_avg document is a list")
        for word in document:
            if word in model.wv.vocab:
                listOfWVs.append(model[word])
        if(len(listOfWVs) == 0):
            return np.zeros([len(model[list(model.wv.vocab.keys())[0]])])
    else:
        _LOGGER.debug(f"embedding_avg document is NOT a list")
        for word in document.split(' '):
            if word in model.wv.vocab:
                listOfWVs.append(model[word])
        if(len(listOfWVs) == 0):
            return np.zeros([len(model[list(model.wv.vocab.keys())[0]])])
    return np.mean(listOfWVs, axis=0)


def document_embedding_avg(document_Embedding, model):
    document_Embedding_avg = {}
    for file, doc  in document_Embedding.items():
        document_Embedding_avg[file] = embedding_avg(model, doc)
    return document_Embedding_avg


# shuffle the document to generate data for word2vec
def shuffling(documents, shuffle_repeat):
    _LOGGER.debug(f"Shuffling documents {shuffle_repeat} times.")
    common_text = list(documents.values())
    training_samples = []
    training_samples.extend(common_text)
    for rn in range(shuffle_repeat):
        [(np.random.shuffle(l)) for l in common_text]
        training_samples.extend(common_text)
    return training_samples


def train_Word2Vec(documents, window_size = 100,
                   dim = 128, min_count = 10, nothreads = 1):
    """
    Train word2vec algorithm
    
    dim == size (set to a multiple of 4 ideally). In general, more data
           means that you can go for a bigger size.
    """
    # sg=0 Training algorithm: 1 for skip-gram; otherwise CBOW.
    message = (
        f"Training Word2Vec embeddings with {window_size} window size, "
        f"at {dim} dimensions, with a minimum 'word' count of "
        f"{min_count} and using {nothreads} threads."
    )
    _LOGGER.debug(message)
    model = Word2Vec(sentences=documents, window=window_size,
                     size=dim, min_count=min_count,
                     workers=nothreads)
    return model


# preprocess the labels
def label_preprocessing(y, delim):
    y_cell = []
    for y1 in y:
        y_cell.append(y1.split(delim)[0])
    return y_cell


# This function reduces the dimension using umap and plot 
def UMAP_plot(data_X, y, title, nn, filename, umet,
              rasterize=False):
    np.random.seed(42)
    # TODO: make low_memory a tool argument
    ump = umap.UMAP(a=None, angular_rp_forest=False, b=None,
                    force_approximation_algorithm=False, init='spectral',
                    learning_rate=1.0, local_connectivity=1.0,
                    low_memory=False, metric=umet, metric_kwds=None,
                    min_dist=0.1, n_components=2, n_epochs=1000,
                    n_neighbors=nn, negative_sample_rate=5,
                    output_metric=umet, output_metric_kwds=None,
                    random_state=42, repulsion_strength=1.0,
                    set_op_mix_ratio=1.0, spread=1.0,
                    target_metric='categorical', target_metric_kwds=None,
                    target_n_neighbors=-1, target_weight=0.5,
                    transform_queue_size=4.0, transform_seed=42,
                    unique=False, verbose=False)
    _LOGGER.info(f'-- Fitting UMAP data --')
    # For UMAP,`pip install --user --upgrade pynndescent` for large data
    ump.fit(data_X)
    ump_data = pd.DataFrame(ump.transform(data_X))
    ump_data = pd.DataFrame({'UMAP 1':ump_data[0],
                            'UMAP 2':ump_data[1],
                            title:y})
    ump_data.to_csv(filename, index=False)
    _LOGGER.info(f'-- Saved UMAP data as {filename} --')
    fig, ax = plt.subplots(figsize=(8,6.4))
    plt.rc('font', size=11)
    plate =(sns.color_palette("husl", n_colors=len(set(y))))
    sns.scatterplot(x="UMAP 1", y="UMAP 2", hue=title, s= 10,ax= ax,
                    palette = plate, sizes=(10, 40),
                    data=ump_data, #.sort_values(by = title),
                    rasterized=rasterize)
    # TODO: only label a subset of the samples...
    # bbox_to_anchor(xpos, ypos)
    # See: https://stackoverflow.com/questions/30413789/matplotlib-automatic-legend-outside-plot
    lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize =  11,
                     markerscale=2, edgecolor = 'black')
    plt.gcf().canvas.draw()
    #Then define the transformation to go from pixel coordinates to Figure coordinates:
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
    #plt.legend(bbox_to_anchor=(1.1,1), loc="upper right", fontsize =  11,
    #           markerscale=2, edgecolor = 'black')
    return fig


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        try:
            ret_di = pickle.load(f)
            return ret_di
        except EOFError:
            _LOGGER.error("Size (In bytes) of '%s':" %filename_, os.path.getsize(filename_))


def build_dict(mtx, SIZE=100_000):
    documents = {}
    for i1, i2, chunk in mtx.evaluate_iterator(mtx[:,0], chunk_size=SIZE):
        for x in chunk:
            row, col, entry = x.as_py().split()
            _LOGGER.debug(f"{row}, {col}, {entry}")
            if col not in documents:
                documents[str(col)] = []
            val = sys.intern(row)
            documents[col].append(val)
    return documents


def replace_keys(a_dict, new_keys):
    for key in list(a_dict.keys()):
        try:
            new_key = new_keys[int(key)-1][0]
            a_dict[new_key] = a_dict.pop(key)
        except (KeyError, AssertionError) as err:
            _LOGGER.error(f"err: {err}")
            pass


def replace_values(a_dict, new_values):
    for key in list(a_dict.keys()):
        try:
            int_list = list(map(int, a_dict[key]))
            a_dict[key] = [sys.intern(new_values[i-1]) for i in int_list]
        except:
            e = sys.exc_info()[0]
            _LOGGER.error(f"err: {err}")
            pass


def load_data(filename, header=None, SIZE=5_000_000):
    _LOGGER.debug(f"Loading {filename}")
    if os.path.exists(filename + ".hdf5"):
        data = vaex.open(filename + ".hdf5")
    else:
        # initialize
        data = vaex.from_csv(filename, sep="\t", convert=True,
                             chunk_size=SIZE, copy_index=False,
                             header=header)
    return data
