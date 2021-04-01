from ._version import __version__
from .const import *
from .scembed import *

import logmuse

logmuse.init_logger(PKG_NAME)

__all__ = [
    "embedding_avg",
    "document_embedding_avg",
    "shuffling",
    "train_Word2Vec",
    "label_preprocessing",
    "UMAP_plot",
    "save_dict",
    "load_dict",
    "build_dict",
    "replace_keys",
    "replace_values",
    "load_data",
]
