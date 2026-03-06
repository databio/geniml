# Submodules with heavy optional dependencies (torch, scanpy, gensim)
# are NOT imported here. Import them directly:
#   from geniml.region2vec import Region2VecExModel
#   from geniml.scembed import ScEmbed
# Optional dependency groups:
#   pip install geniml[ml]       # torch, transformers, gensim — embeddings & ML models
#   pip install geniml[sc]       # scanpy, anndata — single-cell data processing
#   pip install geniml[search]   # qdrant-client, fastembed — vector search
#   pip install geniml[all]      # everything (ml + sc + search)

from logging import getLogger

from ._version import __version__
from .const import PKG_NAME

_LOGGER = getLogger(PKG_NAME)
