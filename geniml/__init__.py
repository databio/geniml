# Submodules with heavy optional dependencies (torch, scanpy, gensim)
# are NOT imported here. Import them directly:
#   from geniml.region2vec import Region2VecExModel
#   from geniml.scembed import ScEmbed
# Install ML dependencies with: pip install geniml[ml]

from logging import getLogger

from ._version import __version__
from .const import PKG_NAME

_LOGGER = getLogger(PKG_NAME)
