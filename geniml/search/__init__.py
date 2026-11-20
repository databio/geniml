"""Vector search over region set embeddings.

Names are imported lazily (on first attribute access), so importing
``geniml.search`` or ``geniml.search.backends`` does not load torch or fastembed.
Only the classes that need a model (``BED2Vec``, ``Text2Vec``) load them, when they
are constructed.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Never runs. It names the lazy members for type checkers and static
    # docs tools (griffe), which cannot see through __getattr__. Import from
    # the defining modules: the sub-packages are lazy too.
    from .backends.dbbackend import QdrantBackend
    from .backends.filebackend import HNSWBackend
    from .filebackend_tools import merge_backends
    from .interfaces.bed2bed import BED2BEDSearchInterface
    from .interfaces.text2bed import Text2BEDSearchInterface
    from .query2vec.bed2vec import BED2Vec
    from .query2vec.text2vec import Text2Vec
    from .search_eval import anecdotal_search_from_hf_data
    from .utils import rand_eval

_LAZY = {
    "HNSWBackend": ".backends",
    "QdrantBackend": ".backends",
    "merge_backends": ".filebackend_tools",
    "BED2BEDSearchInterface": ".interfaces",
    "Text2BEDSearchInterface": ".interfaces",
    "BED2Vec": ".query2vec",
    "Text2Vec": ".query2vec",
    "anecdotal_search_from_hf_data": ".search_eval",
    "rand_eval": ".utils",
}

# A literal list (not list(_LAZY)) so static tools can read it.
__all__ = [
    "HNSWBackend",
    "QdrantBackend",
    "merge_backends",
    "BED2BEDSearchInterface",
    "Text2BEDSearchInterface",
    "BED2Vec",
    "Text2Vec",
    "anecdotal_search_from_hf_data",
    "rand_eval",
]


def __getattr__(name):
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
