"""Vector search over region set embeddings.

Names are imported lazily (on first attribute access), so importing
``geniml.search`` or ``geniml.search.backends`` does not load torch or fastembed.
Only the classes that need a model (``BED2Vec``, ``Text2Vec``) load them, when they
are constructed.
"""

from importlib import import_module

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

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
