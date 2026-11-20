"""Lazily imported (see ``geniml.search``), so no model library loads on import."""

from importlib import import_module

_LAZY = {
    "BED2Vec": ".bed2vec",
    "Text2Vec": ".text2vec",
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
