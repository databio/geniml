import ast
from pathlib import Path

import geniml.search as search

INIT = Path(search.__file__)


def test_all_matches_lazy_table():
    # __all__ is a literal so docs tools can read it; keep it in sync.
    assert sorted(search.__all__) == sorted(search._LAZY)


def test_type_checking_imports_match_lazy_table():
    # The TYPE_CHECKING block names the lazy members for static tools.
    tree = ast.parse(INIT.read_text())
    block = next(
        n
        for n in tree.body
        if isinstance(n, ast.If) and getattr(n.test, "id", "") == "TYPE_CHECKING"
    )
    imported = {
        alias.name
        for node in block.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imported == set(search._LAZY)
