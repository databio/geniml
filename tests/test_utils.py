import os
import sys
import pytest
import scanpy as sc
import logging
from itertools import chain

# add parent directory to path
sys.path.append("../")

from gitk import utils

# set to DEBUG to see more info
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def pbmc_data():
    return sc.read_h5ad("tests/data/pbmc.h5ad")


@pytest.fixture
def pbmc_data_backed():
    return sc.read_h5ad("tests/data/pbmc.h5ad", backed="r")


def test_anndata_chunker(pbmc_data_backed: sc.AnnData):
    chunker = utils.AnnDataChunker(pbmc_data_backed, chunk_size=10)
    assert len(chunker) == len(pbmc_data_backed) // 10 + 1
    for chunk in chunker:
        assert chunk.shape[0] > 0
