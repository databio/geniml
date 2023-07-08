import logging
import sys

import pytest
import scanpy as sc

# add parent directory to path
sys.path.append("../")

from gitk import utils

# set to DEBUG to see more info
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def pbmc_data():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad")


@pytest.fixture
def pbmc_data_backed():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad", backed="r")


def test_anndata_chunker(pbmc_data_backed: sc.AnnData):
    chunker = utils.AnnDataChunker(pbmc_data_backed, chunk_size=2)
    assert len(chunker) == len(pbmc_data_backed) // 2 + 1
    for chunk in chunker:
        print(chunk)
        assert chunk.shape[0] > 0
