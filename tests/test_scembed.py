import pytest
import scanpy as sc
from gitk import scembed


@pytest.fixture
def pbmc_data():
    return sc.read_h5ad("tests/data/pbmc.h5ad")


def test_import():
    assert scembed


def test_load_scanpy():
    scembed.load_scanpy_data("tests/data/pbmc.h5ad")


def test_extract_region_list(pbmc_data: sc.AnnData):
    regions = scembed.extract_region_list(pbmc_data.var)
    assert len(regions) == pbmc_data.shape[1]
    for region in regions:
        assert isinstance(region, str)


def test_remove_zero_regions(pbmc_data: sc.AnnData):
    cell_dict = pbmc_data.X[0].toarray()[0].tolist()
    cell_dict = dict(zip(pbmc_data.var.index.tolist(), cell_dict))
    cell_dict = scembed.remove_zero_regions(cell_dict)
    assert len(cell_dict) < pbmc_data.shape[1]
    for k, v in cell_dict.items():
        assert isinstance(k, str)
        assert isinstance(v, (int, float))
        assert v > 0


def test_document_creation(pbmc_data: sc.AnnData):
    docs = scembed.convert_anndata_to_documents(pbmc_data)
    assert len(docs) == pbmc_data.shape[0]
    for doc in docs:
        assert all([isinstance(r, str) for r in doc])


def test_document_shuffle():
    docs = [["a", "b", "c"], ["d", "e", "f"]]
    shuffled = scembed.shuffle_documents(docs, 10)
    assert len(shuffled) == len(docs)
    for doc in shuffled:
        assert len(doc) == len(docs[0])
        # by pure random chance, the following COULD fail, so we'll just comment it out
        # assert doc != docs[0]
        # assert doc != docs[1]
