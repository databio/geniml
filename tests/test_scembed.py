import pytest
import scanpy as sc
import logging
from itertools import chain
from gitk import scembed

# set to DEBUG to see more info
logging.basicConfig(level=logging.INFO)


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
    # convert pbmc_data to df and drop any columns (regions with all 0 signal)
    pbmc_df = pbmc_data.to_df()
    pbmc_df_dropped = pbmc_df.loc[:, (pbmc_df != 0).any(axis=0)]

    # convert to docs
    docs = scembed.convert_anndata_to_documents(pbmc_data)

    # ensure all cells converted
    assert len(docs) == pbmc_data.shape[0]

    # ensure all regions represented
    all_regions = set(list(chain(*docs)))
    assert len(all_regions) == pbmc_df_dropped.shape[1]

    # ensure all regions are strings and contain no spaces
    for doc in docs:
        assert all([isinstance(r, str) for r in doc])
        assert all([" " not in r for r in doc])


def test_document_shuffle():
    docs = [["a", "b", "c"], ["d", "e", "f"]]
    shuffled = scembed.shuffle_documents(docs, 10)
    assert len(shuffled) == len(docs)
    for doc in shuffled:
        assert len(doc) == len(docs[0])
        # by pure random chance, the following COULD fail, so we'll just comment it out
        # assert doc != docs[0]
        # assert doc != docs[1]


def test_model_creation(pbmc_data: sc.AnnData):
    model = scembed.SCEmbed(pbmc_data)
    assert model


def test_model_training(pbmc_data: sc.AnnData):
    # remove gensim logging
    logging.getLogger("gensim").setLevel(logging.ERROR)
    model = scembed.SCEmbed(pbmc_data)
    model.train(epochs=3)
    assert model.trained
    assert isinstance(model.region2vec, dict)
