import logging
import os
import sys

import pytest
import scanpy as sc

# add parent directory to path
sys.path.append("../")

from geniml.io.io import Region
from geniml.region2vec.utils import wordify_region, wordify_regions
from geniml.scembed.main import ScEmbed

# set to DEBUG to see more info
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def hf_model():
    return "databio/r2v-pbmc-hg38-small"


@pytest.fixture
def pbmc_data():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad")


@pytest.fixture
def pbmc_data_backed():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad", backed="r")


def test_model_creation():
    model = ScEmbed()
    assert model


def test_model_training(pbmc_data: sc.AnnData):
    # remove gensim logging
    logging.getLogger("gensim").setLevel(logging.ERROR)
    model = ScEmbed(min_count=1)  # set to 1 for testing
    model.train(pbmc_data, epochs=3)

    # keep only columns with values > 0
    pbmc_data = pbmc_data[:, pbmc_data.X.sum(axis=0) > 0]

    chrs = pbmc_data.var["chr"].values.tolist()
    starts = pbmc_data.var["start"].values.tolist()
    ends = pbmc_data.var["end"].values.tolist()

    # list of regions
    regions = [Region(c, int(s), int(e)) for c, s, e in zip(chrs, starts, ends)]
    region_words = wordify_regions(regions)

    #

    assert model.trained
    assert all([word in model.wv for word in region_words])


def test_model_train_and_export(pbmc_data: sc.AnnData):
    # remove gensim logging
    logging.getLogger("gensim").setLevel(logging.ERROR)
    model = ScEmbed(min_count=1)  # set to 1 for testing
    model.train(pbmc_data, epochs=3)
    assert model.trained

    # save
    try:
        model.export("tests/data/model-tests")
        model = ScEmbed()
        model.from_pretrained(
            "tests/data/model-tests/model.bin",
            "tests/data/model-tests/universe.bed",
        )

        # ensure model is still trained and has region2vec
        assert model.trained
        assert len(model.wv) > 0

    finally:
        os.remove("tests/data/model-tests/model.bin")
        os.remove("tests/data/model-tests/universe.bed")


# @pytest.mark.skip(reason="Need to get a pretrained model first")
def test_pretrained_scembed_model(hf_model: str, pbmc_data: sc.AnnData):
    model = ScEmbed(hf_model)
    embeddings = model.encode(pbmc_data)
    assert embeddings.shape[0] == pbmc_data.shape[0]
