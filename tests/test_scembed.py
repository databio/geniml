import logging
import os
import sys

import pytest
import scanpy as sc

from geniml.tokenization.main import ITTokenizer
from geniml.scembed.main import ScEmbed

# add parent directory to path
sys.path.append("../")


# set to DEBUG to see more info
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def universe_file():
    return "tests/data/universe.bed"


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


def test_model_training(universe_file: str, pbmc_data: sc.AnnData):
    # remove gensim logging
    logging.getLogger("gensim").setLevel(logging.ERROR)
    model = ScEmbed(tokenizer=ITTokenizer(universe_file))  # set to 1 for testing
    model.train(pbmc_data, epochs=3)

    # keep only columns with values > 0
    pbmc_data = pbmc_data[:, pbmc_data.X.sum(axis=0) > 0]

    assert model.trained


def test_model_train_and_export(universe_file: str, pbmc_data: sc.AnnData):
    # remove gensim logging
    logging.getLogger("gensim").setLevel(logging.ERROR)
    model = ScEmbed(tokenizer=ITTokenizer(universe_file))  # set to 1 for testing
    model.train(pbmc_data, epochs=3)

    assert model.trained

    # save
    try:
        model.export("tests/data/model-tests")
        model = ScEmbed.from_pretrained("tests/data/model-tests")

        # ensure model is still trained and has region2vec
        assert model.trained

    finally:
        os.remove("tests/data/model-tests/checkpoint.pt")
        os.remove("tests/data/model-tests/universe.bed")
        os.remove("tests/data/model-tests/config.yaml")


@pytest.mark.skip(reason="Need to get a pretrained model first")
def test_pretrained_scembed_model(hf_model: str, pbmc_data: sc.AnnData):
    model = ScEmbed(hf_model)
    embeddings = model.encode(pbmc_data)
    assert embeddings.shape[0] == pbmc_data.shape[0]
