import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from geniml.io.io import Region, RegionSet
from geniml.region2vec.main import Region2Vec
from geniml.region2vec.main import Region2VecExModel
from geniml.tokenization.main import ITTokenizer


@pytest.fixture
def universe_file():
    return "tests/data/universe.bed"


def test_init_region2vec():
    model = Region2Vec(
        vocab_size=10000,
        embedding_dim=100,
    )
    assert model is not None


@pytest.mark.skip(reason="Model is too big to download in the runner, takes too long.")
def test_pretrained_model():
    model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38")

    region = Region("chr1", 63403166, 63403785)
    embedding = model.encode(region)

    assert embedding is not None
    assert isinstance(embedding, np.ndarray)


def test_r2v_pytorch_forward():
    vocab_size = 10000
    embedding_dim = 100

    model = Region2Vec(vocab_size, embedding_dim)
    assert model is not None

    # create a random tensor with 10 tokens
    x = torch.randint(low=0, high=100, size=(10,))
    y = model.forward(x)
    assert y.shape == (10, 100)


def test_r2v_pytorch_tokenizer_is_file_on_disk(universe_file: str):
    model = Region2VecExModel(tokenizer=universe_file)
    assert model is not None
    assert len(model.tokenizer) == 2380


def test_r2v_pytorch_tokenizer_is_on_hf():
    model = Region2VecExModel(tokenizer="databio/r2v-ChIP-atlas-hg38-v2")
    assert model is not None
    assert len(model.tokenizer) == 1_698_713


def test_r2v_pytorch_exmodel_train(universe_file: str):
    model = Region2VecExModel(tokenizer=ITTokenizer(universe_file))
    assert model is not None

    rs1 = list(RegionSet("tests/data/to_tokenize.bed"))
    rs2 = list(RegionSet("tests/data/to_tokenize2.bed"))
    rs3 = rs1[0:10] + rs2[0:10]

    loss = model.train([rs1, rs2, rs3], epochs=10)
    assert loss[0] > loss[-1]


def test_r2v_pytorch_encode(universe_file: str):
    model = Region2VecExModel(tokenizer=ITTokenizer(universe_file))
    assert model is not None

    r = Region("chr1", 63403166, 63403785)
    embedding = model.encode(r)
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (100,)

    rs = RegionSet("tests/data/to_tokenize.bed")
    embedding = model.encode(list(rs))
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (100,)


def test_save_load_pytorch_exmodel(universe_file: str):
    model = Region2VecExModel(tokenizer=ITTokenizer(universe_file))
    assert model is not None

    rs1 = list(RegionSet("tests/data/to_tokenize.bed"))
    rs2 = list(RegionSet("tests/data/to_tokenize2.bed"))
    rs3 = rs1[0:10] + rs2[0:10]

    loss = model.train([rs1, rs2, rs3], epochs=10)
    before_embedding = model.encode(Region("chr1", 63403166, 63403785))
    assert loss[0] > loss[-1]
    try:
        # save the model
        model.export("tests/data/test_model/")
        assert os.path.exists("tests/data/test_model/checkpoint.pt")
        assert os.path.exists("tests/data/test_model/universe.bed")

        # load in
        model_loaded = Region2VecExModel.from_pretrained("tests/data/test_model")

        # the region embeddings should be the same
        after_embedding = model_loaded.encode(Region("chr1", 63403166, 63403785))
        assert np.allclose(before_embedding, after_embedding)

    finally:
        try:
            os.remove("tests/data/test_model/checkpoint.pt")
            os.remove("tests/data/test_model/universe.bed")
            os.remove("tests/data/test_model/config.yaml")
            os.rmdir("tests/data/test_model/")
        except Exception as e:
            # just try to remove it, if it doesn't work, then pass, means something
            # else wrong occured up the stack
            print(e)
            pass
