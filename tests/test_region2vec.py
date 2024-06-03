import os

import numpy as np
import pytest
import torch


from geniml.io.io import Region, RegionSet
from geniml.tokenization.main import TreeTokenizer
from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.region2vec.utils import Region2VecDataset


@pytest.fixture
def universe_file():
    return "tests/data/universe.uniq.bed"


def test_init_region2vec():
    model = Region2Vec(
        vocab_size=10000,
        embedding_dim=100,
    )
    assert model is not None


def test_make_region2vec_dataset():
    path_to_data = "tests/data/gtok_sample/"
    dataset = Region2VecDataset(path_to_data)

    first = next(iter(dataset))
    assert all([isinstance(x, int) for x in first])


@pytest.mark.skip(reason="Model is too big to download in the runner, takes too long.")
def test_pretrained_model():
    model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38-v2")

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
    assert (
        len(model.tokenizer) == 2_385
    )  # 2378 + 7 special tokens (unk, pad, mask, sep, cls, eos, bos)


@pytest.mark.skip(reason="Downloading the model takes too long.")
def test_r2v_pytorch_tokenizer_is_on_hf():
    model = Region2VecExModel(tokenizer="databio/r2v-ChIP-atlas-hg38-v2")
    assert model is not None
    assert len(model.tokenizer) == 1_698_713


def test_r2v_pytorch_exmodel_train(universe_file: str):
    model = Region2VecExModel(tokenizer=TreeTokenizer(universe_file))
    assert model is not None

    dataset = Region2VecDataset("tests/data/gtok_sample/", convert_to_str=True)

    loss = model.train(dataset, epochs=10, min_count=1)
    assert loss


def test_r2v_pytorch_encode(universe_file: str):
    model = Region2VecExModel(tokenizer=TreeTokenizer(universe_file))
    assert model is not None

    r = Region("chr1", 63403166, 63403785)
    embedding = model.encode(r)
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 100)

    rs = RegionSet("tests/data/to_tokenize.bed")
    embedding = model.encode(rs)
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (13, 100)


def test_save_load_pytorch_exmodel(universe_file: str):
    model = Region2VecExModel(tokenizer=TreeTokenizer(universe_file))
    assert model is not None

    dataset = Region2VecDataset("tests/data/gtok_sample/", convert_to_str=True)
    loss = model.train(dataset, epochs=10, min_count=1)

    before_embedding = model.encode(Region("chr1", 63403166, 63403785))
    assert loss
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
            # else wrong occurred up the stack
            print(e)
            pass
