import os
from typing import List

import pytest
import numpy as np
import torch


from geniml.io.io import Region, RegionSet
from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.region2vec.pooling import max_pooling, mean_pooling
from geniml.tokenization.main import InMemTokenizer, ITTokenizer
from geniml.utils import wordify_region, wordify_regions
from geniml.region2vec.experimental import (
    Region2Vec as Region2VecV2,
    Region2VecExModel as Region2VecExModelV2,
)


@pytest.fixture
def bed_file():
    return "tests/data/to_tokenize.bed"


@pytest.fixture
def bed_file2():
    return "tests/data/to_tokenize2.bed"


@pytest.fixture
def universe_file():
    return "tests/data/universe.bed"


@pytest.fixture
def regions(bed_file: str):
    return RegionSet(bed_file)


@pytest.fixture
def region_sets(regions: RegionSet):
    # split regions into 5 region sets
    sub_size = len(regions) // 5
    region_sets = [RegionSet(regions[i * sub_size : (i + 1) * sub_size]) for i in range(5)]
    return region_sets


def test_init_region2vec():
    model = Region2Vec()
    assert model is not None


def test_wordify_regions(regions: RegionSet):
    region_words = wordify_regions(regions)
    assert region_words is not None
    assert all([isinstance(r, str) for r in region_words])
    assert all([len(r.split("_")) == 3 for r in region_words])

    region_word = wordify_region(regions[0])
    assert region_word is not None
    assert isinstance(region_word, str)
    assert len(region_word.split("_")) == 3


def test_train_region2vec(region_sets: List[RegionSet]):
    model = Region2Vec(
        min_count=1,  # for testing, we need to set min_count to 1
    )

    model.train(region_sets, epochs=100)

    assert model.trained is True

    # make sure all regions in region_sets are in the vocabulary
    for region_set in region_sets:
        for region in region_set:
            region_word = wordify_region(region)
            assert region_word in model.wv
            assert isinstance(model(region), np.ndarray)
            assert isinstance(model.forward(region), np.ndarray)


def test_train_from_bed_files(bed_file: str):
    region_sets = [RegionSet(bed_file) for _ in range(10)]
    model = Region2Vec(
        min_count=1,  # for testing, we need to set min_count to 1
    )
    model.train(region_sets, epochs=5)

    regions = RegionSet(bed_file)
    for region in regions:
        region_word = wordify_region(region)
        assert region_word in model.wv
        assert isinstance(model(region), np.ndarray)
        assert isinstance(model.forward(region), np.ndarray)


def test_save_and_load_model(region_sets: RegionSet):
    model = Region2Vec(
        min_count=1,  # for testing, we need to set min_count to 1
    )
    model.train(region_sets, epochs=100)

    try:
        model.save("tests/data/test_model.model")
        assert os.path.exists("tests/data/test_model.model")
        # load in
        model_loaded = Region2Vec.load("tests/data/test_model.model")
        assert model_loaded is not None
        for region_set in region_sets:
            for region in region_set:
                region_word = wordify_region(region)
                assert region_word in model_loaded.wv
                assert isinstance(model_loaded(region), np.ndarray)
                assert isinstance(model_loaded.forward(region), np.ndarray)
    finally:
        os.remove("tests/data/test_model.model")


def test_train_exmodel(region_sets: List[RegionSet], universe_file: str):
    model = Region2VecExModel(
        min_count=1,  # for testing, we need to set min_count to 1
        tokenizer=InMemTokenizer(universe_file),
    )
    # or
    # model = Region2VecExModel(min_count=1)
    # model.add_tokenizer_from_universe(universe_file)
    model.train(region_sets, epochs=100)

    try:
        model.export("tests/data/model-r2v-test/")
        assert os.path.exists("tests/data/model-r2v-test/model.bin")
        assert os.path.exists("tests/data/model-r2v-test/universe.bed")

        # load in
        model_loaded = Region2VecExModel()
        model_loaded.from_pretrained(
            "tests/data/model-r2v-test/model.bin",
            "tests/data/model-r2v-test/universe.bed",
        )
        assert model_loaded is not None
        for region_set in region_sets:
            for region in region_set:
                _region_word = wordify_region(region)
                assert len(model_loaded.wv) > 0

    finally:
        os.remove("tests/data/model-r2v-test/model.bin")
        os.remove("tests/data/model-r2v-test/universe.bed")
        os.rmdir("tests/data/model-r2v-test/")


# @pytest.mark.skip(reason="Model is too big to download in the runner, takes too long.")
def test_pretrained_model():
    model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38")

    region = Region("chr1", 63403166, 63403785)
    embedding = model.encode(region)

    assert embedding is not None
    assert isinstance(embedding, np.ndarray)


def test_mean_pooling():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    assert np.allclose(mean_pooling([a, b]), np.array([2.5, 3.5, 4.5]))
    assert np.allclose(mean_pooling(np.array([a, b])), np.array([2.5, 3.5, 4.5]))
    assert mean_pooling([a, b]).shape == (3,)
    assert mean_pooling(np.array([a, b])).shape == (3,)


def test_max_pooling():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    assert np.allclose(max_pooling([a, b]), np.array([4, 5, 6]))
    assert np.allclose(max_pooling(np.array([a, b])), np.array([4, 5, 6]))
    assert max_pooling([a, b]).shape == (3,)
    assert max_pooling(np.array([a, b])).shape == (3,)


def test_model_pooling():
    r1 = Region("chr11", 45639005, 45639830)
    r2 = Region("chr1", 89566099, 89566939)  # will be None
    r3 = Region("chr11", 63533954, 63534897)

    model = Region2VecExModel()
    model.from_pretrained("tests/data/tiny-model/model.bin", "tests/data/tiny-model/universe.bed")

    r1_vector = model.encode(r1)
    r2_vector = model.encode(r2)
    r3_vector = model.encode(r3)

    assert all([isinstance(v, np.ndarray) for v in [r1_vector, r3_vector]])
    assert r2_vector is None
    assert r1_vector.shape == (100,)
    assert r3_vector.shape == (100,)

    vectors = model.encode([r1, r2, r3])
    assert isinstance(
        vectors, list
    )  # should return a list of vectors, not an np.ndarray. List of np.ndarray is fine and also more conducive to downstream processing. It also mirrors the input.
    assert len(vectors) == 3
    assert vectors[0].shape == (100,)
    assert vectors[1] is None
    assert vectors[2].shape == (100,)

    vector_mean = model.encode([r1, r2, r3], pool="mean")
    vector_max = model.encode([r1, r2, r3], pool="max")
    assert vector_mean.shape == (100,)
    assert vector_max.shape == (100,)

    # custom pooling function that just sums them
    def sum_pooling(vectors):
        vectors = [v for v in vectors if v is not None]
        return np.sum(vectors, axis=0)

    vector_sum = model.encode([r1, r2, r3], pool=sum_pooling)
    assert vector_sum.shape == (100,)


def test_r2v_pytorch_forward():
    vocab_size = 10000
    embedding_dim = 100

    model = Region2VecV2(vocab_size, embedding_dim)
    assert model is not None

    # create a random tensor with 10 tokens
    x = torch.randint(low=0, high=100, size=(10,))
    y = model.forward(x)
    assert y.shape == (10, 100)


def test_r2v_pytorch_exmodel_train(universe_file: str):
    model = Region2VecExModelV2(tokenizer=ITTokenizer(universe_file))
    assert model is not None

    rs1 = list(RegionSet("tests/data/to_tokenize.bed"))
    rs2 = list(RegionSet("tests/data/to_tokenize2.bed"))
    rs3 = rs1[0:10] + rs2[0:10]

    loss = model.train([rs1, rs2, rs3], epochs=10)
    assert loss[0] > loss[-1]


def test_r2v_pytorch_encode(universe_file: str):
    model = Region2VecExModelV2(tokenizer=ITTokenizer(universe_file))
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
    model = Region2VecExModelV2(tokenizer=ITTokenizer(universe_file))
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
        model_loaded = Region2VecExModelV2.from_pretrained("tests/data/test_model")

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
