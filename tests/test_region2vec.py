import os
from typing import List

import pytest
import numpy as np

from gitk.io.io import RegionSet, Region
from gitk.region2vec.main import Region2Vec, Region2VecExModel
from gitk.region2vec.utils import wordify_region, wordify_regions
from gitk.region2vec.pooling import mean_pooling, max_pooling
from gitk.tokenization.main import InMemTokenizer


@pytest.fixture
def bed_file():
    return "tests/data/to_tokenize.bed"


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

    assert model.trained == True

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
                region_word = wordify_region(region)
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
