import os
from typing import List

import pytest
import numpy as np

from gitk.io.io import RegionSet
from gitk.region2vec.main import Region2Vec, Region2VecExModel
from gitk.region2vec.utils import wordify_region, wordify_regions
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


def test_train_exmodel(region_sets: RegionSet, universe_file: str):
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
                assert region_word in model_loaded.wv
                assert isinstance(model_loaded(region), np.ndarray)
                assert isinstance(model_loaded.encode(region), np.ndarray)

    finally:
        os.remove("tests/data/model-r2v-test/model.bin")
        os.remove("tests/data/model-r2v-test/universe.bed")
        os.rmdir("tests/data/model-r2v-test/")
