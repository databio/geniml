import os
import pytest

from geniml.atacformer.utils import AtacformerMLMDataset
from geniml.tokenization.main import ITTokenizer


@pytest.fixture
def universe_file():
    return "tests/data/universe.bed"


@pytest.fixture
def data():
    return "tests/data/gtok_sample/"


def test_atacformer_dataset():
    path_to_data = "tests/data/gtok_sample/"
    dataset = AtacformerMLMDataset(path_to_data, 999)

    assert dataset is not None
    assert all([isinstance(x, tuple) for x in dataset])
