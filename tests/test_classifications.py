import os

import pytest
import numpy as np
import torch

from huggingface_hub.utils._errors import RepositoryNotFoundError

from geniml.io.io import Region, RegionSet
from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.tokenization.main import ITTokenizer
from geniml.classification.main import SingleCellTypeClassifier


def test_init_singlecell_classifier_from_scratch():
    """
    Test the initialization of a SingleCellTypeClassifier from scratch.
    """
    region2vec = Region2Vec(1000, 100)  # 1000 vocab size, 100 embedding size
    classifier = SingleCellTypeClassifier(region2vec, 10)  # 10 classes

    assert classifier.embedding_dim == 100
    assert classifier.num_classes == 10
    assert classifier.region2vec == region2vec


def test_init_singlecell_classifier_should_fail():
    """
    Test the initialization of a SingleCellTypeClassifier from scratch.
    """
    with pytest.raises(RepositoryNotFoundError):
        # its going to try and download the model from huggingface, but it doesn't exist
        _ = SingleCellTypeClassifier("test", 10)  # 10 classes

    with pytest.raises(ValueError):
        # its going to be mad that the region2vec is not a string or a Region2Vec instance
        _ = SingleCellTypeClassifier(10, 10)


def test_init_singlecell_classifier_from_fresh_r2v():
    """
    Test the initialization of a SingleCellTypeClassifier from huggingface.
    """
    classifier = SingleCellTypeClassifier(
        region2vec="databio/r2v-ChIP-atlas-hg38-v2", num_classes=10
    )

    assert classifier.embedding_dim == 100
    assert classifier.num_classes == 10


def test_init_singlecell_classifier_from_region2vecexmodel():
    """
    Test the initialization of a SingleCellTypeClassifier from huggingface, through
    a Region2VecExModel.
    """
    model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38-v2")
    classifier = SingleCellTypeClassifier(model.model, num_classes=10)

    assert classifier.embedding_dim == 100
    assert classifier.num_classes == 10
