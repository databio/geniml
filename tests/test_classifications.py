import os

import pytest
import scanpy as sc
import torch

from huggingface_hub.utils._errors import RepositoryNotFoundError

from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.tokenization.main import ITTokenizer
from geniml.classification.main import SingleCellTypeClassifier, SingleCellTypeClassifierExModel


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


def test_singlecell_classifier_forward():
    """
    Test the forward pass of a SingleCellTypeClassifier.
    """
    region2vec = Region2Vec(1000, 100)  # 1000 vocab size, 100 embedding size
    classifier = SingleCellTypeClassifier(region2vec, 10)

    # create some fake regions, simulate a batch
    ids = torch.tensor([42, 111, 432, 123, 543])
    ids = ids.unsqueeze(0)  # add a batch dimension

    # run the forward pass
    probs = classifier(ids)

    # check the output
    assert probs.shape == (1, 10)

def test_train_singlecell_classifier():
    data = 