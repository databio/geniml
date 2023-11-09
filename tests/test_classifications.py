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


def test_init_classifier_from_pretrained():
    # get pre-trained r2v
    r2v = Region2VecExModel("databio/r2v-ChIP-atlas-hg38-v2")

    # create classifier
    classifier = SingleCellTypeClassifier(r2v.model, 10)

    assert classifier.region2vec.embedding_dim == 100
    assert classifier.num_classes == 10
    assert classifier.region2vec == r2v.model


def test_init_exmodel_from_scratch():
    classifier = SingleCellTypeClassifierExModel(
        tokenizer=ITTokenizer("tests/data/universe.bed"),
        num_classes=10,
    )

    assert classifier.region2vec.embedding_dim == 100
    assert classifier.num_classes == 10
    assert classifier.region2vec.vocab_size == 2380


def test_init_exmodel_from_scratch_r2v():
    r2v = Region2Vec(2380, 100)

    classifier = SingleCellTypeClassifierExModel(
        tokenizer=ITTokenizer("tests/data/universe.bed"),
        region2vec=r2v,
        num_classes=10,
    )

    assert classifier.region2vec.embedding_dim == 100
    assert classifier.num_classes == 10
    assert classifier.region2vec.vocab_size == 2380
