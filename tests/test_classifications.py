import os
import pytest

import numpy as np

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


@pytest.mark.parametrize(
    "r2v, tokenizer",
    [
        (None, ITTokenizer("tests/data/universe.bed")),
        (None, "tests/data/universe.bed"),
        (None, "databio/r2v-ChIP-atlas-hg38-v2"),
        (Region2Vec(2380, 100), ITTokenizer("tests/data/universe.bed")),
        (Region2Vec(2380, 100), "tests/data/universe.bed"),
        (Region2Vec(1_698_713, 100), "databio/r2v-ChIP-atlas-hg38-v2"),
        ("databio/r2v-ChIP-atlas-hg38-v2", ITTokenizer("tests/data/universe.bed")),
        ("databio/r2v-ChIP-atlas-hg38-v2", "tests/data/universe.bed"),
        ("databio/r2v-ChIP-atlas-hg38-v2", "databio/r2v-ChIP-atlas-hg38-v2"),
    ],
)
def test_init_exmodel(r2v, tokenizer):
    model = SingleCellTypeClassifierExModel(region2vec=r2v, tokenizer=tokenizer, num_classes=10)
    assert model is not None
    assert model.region2vec is not None
    assert model.tokenizer is not None
    assert model.num_classes == 10


def test_freeze_r2v():
    r2v = Region2Vec(2380, 100)
    model = SingleCellTypeClassifierExModel(
        region2vec=r2v, tokenizer="tests/data/universe.bed", num_classes=10
    )
    assert model is not None
    assert model.region2vec is not None
    assert model.tokenizer is not None
    assert model.num_classes == 10

    assert model.region2vec.projection.weight.requires_grad is True
    model.freeze_r2v()
    assert model.region2vec.projection.weight.requires_grad is False


def test_export_exmodel():
    r2v = Region2Vec(2380, 100)
    model = SingleCellTypeClassifierExModel(
        region2vec=r2v, tokenizer="tests/data/universe.bed", num_classes=10
    )
    assert model is not None

    # grab some weights from the inner r2v
    r2v_weights = model.region2vec.projection.weight.detach().clone().numpy()
    output_weights = model._model.output_layer.weight.detach().clone().numpy()

    # save
    try:
        model.export("tests/data/model-tests")
        model = SingleCellTypeClassifierExModel.from_pretrained("tests/data/model-tests")

        # ensure model is still trained and has region2vec
        assert model.trained

        # ensure weights are the same
        assert np.allclose(
            r2v_weights, model.region2vec.projection.weight.detach().clone().numpy()
        )
        assert np.allclose(
            output_weights, model._model.output_layer.weight.detach().clone().numpy()
        )
    finally:
        for file in os.listdir("tests/data/model-tests"):
            os.remove(os.path.join("tests/data/model-tests", file))
