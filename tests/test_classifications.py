import os
from typing import Literal
import pytest

import torch
import numpy as np
import scanpy as sc

from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.tokenization.main import ITTokenizer
from geniml.classification.main import Region2VecClassifier, SingleCellTypeClassifier


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
    classifier = Region2VecClassifier(r2v.model, 10)

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
def test_init_exmodel(
    r2v: Region2Vec | Literal["databio/r2v-ChIP-atlas-hg38-v2"] | None,
    tokenizer: ITTokenizer | Literal["tests/data/universe.bed", "databio/r2v-ChIP-atlas-hg38-v2"],
):
    model = SingleCellTypeClassifier(region2vec=r2v, tokenizer=tokenizer, num_classes=10)
    assert model is not None
    assert model.region2vec is not None
    assert model.tokenizer is not None
    assert model.num_classes == 10


def test_freeze_r2v():
    r2v = Region2Vec(2380, 100)
    model = SingleCellTypeClassifier(
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
    model = SingleCellTypeClassifier(
        region2vec=r2v, tokenizer="tests/data/universe.bed", num_classes=10
    )
    assert model is not None

    # grab some weights from the inner r2v
    r2v_weights = model.region2vec.projection.weight.detach().clone().numpy()
    output_weights = model._model.output_layer.weight.detach().clone().numpy()

    # save
    try:
        model.export("tests/data/model-tests")
        model = SingleCellTypeClassifier.from_pretrained("tests/data/model-tests")

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


def test_train_ex_model():
    data = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")
    model = SingleCellTypeClassifier(
        region2vec=Region2Vec(2380, 100),
        tokenizer="tests/data/universe.bed",
        num_classes=data.obs["cell_type"].nunique(),
    )

    losses = model.train(data, label_key="cell_type", batch_size=2, epochs=3)
    assert model.trained
    assert losses[0] > losses[-1]


def test_train_ex_model_save_load():
    data = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")
    model = SingleCellTypeClassifier(
        region2vec=Region2Vec(2380, 100),
        tokenizer="tests/data/universe.bed",
        num_classes=data.obs["cell_type"].nunique(),
    )

    losses = model.train(data, label_key="cell_type", batch_size=2, epochs=3)
    assert model.trained
    assert losses[0] > losses[-1]

    input_tokens = [t.id for t in model.tokenizer.tokenize(data[0, :].to_memory())[0]]
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)

    pre_save_output = model._model(input_tokens)

    model.export("tests/data/model-tests")
    model = SingleCellTypeClassifier.from_pretrained("tests/data/model-tests")

    post_save_output = model._model(input_tokens)

    assert np.allclose(pre_save_output.detach().numpy(), post_save_output.detach().numpy())


def test_train_and_run_predictions():
    data = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")
    model = SingleCellTypeClassifier(
        region2vec=Region2Vec(2380, 100),
        tokenizer="tests/data/universe.bed",
        num_classes=data.obs["cell_type"].nunique(),
    )

    _ = model.train(data, label_key="cell_type", batch_size=2, epochs=3)
    assert model.trained

    predictions = model.predict(data, label_key="cell_type")
    assert len(predictions) == len(data)
    assert all([p in data.obs["cell_type"].cat.categories for p in predictions])
