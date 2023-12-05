import os
from typing import Literal

import torch
import pytest
import numpy as np
import scanpy as sc

from geniml.tokenization.main import ITTokenizer
from geniml.classification.utils import generate_fine_tuning_dataset
from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.classification.finetuning import Region2VecFineTuner, RegionSet2Vec


def test_generate_finetuning_dataset():
    t = ITTokenizer("tests/data/universe.bed")
    adata = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")

    pos, neg, pos_labels, neg_labels = generate_fine_tuning_dataset(adata, t, negative_ratio=1.0)

    # total positive pairs should be equal to total negative pairs
    # total positive pairs will be equal to sum([n*(n - 1) for n in adata.obs.groupby("cell_type").size()])
    # total negative pairs only equals number of positive pairs when negative_ratio=1.0

    assert len(pos) == len(neg)
    assert len(pos) == len(pos_labels)
    assert len(neg) == len(neg_labels)
    assert len(pos) == sum([(n * (n - 1)) for n in adata.obs.groupby("cell_type").size()])


def test_init_region2vec_finetuner_from_scratch():
    """
    Test the initialization of a FineTuner from scratch.
    """
    region2vec = Region2Vec(2380, 100)  # 1000 vocab size, 100 embedding size
    rs2v = Region2VecFineTuner(tokenizer="tests/data/universe.bed", region2vec=region2vec)

    assert rs2v.region2vec.embedding_dim == 100
    assert rs2v.region2vec == region2vec


@pytest.mark.skip(reason="Model is too big to download in the runner, takes too long.")
def test_init_classifier_from_pretrained():
    # get pre-trained r2v
    r2v = Region2VecExModel("databio/r2v-ChIP-atlas-hg38-v2")

    # create RegionSet2Vec
    rs2v = RegionSet2Vec(r2v.model)

    assert rs2v.region2vec.embedding_dim == 100
    assert rs2v.region2vec == r2v.model


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
    model = Region2VecFineTuner(region2vec=r2v, tokenizer=tokenizer)
    assert model is not None
    assert model.region2vec is not None
    assert model.tokenizer is not None


def test_train_exmodel():
    data = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")
    model = Region2VecFineTuner(
        region2vec=Region2Vec(2380, 100),
        tokenizer="tests/data/universe.bed",
    )
    torch.manual_seed(0)  # for reproducibility
    result = model.train(data, label_key="cell_type", batch_size=2, epochs=50, seed=42)
    assert result is not None
    assert model.trained
    assert result.epoch_loss[0] > result.all_loss[-1]


def test_export_exmodel():
    r2v = Region2Vec(2380, 100)
    model = Region2VecFineTuner(region2vec=r2v, tokenizer="tests/data/universe.bed")
    assert model is not None

    # grab some weights from the inner r2v
    r2v_weights = model.region2vec.projection.weight.detach().clone().numpy()

    # save
    try:
        model.export("tests/data/model-tests")
        model = Region2VecFineTuner.from_pretrained("tests/data/model-tests")

        # ensure model is still trained and has region2vec
        assert model.trained

        # ensure weights are the same
        assert np.allclose(
            r2v_weights, model.region2vec.projection.weight.detach().clone().numpy()
        )
    finally:
        for file in os.listdir("tests/data/model-tests"):
            os.remove(os.path.join("tests/data/model-tests", file))


def test_train_ex_model_save_load():
    data = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")
    model = Region2VecFineTuner(
        region2vec=Region2Vec(2380, 100),
        tokenizer="tests/data/universe.bed",
    )
    torch.manual_seed(0)  # for reproducibility
    result = model.train(data, label_key="cell_type", batch_size=2, epochs=50, seed=42)
    assert model.trained
    assert result.all_loss[0] > result.all_loss[-1]

    input_tokens = [t.id for t in model.tokenizer.tokenize(data[0, :].to_memory())[0]]
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)

    pre_save_output = model._model(input_tokens)

    model.export("tests/data/model-tests")
    model = Region2VecFineTuner.from_pretrained("tests/data/model-tests")

    post_save_output = model._model(input_tokens)

    assert np.allclose(pre_save_output.detach().numpy(), post_save_output.detach().numpy())
