from typing import Literal

import torch
import pytest
import scanpy as sc

from geniml.tokenization.main import ITTokenizer
from geniml.classification.utils import generate_fine_tuning_dataset
from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.classification.finetuning import Region2VecFineTuner, RegionSet2Vec


def test_generate_finetuning_dataset():
    t = ITTokenizer("tests/data/universe.bed")
    adata = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")

    pos, neg, pos_labels, neg_labels = generate_fine_tuning_dataset(adata, t)

    assert len(pos) == len(neg)
    assert len(pos) == len(pos_labels)
    assert len(neg) == len(neg_labels)


def test_init_region2vec_finetuner_from_scratch():
    """
    Test the initialization of a FineTuner from scratch.
    """
    region2vec = Region2Vec(2380, 100)  # 1000 vocab size, 100 embedding size
    rs2v = Region2VecFineTuner(tokenizer="tests/data/universe.bed", region2vec=region2vec)

    assert rs2v.region2vec.embedding_dim == 100
    assert rs2v.region2vec == region2vec


def test_init_classifier_from_pretrained():
    # get pre-trained r2v
    r2v = Region2VecExModel("databio/r2v-ChIP-atlas-hg38-v2")

    # create classifier
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
