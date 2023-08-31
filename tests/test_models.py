import os
import sys

import pytest
import scanpy as sc
import numpy as np

# add parent directory to path
sys.path.append("../")

from geniml import models


@pytest.fixture
def adata():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad")


@pytest.fixture
def tokenizer_from_file():
    return models.HardTokenizer("tests/data/universe.bed")


@pytest.fixture
def tokenizer_from_list():
    return models.HardTokenizer(["chr1_0_100", "chr2_0_100"])


@pytest.fixture
def universe_from_file():
    return models.Universe("tests/data/universe.bed")


@pytest.fixture
def pretrained_model():
    return models.PretrainedScembedModel("nleroy917/luecken2021")


def test_init_tokenizers(
    tokenizer_from_file: models.HardTokenizer,
    tokenizer_from_list: models.HardTokenizer,
):
    # assert that the tokenizers were created
    assert tokenizer_from_file is not None
    assert tokenizer_from_list is not None

    # assert that the regions file was created
    assert all([isinstance(t, str) for t in tokenizer_from_list.regions])
    assert all([isinstance(t, str) for t in tokenizer_from_file.regions])


def test_init_universe(
    universe_from_file: models.Universe,
):
    # assert that the universe was created
    assert universe_from_file is not None

    # assert interval trees were created
    for tree in universe_from_file:
        assert tree is not None


def test_universe_set(
    universe_from_file: models.Universe,
):
    # assert that the universe_set is the same length as the number of regions
    # in the universe file
    universe_set = universe_from_file.universe_set
    assert len(universe_set) == 10000


def test_universe_contains(universe_from_file: models.Universe):
    regions = [
        ("chr11", 62732530, 62733438),
        ("chr1", 34918742, 34919559),
    ]
    for region in regions:
        assert region in universe_from_file


def test_universe_not_contains(universe_from_file: models.Universe):
    regions = [("chr10", 132, 1324), ("chr1", 247, 2478), ("chr999", 0, 100)]
    for region in regions:
        assert region not in universe_from_file


def test_universe_query(universe_from_file: models.Universe):
    regions = [
        ("chr10", 132, 1324),
        ("chr11", 62732550, 62733458),
        ("chr1", 34918762, 34919579),
    ]
    res = universe_from_file.query(regions)
    assert len(res) == 2


def test_universe_conversion():
    """
    Test tokenization for three scenarios:
    1. One overlap in B per region in A
    2. At least one region in A doesn't overlap with any region in B
    3. A region in A overlaps with multiple regions in B

    Scenario 1:
    A: |-----------|        |-----------|           |-----------|
    B:     |-----------|        |-----------|           |-----------|

    Scenario 2:
    A: |-----------|        |-------|              |-----------|
    B:     |-----------|                 |-------|          |-----------|

    Scenario 3:
    A: |-----------|        |-----------|           |-----------|
    B:     |-----------|  |--------| |-----------|           |-----------|

    """
    # scenario 1
    with open("tests/data/s1_a.bed", "r") as f:
        lines = f.readlines()
        a = [tuple(l.strip().split("\t")) for l in lines]
    u = models.Universe("tests/data/s1_b.bed")

    conversion_map = models.utils.generate_var_conversion_map(a, u)
    assert conversion_map == {
        "chr1_10_30": "chr1_20_40",
        "chr1_110_130": "chr1_120_140",
        "chr1_210_230": "chr1_220_240",
    }

    # scenario 2
    with open("tests/data/s2_a.bed", "r") as f:
        lines = f.readlines()
        a = [tuple(l.strip().split("\t")) for l in lines]
    u = models.Universe("tests/data/s2_b.bed")

    conversion_map = models.utils.generate_var_conversion_map(a, u)
    assert conversion_map == {
        "chr1_10_30": "chr1_20_40",
        "chr1_110_130": None,
        "chr1_210_230": "chr1_220_240",
    }

    # scenario 3
    with open("tests/data/s3_a.bed", "r") as f:
        lines = f.readlines()
        a = [tuple(l.strip().split("\t")) for l in lines]
    u = models.Universe("tests/data/s3_b.bed")

    conversion_map = models.utils.generate_var_conversion_map(a, u)
    assert conversion_map == {
        "chr1_10_30": "chr1_20_40",
        "chr1_110_130": "chr1_100_120",
        "chr1_210_230": "chr1_220_240",
    }


def test_tokenize_bedfile(tokenizer_from_file: models.HardTokenizer):
    # assert that the tokenizer works
    tokens = tokenizer_from_file.tokenize("tests/data/to_tokenize.bed")
    assert tokens is not None
    print(len(tokens))
    assert len(tokens) == 10
    assert all([len(t.split("_")) == 3 for t in tokens])


def test_tokenize_anndata(adata: sc.AnnData, tokenizer_from_file: models.HardTokenizer):
    tokens: list[list[str]] = tokenizer_from_file.tokenize(adata)
    # make sure each token in each cell is inside the universe.bed file
    for cell in tokens:
        for token in cell:
            chr, start, end = token.split("_")
            token_tuple = (chr, int(start), int(end))
            assert token_tuple in tokenizer_from_file.universe.universe_set


# skip
@pytest.mark.skip
def test_init_pretrained_model(
    pretrained_model: models.PretrainedScembedModel,
):
    # assert that the model was created
    assert pretrained_model is not None

    # assert that the tokenizer was created
    assert pretrained_model.tokenizer is not None

    # assert that the model was created
    assert pretrained_model.model is not None


@pytest.mark.skip
def test_encode_anndata(adata: sc.AnnData, pretrained_model: models.PretrainedScembedModel):
    # encode the anndata
    encoded = pretrained_model.encode(adata)

    # assert that the encoded object is a numpy array
    assert isinstance(encoded, np.ndarray)

    # assert that the encoded object has the correct shape
    assert encoded.shape == (adata.n_obs, 100)
