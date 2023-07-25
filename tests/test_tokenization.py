import os
import sys

import pytest
import scanpy as sc
import numpy as np

from gitk.io import Region
from gitk.tokenization import Universe, InMemTokenizer

# add parent directory to path
sys.path.append("../")


@pytest.fixture
def adata():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad")


@pytest.fixture
def universe_bed_file():
    return "tests/data/universe.bed"


def test_create_universe(universe_bed_file: str):
    u = Universe(universe_bed_file)
    assert u is not None
    assert len(u) == 10_000


def test_region_in_universe(universe_bed_file: str):
    u = Universe(universe_bed_file)

    # regions in
    r_is_in1 = Region("chr17", 78_168_158, 78_169_026)
    r_is_in2 = Region(
        "chr2",
        241_859_589,
        241_860_443,
    )
    r_is_in3 = Region(
        "chr22",
        38_863_328,
        38_864_072,
    )

    # regions not in
    r_not_in1 = Region(
        "chr22",
        1,
        100,
    )
    r_not_in2 = Region(
        "chr17",
        200,
        300,
    )
    r_not_in3 = Region(
        "chr2",
        1_000_000_000_000,
        1_000_000_000_100,
    )

    assert r_is_in1 in u
    assert r_is_in2 in u
    assert r_is_in3 in u

    assert r_not_in1 not in u
    assert r_not_in2 not in u
    assert r_not_in3 not in u


def test_universe_query(universe_bed_file: str):
    u = Universe(universe_bed_file)

    regions = [
        Region("chr17", 78_168_158, 78_169_026),
        Region(
            "chr2",
            241_859_589,
            241_860_443,
        ),
        Region(
            "chr2",
            1_000_000_000_000,
            1_000_000_000_100,
        ),
    ]

    # query regions
    olaps = u.query(regions)

    assert len(olaps) == 2


def test_make_in_mem_tokenizer(universe_bed_file: str):
    t = InMemTokenizer(universe_bed_file)
    assert t is not None


def test_tokenize_bed_file(universe_bed_file: str):
    """
    Use the in memory tokenizer to tokenize a bed file.

    The bed file contains 13 regions, 10 of which are in the universe. None
    of them should be the original regions in the file.
    """
    t = InMemTokenizer(universe_bed_file)
    assert t is not None

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in the bed file to test
    with open(bed_file, "r") as f:
        lines = f.readlines()
        regions = []
        for line in lines:
            chr, start, stop = line.strip().split("\t")
            regions.append(Region(chr, int(start), int(stop)))

    tokens = t.tokenize(bed_file)

    # ensure that the tokens are unqiue from the original regions
    assert len(set(tokens).intersection(set(regions))) == 0
    assert len(tokens) == 10
