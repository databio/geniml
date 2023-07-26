import os
import sys

import pytest
import scanpy as sc
import numpy as np

from gitk.io import Region, RegionSet
from gitk.tokenization import InMemTokenizer

# add parent directory to path
sys.path.append("../")


@pytest.fixture
def adata():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad")


@pytest.fixture
def universe_bed_file():
    return "tests/data/universe.bed"


def test_create_universe(universe_bed_file: str):
    u = RegionSet(universe_bed_file)
    assert u is not None
    assert len(u) == 10_000


def test_make_in_mem_tokenizer_from_file(universe_bed_file: str):
    t = InMemTokenizer(universe_bed_file)
    assert len(t.universe) == 10_000
    assert t is not None


def test_make_in_mem_tokenizer_from_region_set(universe_bed_file: str):
    u = RegionSet(universe_bed_file)
    t = InMemTokenizer(u)
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


def test_tokenize_list_of_regions(universe_bed_file: str):
    """
    Use the in memory tokenizer to tokenize a list of regions.

    The bed file contains 13 regions, 10 of which are in the universe. None
    of them should be the original regions in the file.
    """
    t = InMemTokenizer(universe_bed_file)
    assert t is not None

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in each and cast as a region
    with open(bed_file, "r") as f:
        lines = f.readlines()
        regions = []
        for line in lines:
            chr, start, stop = line.strip().split("\t")
            regions.append(Region(chr, int(start), int(stop)))

    tokens = t.tokenize(regions)

    # ensure that the tokens are unqiue from the original regions
    assert len(set(tokens).intersection(set(regions))) == 0
    assert len(tokens) == 10
