import os
import sys

import pytest
import scanpy as sc
import numpy as np

from gitk.io import Region
from gitk.tokenization import Universe

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
