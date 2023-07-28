import pytest
import scanpy as sc
import numpy as np

from gitk.io.io import Region, RegionSet


@pytest.fixture
def universe_bed_file():
    return "tests/data/universe.bed"


def test_make_region():
    r = Region("chr1", 0, 100)
    assert r is not None
    assert r.chr == "chr1"
    assert r.start == 0
    assert r.end == 100


def test_make_region_set(universe_bed_file: str):
    u = RegionSet(universe_bed_file)
    assert u is not None
    assert len(u) == 10_000

    # test we can iterate over it
    for region in u:
        assert isinstance(region, Region)


def test_make_region_set_with_backed(universe_bed_file: str):
    u = RegionSet(universe_bed_file, backed=True)
    assert u is not None
    assert len(u) == 10_000

    # test we can iterate over it
    for region in u:
        assert isinstance(region, Region)
