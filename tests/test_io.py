import pytest

from gitk.io.io import Region, RegionSet


@pytest.fixture
def universe_bed_file():
    return "tests/data/universe.bed"


@pytest.fixture
def gz_file():
    return "tests/data/universe.bed.gz"


def test_make_region():
    r = Region("chr1", 0, 100)
    assert r is not None
    assert r.chr == "chr1"
    assert r.start == 0
    assert r.end == 100


def test_make_region_set(universe_bed_file: str):
    u = RegionSet(universe_bed_file)
    assert u is not None
    assert len(u) == 2_433

    # test we can iterate over it
    for region in u:
        assert isinstance(region, Region)


def test_make_region_set_with_backed(universe_bed_file: str):
    u = RegionSet(universe_bed_file, backed=True)
    assert u is not None
    assert len(u) == 2_433

    # test we can iterate over it
    for region in u:
        assert isinstance(region, Region)


def test_make_region_set_with_list():
    regions = [Region("chr1", 0, 100), Region("chr1", 100, 200)]
    u = RegionSet(regions)
    assert u is not None
    assert len(u) == 2

    # test we can iterate over it
    for region in u:
        assert isinstance(region, Region)


def test_make_region_set_with_gz_file(gz_file: str):
    u = RegionSet(gz_file)
    assert u is not None
    assert len(u) == 2_433

    # test we can iterate over it
    for region in u:
        assert isinstance(region, Region)


def test_make_region_set_with_gz_file_backed(gz_file: str):
    u = RegionSet(gz_file, backed=True)
    assert u is not None
    assert len(u) == 2_433

    # test we can iterate over it
    for region in u:
        assert isinstance(region, Region)
