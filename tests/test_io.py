import pytest

from geniml.io.io import Region, RegionSet, Maf, SNP


@pytest.fixture
def universe_bed_file():
    return "tests/data/universe.bed"


@pytest.fixture
def gz_file():
    return "tests/data/universe.bed.gz"


@pytest.fixture
def maf_file():
    return "tests/data/sample.maf"


@pytest.fixture
def gz_maf_file():
    return "tests/data/sample.maf.gz"


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


def test_make_SNP():
    s = SNP(
        hugo_symbol="TP53",
        chromosome="chr1",
        start_position=0,
        end_position=1,
        ncbi_build="GRCh38",
        strand="+",
    )
    assert s is not None
    assert s.hugo_symbol == "TP53"
    assert s.chromosome == "chr1" and s.chr == "chr1"
    assert s.start_position == 0 and s.start == 0
    assert s.end_position == 1 and s.end == 1
    assert s.ncbi_build == "GRCh38"
    assert s.strand == "+"


def test_read_maf_file(
    maf_file: str,
):
    snps = Maf(maf_file)
    assert snps is not None
    assert len(snps) == 99
    for s in snps:
        assert isinstance(s, SNP)


def test_read_maf_file_backed(
    maf_file: str,
):
    snps = Maf(maf_file, backed=True)
    assert snps is not None
    assert len(snps) == 99
    for s in snps:
        assert isinstance(s, SNP)


def test_read_maf_file_gz(
    gz_maf_file: str,
):
    snps = Maf(gz_maf_file)
    assert snps is not None
    assert len(snps) == 99
    for s in snps:
        assert isinstance(s, SNP)


def test_read_maf_file_gz_backed(
    gz_maf_file: str,
):
    snps = Maf(gz_maf_file, backed=True)
    assert snps is not None
    assert len(snps) == 99
    for s in snps:
        assert isinstance(s, SNP)


def test_snps_to_regions(
    maf_file: str,
):
    snps = Maf(maf_file)
    for snp in snps:
        assert isinstance(snp, SNP)
        assert isinstance(snp.to_region(), Region)
