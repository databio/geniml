import pytest
import os

from geniml.io.io import SNP, Maf, Region, RegionSet
from geniml.io.exceptions import GenimlBaseError
import genomicranges

DATA_TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "io_data",
)
DATA_TEST_FOLDER_BED = os.path.join(DATA_TEST_FOLDER, "bed")
DATA_TEST_FOLDER_MAF = os.path.join(DATA_TEST_FOLDER, "maf")

ALL_BEDFILE_PATH = [
    os.path.join(DATA_TEST_FOLDER_BED, x) for x in os.listdir(DATA_TEST_FOLDER_BED)
]
ALL_MAF_PATH = [os.path.join(DATA_TEST_FOLDER_MAF, x) for x in os.listdir(DATA_TEST_FOLDER_MAF)]


def test_make_region():
    r = Region("chr1", 0, 100)
    assert r is not None
    assert r.chr == "chr1"
    assert r.start == 0
    assert r.end == 100


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


class TestRegionSet:
    @pytest.mark.parametrize(
        "url",
        [
            "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7666nnn/GSM7666464/suppl/GSM7666464_18134-282-06_S51_L003_peaks.narrowPeak.gz"
        ],
    )
    def test_region_set_from_url(self, url):
        region_set = RegionSet(url)
        assert isinstance(region_set.regions[0], Region)

    @pytest.mark.parametrize("url", ALL_BEDFILE_PATH)
    def test_region_set_from_path_backed(self, url):
        region_set = RegionSet(url)
        assert isinstance(region_set.regions[0], Region)

    @pytest.mark.parametrize("url", ALL_BEDFILE_PATH)
    def test_region_set_from_path(self, url):
        region_set = RegionSet(url, backed=True)
        for region in region_set:
            assert isinstance(region, Region)
            break

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7666nnn/GSM7666464/suppl/GSM7666464_18134-282-06_S51_L003_peaks.narrowPeak.gz"
        ],  # This is not the right way how to do it!
    )
    def test_region_set_from_url_cant_be_backed(self, url):
        with pytest.raises(GenimlBaseError):
            RegionSet(url, backed=True)

    def test_make_region_set_with_list(
        self,
    ):
        regions = [Region("chr1", 0, 100), Region("chr1", 100, 200)]
        u = RegionSet(regions)
        assert u is not None
        assert len(u) == 2

        # test we can iterate over it
        for region in u:
            assert isinstance(region, Region)

    @pytest.mark.parametrize("path", ALL_BEDFILE_PATH)
    def test_to_genomic_ranges(self, path):
        bedfile = RegionSet(path)
        gr = bedfile.to_granges()  # should return a GenomicRanges object
        assert isinstance(gr, genomicranges.GenomicRanges)

    def test_calculation_id(self):
        bedfile_id_1 = RegionSet(ALL_BEDFILE_PATH[0]).identifier
        bedfile_id_2 = RegionSet(ALL_BEDFILE_PATH[1]).identifier
        bedfile_id_3 = RegionSet(ALL_BEDFILE_PATH[2]).identifier
        assert len(bedfile_id_2) == 32
        assert bedfile_id_1 == bedfile_id_2 == bedfile_id_3


class TestMaff:
    @pytest.mark.parametrize("path", ALL_MAF_PATH)
    def test_maf_from_path(self, path):
        snps = Maf(path)
        assert snps is not None
        for s in snps:
            assert isinstance(s, SNP)
            break

    @pytest.mark.parametrize("path", ALL_MAF_PATH)
    def test_read_maf_file_backed(
        self,
        path: str,
    ):
        snps = Maf(path, backed=True)
        assert snps is not None
        assert len(snps) == 99
        for s in snps:
            assert isinstance(s, SNP)

    @pytest.mark.parametrize("path", ALL_MAF_PATH)
    def test_snps_to_regions(
        self,
        path: str,
    ):
        snps = Maf(path)
        for snp in snps:
            assert isinstance(snp, SNP)
            assert isinstance(snp.to_region(), Region)
