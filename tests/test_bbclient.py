import os

import genomicranges
import pytest
from geniml.bbclient import BBClient
from geniml.io import BedSet, RegionSet

DATA_TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "io_data",
    "bed",
)
ALL_BEDFILE_PATH = [os.path.join(DATA_TEST_FOLDER, x) for x in os.listdir(DATA_TEST_FOLDER)]


@pytest.fixture
def cache_path():
    return "./data/geniml_bb_cache"


@pytest.fixture
def bedset_id():
    """
    identifier of bedset
    """
    return "2f5fbf7a65c1b666e38bccf7ed9da5e6"


@pytest.fixture
def bedfile_id():
    return "38856b21ff44f584e48081e0db51db0c"


@pytest.fixture
def local_bedfile_path():
    return ALL_BEDFILE_PATH[0]


@pytest.fixture
def local_bedfile_list():
    return ALL_BEDFILE_PATH


class TestBBClientCaching:
    def test_init(self, cache_path):
        """
        Test initialization of BBClient
        """
        bbclient = BBClient(cache_folder=cache_path)
        assert bbclient is not None

    def test_init_no_cache_folder(self):
        """
        Test initialization of BBClient without cache folder
        """
        with pytest.raises(TypeError):
            BBClient(cache_folder=None)

    def test_bed_caching_from_path(self, tmp_path, local_bedfile_path):
        bbclient = BBClient(cache_folder=tmp_path)
        bedfile_id = bbclient.add_bed_to_cache(local_bedfile_path)
        assert bedfile_id is not None

    def test_bed_caching_from_region_set(self, tmp_path, local_bedfile_path):
        bbclient = BBClient(cache_folder=tmp_path)
        bedfile = RegionSet(local_bedfile_path)
        bbclient.add_bed_to_cache(bedfile)
        path_in_cache = bbclient.seek(bedfile.identifier)
        assert bedfile.compute_bed_identifier() == os.path.split(path_in_cache)[1].split(".")[0]

    def test_bedset_caching(self, tmp_path, local_bedfile_list):
        bbclient = BBClient(cache_folder=tmp_path)
        bedset = BedSet(local_bedfile_list)
        bedset_id = bbclient.add_bedset_to_cache(bedset)
        path_in_cache = bbclient.seek(bedset_id)
        assert bedset_id == os.path.split(path_in_cache)[1].split(".")[0]

    def test_error_bed_not_in_cache(self, tmp_path, local_bedfile_path):
        bbclient = BBClient(cache_folder=tmp_path)
        # skip caching
        bedfile = RegionSet(local_bedfile_path)
        with pytest.raises(FileNotFoundError):
            bbclient.seek(bedfile.identifier)

    def test_error_bedset_not_in_cache(self, tmp_path, local_bedfile_list):
        bbclient = BBClient(cache_folder=tmp_path)
        bedset_id = BedSet(local_bedfile_list).identifier
        with pytest.raises(FileNotFoundError):
            bbclient.seek(bedset_id)

    def test_remove_bed_from_cache(self, tmp_path, local_bedfile_path):
        bbclient = BBClient(cache_folder=tmp_path)
        bedfile_id = bbclient.add_bed_to_cache(local_bedfile_path)
        assert bbclient.seek(bedfile_id)
        bbclient.remove_bedfile_from_cache(bedfile_id)
        with pytest.raises(FileNotFoundError):
            bbclient.seek(bedfile_id)

    def test_remove_bedset_from_cache(self, tmp_path, local_bedfile_list):
        bbclient = BBClient(cache_folder=tmp_path)
        bedset = BedSet(local_bedfile_list)
        bedset_id = bbclient.add_bedset_to_cache(bedset)
        assert bbclient.seek(bedset_id)
        bbclient.remove_bedset_from_cache(bedset_id)
        with pytest.raises(FileNotFoundError):
            bbclient.seek(bedset_id)


# @pytest.mark.bedbase
@pytest.mark.skipif(
    "not config.getoption('--bedbase')",
    reason="Only run when --bedbase is given",
)
def test_bedbase_caching(tmp_path, bedset_id, bedfile_id, request):
    """
    Testing caching BED files and BED sets from bedbase files
    only tested when
    """
    bbclient = BBClient(cache_folder=tmp_path)
    bedset = bbclient.load_bedset(bedset_id)
    # check the GenomicRangesList from the BED set
    grl = bedset.to_granges_list()
    assert isinstance(grl, genomicranges.GenomicRangesList)

    # check the path and identifier of BED files which the loaded BED set contains
    for bedfile in bedset:
        bedfile_path = bbclient.seek(bedfile.compute_bed_identifier())
        print(bedfile.compute_bed_identifier())
        print(os.path.split(bedfile_path)[1].split(".")[0])
        assert bedfile.compute_bed_identifier() == os.path.split(bedfile_path)[1].split(".")[0]

    bedfile = bbclient.load_bed(bedfile_id)

    # check the GenomicRanges from the BED file
    gr = bedfile.to_granges()
    assert isinstance(gr, genomicranges.GenomicRanges)

    # check the path and identifier of BED file
    bedfile_path = bbclient.seek(bedfile_id)
    assert bedfile.compute_bed_identifier() == os.path.split(bedfile_path)[1].split(".")[0]
