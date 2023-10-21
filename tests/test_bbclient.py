import gzip
import os
import shutil
from typing import Tuple

import genomicranges
import pytest

from geniml.bbclient import BBClient
from geniml.io import RegionSet, BedSet


@pytest.fixture
def cache_path():
    return "./data/geniml_bb_cache"


@pytest.fixture
def bedset_id():
    """
    identifier of bedset
    """

    return "b4b0466f55d8e7b5ffd19ec35eac40d2"


@pytest.fixture
def bedfile_id():
    return "e7e9893792c90a7ef96be9fe333c6c1d"


@pytest.fixture
def local_bedfile_path():
    return "./data/s1_a.bed"


@pytest.fixture
def local_bedgz_path():
    return "./data/s1_a.bed.gz"


@pytest.fixture
def local_bedfile_list():
    return ["./data/s2_a.bed", "./data/s3_a.bed"]


def test_load_bedset_from_bedbase(cache_path, bedset_id):
    """
    Testing loading of a BED set from BEDbase

    :param cache_path: the local cache folder
    :param bedset_id: the identifier of BED set to be loaded from BED base
    """
    # init BBClient
    bbclient = BBClient(cache_folder=cache_path)
    # check the seek result before the BED set is loaded
    bedset_seek_result = bbclient.seek(bedset_id)
    assert "does not exist" in bedset_seek_result

    # load the bedset
    bedset = bbclient.load_bedset(bedset_id)
    # check the GenomicRangesList from the BED set
    grl = bedset.to_grangeslist()
    assert isinstance(grl, genomicranges.GenomicRangesList)

    # check the path and identifier of BED files which the loaded BED set contains
    for bedfile in bedset:
        bedfile_path = bbclient.seek(bedfile.compute_bed_identifier())
        assert bedfile.compute_bed_identifier() == os.path.split(bedfile_path)[1].split(".")[0]


def test_load_bedfile_from_bedbase(cache_path, bedfile_id):
    """
    Testing loading of a BED set from BEDbase

    :param cache_path: the local cache folder
    :param bedfile_id: the identifier of BED file to be loaded from BED base
    """
    bbclient = BBClient(cache_folder=cache_path)

    # check the seek result before the BED file is loaded
    # bedfile_seek_result = bbclient.seek(bedfile_id)
    # assert "does not exist" in bedfile_seek_result

    # load the bedfile
    bedfile = bbclient.load_bed(bedfile_id)

    # check the GenomicRanges from the BED file
    gr = bedfile.to_granges()
    assert isinstance(gr, genomicranges.GenomicRanges)

    # check the path and identifier of BED file
    bedfile_path = bbclient.seek(bedfile_id)
    assert bedfile.compute_bed_identifier() == os.path.split(bedfile_path)[1].split(".")[0]


def test_load_local_bed_file(cache_path, local_bedfile_path, local_bedgz_path):
    """
    Testing loading of a BED file from local file
    :param cache_path: the local cache folder
    :param local_bedfile_path: path of bed file to be loaded
    :param local_bedgz_path: path of gziped bedfile
    """
    bbclient = BBClient(cache_folder=cache_path)
    # load a local bedfile (not from bedbase)
    bedfile = RegionSet(local_bedfile_path)
    gr = bedfile.to_granges()  # should return a GenomicRanges object
    bedfile_id = bedfile.compute_bed_identifier()  # just compute its ID, without adding to cache

    bedfile_id = bbclient.add_bed_to_cache(bedfile)  # compute its ID and add it to the cache
    path_in_cache = bbclient.seek(bedfile.identifier)
    # check the path and identifier of BED file
    assert bedfile.compute_bed_identifier() == os.path.split(path_in_cache)[1].split(".")[0]

    # testing if same RegionSet from same file but init with different conditions:
    # 1. backed or not
    # 2. file gziped or not
    # will give same identifier
    bedfile_backed = RegionSet(local_bedfile_path, backed=True)
    # load the local bedfile from gziped
    with open(local_bedfile_path, "rb") as f_in:
        with gzip.open(local_bedgz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    bedfile_gz = RegionSet(local_bedgz_path)
    bedfile_backed_gz = RegionSet(local_bedgz_path, backed=True)
    bedfile_in_cache = RegionSet(path_in_cache)
    bedfile_in_cache_backed = RegionSet(path_in_cache, backed=True)

    assert (
        bedfile.compute_bed_identifier()
        == bedfile_gz.compute_bed_identifier()
        == bedfile_in_cache.compute_bed_identifier()
        == bedfile_backed.compute_bed_identifier()
        == bedfile_backed_gz.compute_bed_identifier()
        == bedfile_in_cache_backed.compute_bed_identifier()
    )
    os.remove(local_bedgz_path)


def test_load_local_bedset(cache_path, local_bedfile_list):
    """
    Testing loading of a BED set from local file

    :param cache_path: the local cache folder
    :param local_bedfile_list: list of bed files in the bed set
    """

    bbclient = BBClient(cache_folder=cache_path)
    # load a local bedset file (not from bedbase)
    bedset = BedSet(local_bedfile_list)
    assert isinstance(bedset, BedSet)
    bedset_id = bedset.compute_bedset_identifier()

    # check the seek result before the BED set is loaded
    bedset_seek_result = bbclient.seek(bedset_id)
    assert "does not exist" in bedset_seek_result

    bbclient.add_bedset_to_cache(bedset)

    # check the path and identifier of BED set
    path_in_cache = bbclient.seek(bedset_id)
    assert bedset_id == os.path.split(path_in_cache)[1].split(".")[0]


def test_removal(cache_path, bedset_id, bedfile_id):
    """
    Test removal of bed set and bed file

    :param cache_path: the local cache folder
    :param bedset_id: the identifier of bed set to be removed
    :param bedfile_id: the identifier of bed file to be removed
    """

    def subfolders(cache_path) -> Tuple[str, str]:
        """
        Return the subfolders matching first two digit of the identifier
        """
        # the subfolder that matches the second digit of the identifier
        sub_folder_2 = os.path.split(cache_path)[0]
        # the subfolder that matches the first digit of the identifier
        sub_folder_1 = os.path.split(sub_folder_2)[0]

        return sub_folder_1, sub_folder_2

    bbclient = BBClient(cache_folder=cache_path)
    # testing removal
    # remove bedfile
    bedfile_cache_path = bbclient.seek(bedfile_id)
    subfolder1, subfolder2 = subfolders(bedfile_cache_path)
    bbclient.remove_bedfile_from_cache(bedfile_id)
    # check no empty subfolders exist
    assert not os.path.exists(subfolder2) or len(os.listdir(subfolder2)) > 0
    assert not os.path.exists(subfolder1) or len(os.listdir(subfolder1)) > 0

    # remove bedset
    bedset_cache_path = bbclient.seek(bedset_id)
    subfolder1, subfolder2 = subfolders(bedset_cache_path)
    bbclient.remove_bedset_from_cache(bedset_id)
    # check no empty subfolders exist
    assert not os.path.exists(subfolder2) or len(os.listdir(subfolder2)) > 0
    assert not os.path.exists(subfolder1) or len(os.listdir(subfolder1)) > 0
    shutil.rmtree(cache_path)
