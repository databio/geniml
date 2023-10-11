import gzip
import os
import shutil
from typing import Tuple

import pytest

from geniml.bbclient import BBClient, BedFile, BedSet
from geniml.io import RegionSet


@pytest.fixture
def cache_path():
    return "./data/geniml_bb_cache"


@pytest.fixture
def bedset_identifier():
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


def test_bedset_retrievel(
    cache_path,
    bedset_identifier,
    bedfile_id,
    local_bedfile_path,
    local_bedgz_path,
    local_bedfile_list,
):
    def subfolders(cache_path) -> Tuple[str, str]:
        """
        Return the subfolders matching first two digit of the identifier
        """
        # the subfolder that matches the second digit of the identifier
        sub_folder_2 = os.path.split(cache_path)[0]
        # the subfolder that matches the first digit of the identifier
        sub_folder_1 = os.path.split(sub_folder_2)[0]

        return sub_folder_1, sub_folder_2

    # create a cache path
    assert not os.path.exists(cache_path)
    bbclient = BBClient(cache_folder=cache_path)
    assert os.path.exists(cache_path)

    # load a BedSet from BEDbase
    bedset = bbclient.load_bedset(bedset_identifier)
    grl = bedset.to_grangeslist()
    for bedfile in bedset:
        print(bedfile.identifier)

    # load a bed file from BEDbase
    bedfile = bbclient.load_bed(bedfile_id)
    pth = bedfile.path
    print(pth)
    gr = bedfile.to_granges()

    # load a local bedfile (not from bedbase)
    bedfile = BedFile(local_bedfile_path)
    gr = bedfile.to_granges()  # should return a GenomicRanges object
    bedfile_id = bedfile.compute_bed_identifier()  # just compute its ID, without adding to cache

    bedfile_id = bbclient.add_bed_to_cache(bedfile)  # compute its ID and add it to the cache
    path_in_cache = bbclient.seek(bedfile.identifier)
    print(path_in_cache)

    # testing if same BedFile from same file but init with different conditions:
    # backed or not
    # file gziped or not
    # will give same identifier
    bedfile_backed = BedFile(local_bedfile_path, backed=True)
    # load the local bedfile from gziped
    with open(local_bedfile_path, "rb") as f_in:
        with gzip.open(local_bedgz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    bedfile_gz = BedFile(local_bedgz_path)
    bedfile_backed_gz = BedFile(local_bedgz_path, backed=True)
    bedfile_in_cache = BedFile(path_in_cache)
    bedfile_in_cache_backed = BedFile(path_in_cache, backed=True)

    assert (
        bedfile.compute_bed_identifier()
        == bedfile_gz.compute_bed_identifier()
        == bedfile_in_cache.compute_bed_identifier()
        == bedfile_backed.compute_bed_identifier()
        == bedfile_backed_gz.compute_bed_identifier()
        == bedfile_in_cache_backed.compute_bed_identifier()
    )
    os.remove(local_bedgz_path)

    # load a local bedset file (not from bedbase)
    bedset = BedSet(local_bedfile_list)
    bedset_id = bedset.compute_identifier()
    bbclient.add_bedset_to_cache(bedset)

    # testing removal
    # remove bedfile:
    bedfile_id = bedfile.compute_bed_identifier()
    bedfile_cache_path = bbclient.seek(bedfile_id)
    subfolder1, subfolder2 = subfolders(bedfile_cache_path)
    bbclient.remove_bedfile_from_cache(bedfile_id)
    # check no empty subfolders exist
    assert not os.path.exists(subfolder2) or len(os.listdir(subfolder2)) > 0
    assert not os.path.exists(subfolder1) or len(os.listdir(subfolder1)) > 0

    # remove bedset
    bedset_id = bedset.compute_identifier()
    bedset_cache_path = bbclient.seek(bedset_id)
    subfolder1, subfolder2 = subfolders(bedset_cache_path)
    bbclient.remove_bedset_from_cache(bedset_id)
    # check no empty subfolders exist
    assert not os.path.exists(subfolder2) or len(os.listdir(subfolder2)) > 0
    assert not os.path.exists(subfolder1) or len(os.listdir(subfolder1)) > 0
    shutil.rmtree(cache_path)
