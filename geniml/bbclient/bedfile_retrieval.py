import os
from typing import List, Union

import genomicranges
import requests

from ..io import RegionSet
from .utils import (BedCacheManager, bedset_to_grangeslist,
                    create_bedset_from_file)

# How should I be able to use this?

# bbclient = BBClient(cache_folder="cache", bedbase_api="https://api.bedbase.org")
# bedset_identifier = "xyz" # find some interesting bedset on bedbase.org
# bedset = bbclient.load_bedset(bedset_identifier)  # should download, cache and return a BedSet object
# grl = bedset.to_grangeslist()  # should return a GenomicRangesList object

# get region set embeddings
# r2v_exmodel = Region2VecExModel("databio/r2v-ChIP-atlas-hg38")
# r2v_exmodel.get_embeddings(bedset)

# for bedfile in bedset:
#     print(bedfile.identifier)
#     r2v_exmodel.get_embeddings(bedfile)

# bedfile_id = "...."  # find interesting bedfile on bedbase
# bedfile = bbclient.load_bed(bedfile_id)  # should download, cache and return a RegionSet object
# gr = bedfile.to_granges()  # should return a GenomicRanges object

# Let's say I have a local bedfile, not on bedbase
# bedfile = RegionSet("path/to/bedfile")
# gr = bedfile.to_granges()  # should return a GenomicRanges object
# compute its ID and add it to the cache
# bbclient.add_bed_to_cache(bedfile)

# Let's say I want to create a new bedset
# bedset = BedSet(["path/to/bedfile1", "path/to/bedfile2"])
# or:
# rs1 = RegionSet("path/to/bedfile1")
# rs2 = RegionSet("path/to/bedfile2")
# rs3 = RegionSet(cool_new_identifier)  # from bedbase
# bedset = BedSet([rs1, rs2, rs3])
# bedset.compute_identifier() or bedset.identifier    # what's it's ID?
# I want to add it to my local cache
# bbclient.add_bedset_to_cache(bedset)
# This should also cache those 2 bed files, since they're part of the bedset


class BedSet(object):
    def __init__(
        self,
        region_sets: Union[List[RegionSets], List[str], List[List[Regions]], None],
        file_path: str = None,
        identifier: str = None,
    ):
        if isinstance(region_sets, List[RegionSets]):
            self.region_sets = region_sets
        elif isinstance(region_sets, List[str]):
            self.region_sets = []
            for r in region_sets:
                self.region_sets.append(RegionSet(r))  # Needs to run through bbclient
        elif isinstance(region_sets, List[List[Region]]):
            self.region_sets = []
            for r in region_sets:
                self.region_sets.append(RegionSet(r))
        elif file_path is not None:
            if os.path.isfile(file_path):
                self.region_sets = [RegionSet(r) for r in read_bedset_file(file_path)]
            else:
                raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
            
            for r in read_bedset_file(region_sets):
                self.region_sets.append(RegionSet(r))

        self.bedset_identifier = identifier or self.compute_identifier()

    def __len__(self):
        return len(self.region_sets)

    def __iter__(self):
        for region_set in self.region_sets:
            yield region_set

    def __getitem__(self, indx: int):
        return self.region_sets[indx]

    def compute_identifier(self):
        # TODO: set the bedset identifier
        # If the bedset identifier is not set, we should set it using
        # the algorithm we use to compute bedset identifiers
        # (see bedboss/bedbuncher pipeline)
        # I believe the bedset identifier is computed in bedbuncher.py line 76 with function 'get_bedset_digest'

        # something like this?
        import hashlib as md5

        if self.bedset_identifier is not None:
            return self.bedset_identifier
        
        # Compute MD5 hash
        m = md5()
        m.update(self.identifier_string.encode('utf-8'))
        computed_identifier = m.hexdigest()

        # Set bedset identifier
        self.bedset_identifier = computed_identifier
        
        return computed_identifier

    def to_grangeslist(self) -> genomicranges.GenomicRangesList:
        """Process a list of BED file identifiers and returns a GenomicRangesList object"""
        # return this bedset object
        return bedset_to_grangeslist(self.bedset_identifier)


class BBClient(BedCacheManager):
    def __init__(self, cache_folder: str, bedbase_uri: str = DEFAULT_BEDBASE_URI):
        super().__init__(cache_folder)
        self.bedbase_uri = bedbase_uri

    def download_and_process_bed_region_data(
        self, bed_identifier: str, chr_num: str, start: int, end: int
    ) -> genomicranges.GenomicRanges:
        """Download regions of a BED file from BEDbase API and return the file content as bytes"""
        bed_url = f"{self.bedbase_uri}/{bed_identifier}/regions/{chr_num}?start={start}&end={end}"
        response = requests.get(bed_url)
        response.raise_for_status()
        response_content = response.content
        gr_bed_regions = self.decompress_and_convert_to_genomic_ranges(response_content)

        return gr_bed_regions

    def download_bed_data(self, bed_identifier: str) -> bytes:
        """Download BED file from BEDbase API and return the file content as bytes"""
        bed_url = f"http://bedbase.org/api/bed/{bed_identifier}/file/bed"
        response = requests.get(bed_url)
        response.raise_for_status()

        return response.content

    def load_bedset(self, bedset_identifier: str) -> BedSet:
        """Download BEDset (List of bedfiles) from BEDbase API and return the file content as bytes"""
        bed_url = f"http://bedbase.org/api/bedset/{bedset_identifier}/bedfiles?ids=md5sum"
        response = requests.get(bed_url)
        data = response.json()
        extracted_data = [entry[0] for entry in data["data"]]
        filename = f"bedset_{bedset_identifier}.txt"
        folder_name = f"bedsets"
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, filename)
        # cache the file
        with open(file_path, "w") as file:
            for value in extracted_data:
                file.write(value + "\n")

        return BedSet(extracted_data)

    def load_bed(self, bed_file_identifier: str) -> RegionSet:
        """Loads a BED file from cachce, or downloads and caches it if it doesn't exist"""
        cached_file_path_existing = os.path.join(
            self.cache_folder, bed_file_identifier[0], bed_file_identifier[1], bed_file_identifier
        )

        if os.path.exists(cached_file_path_existing):
            print("File already exists in cache.")
        else:
            bed_data = self.download_bed_data(bed_file_identifier)
            subfolder_path = os.path.join(
                self.cache_folder, bed_file_identifier[0], bed_file_identifier[1]
            )
            self.create_cache_folder(subfolder_path=subfolder_path)
            cached_file_path = os.path.join(subfolder_path, bed_file_identifier)

            with open(cached_file_path, "wb") as f:
                f.write(bed_data)
            print("File downloaded and cached successfully.")

            return RegionSet(cached_file_path)
