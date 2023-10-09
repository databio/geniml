import gzip
import os
from io import BytesIO
from typing import List, Optional, Union

import genomicranges
import pandas as pd
import requests

from ..io import Region, RegionSet


class BedFile(RegionSet):
    """
    RegionSet with identifier and local path
    """

    def __init__(self, regions: str, backed: bool = False, identifier: str = None):
        super().__init__(regions, backed)
        self.path = regions
        self.identifier = identifier

    def to_granges(self):
        gr_dict = {}
        seqnames = []
        starts = []
        ends = []
        for region in self.regions:
            seqnames.append(region.chr)
            starts.append(region.start)
            ends.append(region.end)

        gr_dict["seqnames"] = seqnames
        gr_dict["starts"] = starts
        gr_dict["ends"] = ends

        return genomicranges.GenomicRanges(gr_dict)


class BedSet(object):
    """
    Storing BedFile
    """

    def __init__(
        self,
        # region_sets: Union[List[RegionSet], List[str], List[List[Region]], None],\
        region_sets=List[BedFile],
        file_path: str = None,
        identifier: str = None,
    ):
        if isinstance(region_sets, list) and all(
            isinstance(region_set, BedFile) for region_set in region_sets
        ):
            self.region_sets = region_sets

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
        m.update(self.identifier_string.encode("utf-8"))
        computed_identifier = m.hexdigest()

        # Set bedset identifier
        self.bedset_identifier = computed_identifier

        return computed_identifier

        # raise NotImplementedError("BedSet object does not have a bedset identifier")

    def to_grangeslist(self) -> genomicranges.GenomicRangesList:
        """Process a list of BED file identifiers and returns a GenomicRangesList object"""
        gr_list = []
        for regionset in self.region_sets:
            gr_list.append(regionset.to_granges())

        return genomicranges.GenomicRangesList(ranges=gr_list)


class BedCacheManager:
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder
        self.create_cache_folder()

    def create_cache_folder(self, subfolder_path: Optional[str] = None) -> None:
        """Create cache folder if it doesn't exist"""
        if subfolder_path is None:
            subfolder_path = self.cache_folder

        full_path = os.path.abspath(subfolder_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    @staticmethod
    def process_local_bed_data(file_path: str) -> genomicranges.GenomicRanges:
        """Process a local BED file and return the file content as bytes"""
        with open(file_path, "rb") as local_file:
            file_content = local_file.read()

        gr_bed_local = BedCacheManager.decompress_and_convert_to_genomic_ranges(file_content)

        return gr_bed_local

    @staticmethod
    def decompress_and_convert_to_genomic_ranges(content: bytes) -> genomicranges.GenomicRanges:
        """Decompress a BED file and convert it to a GenomicRanges object"""
        is_gzipped = content[:2] == b"\x1f\x8b"

        if is_gzipped:
            with gzip.GzipFile(fileobj=BytesIO(content), mode="rb") as f:
                df = pd.read_csv(f, sep="\t", header=None, engine="pyarrow")
        else:
            df = pd.read_csv(BytesIO(content), sep="\t", header=None, engine="pyarrow")

        header = [
            "seqnames",
            "starts",
            "ends",
            "name",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "blockCount",
        ]
        df.columns = header[: len(df.columns)]
        gr = genomicranges.from_pandas(df)

        return gr


@classmethod
def bedset_to_grangeslist(
    cls, bedset: BedSet, bedset_identifier: str
) -> genomicranges.GenomicRangesList:
    """Convert a bedset into a GenomicRangesList object"""
    gr_dict = {}  # Create empty dict to store GenomicRanges objects

    bed_identifiers = cls.read_bed_identifiers_from_file(bedset_identifier)

    for bed_identifier in bed_identifiers:
        gr = cls.process_bed_file(bed_identifier)
        gr_dict[bed_identifier] = gr
        print(f"Processed {bed_identifier}")
        print(gr)

    # Create a GenomicRangesList object from the dictionary
    grl = genomicranges.GenomicRangesList(**gr_dict)
    return grl


# QUESTION: should this move to the RegionSet object?
@staticmethod
def regionset_to_granges(regionset: RegionSet) -> genomicranges.GenomicRanges:
    """Convert a regionset into a GenomicRanges object"""
    with open(regionset.path, "rb") as f:
        bed_data = f.read()
        gr = BedCacheManager.decompress_and_convert_to_genomic_ranges(bed_data)

    return gr


def read_bedset_file(file_path: str) -> List[str]:
    """Load a bedset from a text file"""
    bed_identifiers = []

    with open(file_path, "r") as f:
        for line in f:
            bed_identifiers.append(line.strip())
    return bed_identifiers
