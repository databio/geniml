import gzip
import os
from hashlib import md5
from io import BytesIO
from typing import List, Optional, Union

import genomicranges
import pandas as pd

from ..io import Region, RegionSet, is_gzipped


class BedFile(RegionSet):
    """
    RegionSet with identifier
    """

    def __init__(
        self, regions: Union[str, List[Region]], identifier: str = None, backed: bool = False
    ):
        """
        Inherit from class RegionSet in geniml/io/io.py.

        :param regions: path to bed file or list of Region objects
        :param identifier: a unique identifier of the RegionSet, can computed later
        :param backed: whether to load the bed file into memory or not
        """
        super().__init__(regions, backed)
        self.identifier = identifier

    def to_granges(self) -> genomicranges.GenomicRanges:
        """
        Return GenomicRanges contained in this BED file
        """
        seqnames, starts, ends = zip(
            *[(region.chr, region.start, region.end) for region in self.regions]
        )
        gr_dict = {"seqnames": seqnames, "starts": starts, "ends": ends}
        return genomicranges.GenomicRanges(gr_dict)


class BedSet(object):
    """
    Storing BedFile
    """

    def __init__(
        self,
        region_sets: Union[
            List[BedFile], List[str], List[List[Region]], None
        ],  # region_sets=Union[List[BedFile], List[str]],
        file_path: str = None,
        identifier: str = None,
    ):
        """
        :param region_sets: list of BED file paths, BedFile, or 2-dimension list of Region
        :param file_path: path to the .txt file with identifier of all BED files in it
        :param identifier: the identifier of the BED set
        """

        if isinstance(region_sets, list):
            # init with a list of BED files
            if all(isinstance(region_set, BedFile) for region_set in region_sets):
                self.region_sets = region_sets
            # init with a list of file paths or a 2d list of Region
            else:
                self.region_sets = []
                for r in region_sets:
                    self.region_sets.append(BedFile(r))

        elif file_path is not None:
            if os.path.isfile(file_path):
                self.region_sets = [RegionSet(r) for r in read_bedset_file(file_path)]
            else:
                raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

            for r in read_bedset_file(region_sets):
                self.region_sets.append(RegionSet(r))

        self.bedset_identifier = identifier

    def __len__(self):
        return len(self.region_sets)

    def __iter__(self):
        for region_set in self.region_sets:
            yield region_set

    def __getitem__(self, indx: int):
        return self.region_sets[indx]

    def to_grangeslist(self) -> genomicranges.GenomicRangesList:
        """
        Process a list of BED set identifiers and returns a GenomicRangesList object
        """
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


def compute_bed_identifier(bedfile: BedFile):
    """
    Return the identifier. If it is not set, compute one

    from digest_bedfile in bedboss/bedstat/bedstat.py
    https://github.com/databio/bedboss/blob/main/bedboss/bedstat/bedstat.py
    """
    if bedfile.identifier is not None:
        return bedfile.identifier
    else:
        if not bedfile.backed:
            # concate column values
            chrs = ",".join([region.chr for region in bedfile.regions])
            starts = ",".join([str(region.start) for region in bedfile.regions])
            ends = ",".join([str(region.end) for region in bedfile.regions])

        else:
            open_func = open if not is_gzipped(bedfile.path) else gzip.open
            mode = "r" if not is_gzipped(bedfile.path) else "rt"
            with open_func(bedfile.path, mode) as f:
                # concate column values
                chrs = []
                starts = []
                ends = []
                for row in f:
                    chrs.append(row.split("\t")[0])
                    starts.append(row.split("\t")[1])
                    ends.append(row.split("\t")[2].replace("\n", ""))
                chrs = ",".join(chrs)
                starts = ",".join(starts)
                ends = ",".join(ends)

        # hash column values
        chr_digest = md5(chrs.encode("utf-8")).hexdigest()
        start_digest = md5(starts.encode("utf-8")).hexdigest()
        end_digest = md5(ends.encode("utf-8")).hexdigest()
        # hash column digests
        bed_digest = md5(
            ",".join([chr_digest, start_digest, end_digest]).encode("utf-8")
        ).hexdigest()

        bedfile.identifier = bed_digest

        return bedfile.identifier


def compute_bedset_identifier(bedset: BedSet) -> str:
    """
    Return the identifier. If it is not set, compute one

    :return: the identifier of BED set
    """
    if bedset.bedset_identifier is not None:
        return bedset.bedset_identifier
    if bedset.bedset_identifier is None:
        # based on get_bedset_digest() in bedbuncher/pipelines/bedbuncher.py
        # https://github.com/databio/bedbuncher/blob/master/pipelines/bedbuncher.py

        bedfile_ids = []
        for bedfile in bedset.region_sets:
            bedfile_ids.append(compute_bed_identifier(bedfile))
        bedset.bedset_identifier = md5(";".join(sorted(bedfile_ids)).encode("utf-8")).hexdigest()

        return bedset.bedset_identifier


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
