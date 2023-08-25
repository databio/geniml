import os
from typing import List, Union

import gzip
from intervaltree import Interval

from .utils import extract_maf_col_positions, is_gzipped
from .const import *
from ..utils import *


class Region(Interval):
    def __new__(cls, chr: str, start: int, stop: int, data=None):
        return super(Region, cls).__new__(cls, start, stop, data)

    def __init__(self, chr: str, start: int, stop: int, data=None):
        # no need to call super().__init__() because namedtuple doesn't have __init__()
        self.chr = chr

    @property
    def start(self):
        return self.begin

    def __repr__(self):
        return f"Region({self.chr}, {self.start}, {self.end})"


class RegionSet(object):
    def __init__(self, regions: Union[str, List[Region]], backed: bool = False):
        # load from file
        if isinstance(regions, str):
            self.backed = backed
            self.regions: List[Region] = []
            self.path = regions

            # Open function depending on file type
            open_func = gzip.open if is_gzipped(regions) else open
            mode = "rt" if is_gzipped else "r"

            if backed:
                self.regions = None
                # https://stackoverflow.com/a/32607817/13175187
                with open_func(self.path, mode) as file:
                    self.length = sum(1 for line in file if line.strip())
            else:
                with open_func(regions, mode) as f:
                    lines = f.readlines()
                    for line in lines:
                        # some bed files have more than 3 columns, so we just take the first 3
                        chr, start, stop = line.split("\t")[:3]
                        self.regions.append(Region(chr, int(start), int(stop)))
                    self.length = len(self.regions)

        # load from list
        elif isinstance(regions, list) and all([isinstance(region, Region) for region in regions]):
            self.backed = False
            self.path = None
            self.regions = regions
            self.length = len(self.regions)
        else:
            raise ValueError(f"regions must be a path to a bed file or a list of Region objects")

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if self.backed:
            raise NotImplementedError("Backed RegionSets do not currently support indexing.")
        else:
            return self.regions[key]

    def __repr__(self):
        if self.path:
            if self.backed:
                return f"RegionSet({self.path}, backed=True)"
            else:
                return f"RegionSet({self.path})"
        else:
            return f"RegionSet(n={self.length})"

    def __iter__(self):
        if self.backed:
            # Check if the file is gzipped
            _, file_extension = os.path.splitext(self.path)
            is_gzipped = file_extension == ".gz"

            # Open function depending on file type
            open_func = gzip.open if is_gzipped else open
            mode = "rt" if is_gzipped else "r"

            with open_func(self.path, mode) as f:
                for line in f:
                    chr, start, stop = line.split("\t")[:3]
                    yield Region(chr, int(start), int(stop))
        else:
            for region in self.regions:
                yield region


class SNP(object):
    """
    Python representation of a SNP
    """

    def __init__(
        self,
        hugo_symbol: str = None,
        entrez_gene_id: str = None,
        center: str = None,
        ncbi_build: str = None,
        chromosome: str = None,
        start_position: int = None,
        end_position: int = None,
        strand: str = None,
    ):
        self.hugo_symbol = hugo_symbol
        self.entrez_gene_id = entrez_gene_id
        self.center = center
        self.ncbi_build = ncbi_build
        self.chromosome = chromosome
        self.start_position = start_position
        self.end_position = end_position
        self.strand = strand

    def __repr__(self):
        return f"SNP({self.chromosome}, {self.start_position}, {self.end_position}, {self.strand})"


class Maf(object):
    """
    Python representation of a MAF file, only supports some columns for now
    """

    def __init__(self, maf_file: str, backed: bool = False, bump_end_position: bool = False):
        """
        :param maf_file: path to maf file
        :param backed: whether to load the maf file into memory or not
        :param bump_end_position: whether to bump the end position by 1 or not (this is useful for interval trees and interval lists)
        """
        # load from file
        if isinstance(maf_file, str):
            self.maf_file = maf_file
            self.col_positions = extract_maf_col_positions(maf_file)
            self.backed = backed
            self.mafs: List[SNP] = []

            # Open function depending on file type
            open_func = gzip.open if is_gzipped(maf_file) else open
            mode = "rt" if is_gzipped(maf_file) else "r"

            if backed:
                self.mafs = None
                # https://stackoverflow.com/a/32607817/13175187
                with open_func(self.maf_file, mode) as file:
                    self.length = sum(1 for line in file if line.strip())
            else:
                with open_func(maf_file, mode) as f:
                    lines = f.readlines()
                    for line in lines:
                        # some bed files have more than 3 columns, so we just take the first 3
                        line = line.strip().split(MAF_FILE_DELIM)
                        self.mafs.append(
                            SNP(
                                hugo_symbol=line[self.col_positions[MAF_HUGO_SYMBOL_COL_NAME]],
                                entrez_gene_id=line[
                                    self.col_positions[MAF_ENTREZ_GENE_ID_COL_NAME]
                                ],
                                center=line[self.col_positions[MAF_CENTER_COL_NAME]],
                                ncbi_build=line[self.col_positions[MAF_NCBI_BUILD_COL_NAME]],
                                chromosome=line[self.col_positions[MAF_CHROMOSOME_COL_NAME]],
                                start_position=line[self.col_positions[MAF_START_COL_NAME]],
                                end_position=line[self.col_positions[MAF_END_COL_NAME]],
                                strand=line[self.col_positions[MAF_STRAND_COL_NAME]],
                            )
                        )
                    self.length = len(self.mafs)
        else:
            raise ValueError(f"mafs must be a path to a maf file")

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if self.backed:
            raise NotImplementedError("Backed MAFs do not currently support indexing.")
        else:
            return self.mafs[key]

    def __iter__(self):
        if self.backed:
            # Open function depending on file type
            open_func = gzip.open if is_gzipped(self.maf_file) else open
            mode = "rt" if is_gzipped(self.maf_file) else "r"

            with open_func(self.maf_file, mode) as f:
                for line in f:
                    line = line.strip().split(MAF_FILE_DELIM)
                    yield SNP(
                        hugo_symbol=line[self.col_positions[MAF_HUGO_SYMBOL_COL_NAME]],
                        entrez_gene_id=line[self.col_positions[MAF_ENTREZ_GENE_ID_COL_NAME]],
                        center=line[self.col_positions[MAF_CENTER_COL_NAME]],
                        ncbi_build=line[self.col_positions[MAF_NCBI_BUILD_COL_NAME]],
                        chromosome=line[self.col_positions[MAF_CHROMOSOME_COL_NAME]],
                        start_position=line[self.col_positions[MAF_START_COL_NAME]],
                        end_position=line[self.col_positions[MAF_END_COL_NAME]],
                        strand=line[self.col_positions[MAF_STRAND_COL_NAME]],
                    )
        else:
            for maf in self.mafs:
                yield maf


# TODO: This belongs somewhere else; does it even make sense?
class TokenizedRegionSet(object):
    """Represents a tokenized region set"""

    def __init__(self, tokens: np.ndarray, universe: RegionSet):
        self.tokens = tokens
        self.universe = universe


# Write a class representing a collection of RegionSets
# TODO: This shouldn't read in the actual files, it should just represent the files and use lazy loading
class RegionSetCollection(object):
    """Represents a collection of RegionSets"""

    def __init__(self, region_sets: List[RegionSet] = None, file_globs: List[str] = None):
        if region_sets:
            self.region_sets = region_sets
        elif file_globs:
            self.region_sets = []
            for glob in file_globs:
                self.region_sets.extend([RegionSet(path) for path in glob.glob(glob)])

    def __getitem__(self, key):
        return self.region_sets[key]

    def __len__(self):
        return len(self.region_sets)


# Do we need an EmbeddingSet class?
class EmbeddingSet(object):
    """Represents embeddings and labels"""

    embeddings: np.ndarray
    labels: list
