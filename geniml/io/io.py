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
        """
        Instantiate a Region object. This is a wrapper around the Interval class from the intervaltree package.
        This makes it easier to work with regions.
        """
        # no need to call super().__init__() because namedtuple doesn't have __init__()
        self.chr = chr

    @property
    def start(self):
        return self.begin

    def __repr__(self):
        return f"Region({self.chr}, {self.start}, {self.end})"


class RegionSet(object):
    def __init__(self, regions: Union[str, List[Region]], backed: bool = False):
        """
        Instantiate a RegionSet object. This can be backed or not backed. It represents a set of genomic regions.

        If you specify `backed` as True, then the bed file will not be loaded into memory. This is useful for large
        bed files. You can still iterate over the regions, but you cannot index into them.

        :param regions: path to bed file or list of Region objects
        :param backed: whether to load the bed file into memory or not
        """
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


class BedSet(object):
    def __init__(self, sets: Union[List[RegionSet], List[str]]):
        pass


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

    @property
    def start(self):
        return self.start_position

    @property
    def end(self):
        return self.end_position

    @property
    def chr(self):
        return self.chromosome

    def to_region(self):
        chr = self.chromosome
        start = int(self.start_position)
        end = int(self.end_position)
        # bump end position by 1 if needed
        if start == end:
            end += 1

        return Region(chr, start, end)

    def __len__(self):
        return self.end - self.start

    def __repr__(self):
        return f"SNP({self.chromosome}, {self.start_position}, {self.end_position}, {self.strand})"


class Maf(object):
    """
    Python representation of a MAF file, only supports some columns for now
    """

    def _extract_value_from_col(self, col_name: str, line: str) -> any:
        """
        Extract a value from a column in a line of a MAF file.

        :param col_name: name of column
        :param line: line from MAF file
        :return: value of column
        """
        return line[self.col_positions[col_name]] if self.col_positions[col_name] else None

    def __init__(
        self,
        maf_file: str,
        backed: bool = False,
        bump_end_position: bool = False,
        chr_rep_as_int: bool = False,
    ):
        """
        :param maf_file: path to maf file
        :param backed: whether to load the maf file into memory or not
        :param bump_end_position: whether to bump the end position by 1 or not (this is useful for interval trees and interval lists)
        :param chr_rep_as_int: whether to represent the chromosome as an int or not (this is useful for interval trees and interval lists)
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
                    self.length = (
                        sum(1 for line in file if line.strip()) - 1
                    )  # subtract 1 for header
            else:
                with open_func(maf_file, mode) as f:
                    # skip header
                    lines = f.readlines()[1:]
                    for line in lines:
                        # some bed files have more than 3 columns, so we just take the first 3
                        line = line.strip().split(MAF_FILE_DELIM)
                        self.mafs.append(
                            SNP(
                                hugo_symbol=self._extract_value_from_col(
                                    MAF_HUGO_SYMBOL_COL_NAME, line
                                ),
                                entrez_gene_id=self._extract_value_from_col(
                                    MAF_ENTREZ_GENE_ID_COL_NAME, line
                                ),
                                center=self._extract_value_from_col(MAF_CENTER_COL_NAME, line),
                                ncbi_build=self._extract_value_from_col(
                                    MAF_NCBI_BUILD_COL_NAME, line
                                ),
                                chromosome=self._extract_value_from_col(
                                    MAF_CHROMOSOME_COL_NAME, line
                                ),
                                start_position=int(
                                    self._extract_value_from_col(MAF_START_COL_NAME, line)
                                ),
                                end_position=int(
                                    self._extract_value_from_col(MAF_END_COL_NAME, line)
                                ),
                                strand=self._extract_value_from_col(MAF_STRAND_COL_NAME, line),
                            )
                        )
                    self.length = len(self.mafs)

                # post process according to flags
                for maf in self.mafs:
                    if bump_end_position:
                        maf.end_position += 1
                    if not chr_rep_as_int:
                        maf.chromosome = "chr" + str(maf.chromosome)
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
                # skip header
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    line = line.strip().split(MAF_FILE_DELIM)
                    yield SNP(
                        hugo_symbol=self._extract_value_from_col(MAF_HUGO_SYMBOL_COL_NAME, line),
                        entrez_gene_id=self._extract_value_from_col(
                            MAF_ENTREZ_GENE_ID_COL_NAME, line
                        ),
                        center=self._extract_value_from_col(MAF_CENTER_COL_NAME, line),
                        ncbi_build=self._extract_value_from_col(MAF_NCBI_BUILD_COL_NAME, line),
                        chromosome=self._extract_value_from_col(MAF_CHROMOSOME_COL_NAME, line),
                        start_position=self._extract_value_from_col(MAF_START_COL_NAME, line),
                        end_position=self._extract_value_from_col(MAF_END_COL_NAME, line),
                        strand=self._extract_value_from_col(MAF_STRAND_COL_NAME, line),
                    )

        else:
            for maf in self.mafs:
                yield maf

    def __repr__(self):
        return f"MAF({self.maf_file})"


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
