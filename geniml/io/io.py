import gzip
import os
from typing import List, Union, NoReturn
import pyarrow
from ubiquerg import is_url
import logging

import numpy as np
import pandas as pd
import genomicranges
from iranges import IRanges
from hashlib import md5

from .const import (
    MAF_CENTER_COL_NAME,
    MAF_CHROMOSOME_COL_NAME,
    MAF_END_COL_NAME,
    MAF_ENTREZ_GENE_ID_COL_NAME,
    MAF_FILE_DELIM,
    MAF_HUGO_SYMBOL_COL_NAME,
    MAF_NCBI_BUILD_COL_NAME,
    MAF_START_COL_NAME,
    MAF_STRAND_COL_NAME,
)
from .utils import extract_maf_col_positions, is_gzipped, read_bedset_file
from .exceptions import BackedFileNotAvailableError, BEDFileReadError

_LOGGER = logging.getLogger("bbclient")


class Region:
    def __init__(self, chr: str, start: int, stop: int):
        """
        Instantiate a Region object.

        :param str chr: chromosome
        :param int start: start position
        :param int stop: stop position
        """
        self.chr = chr
        self.start = start
        self.end = stop

    def __repr__(self):
        return f"Region({self.chr}, {self.start}, {self.end})"


class RegionSet:
    def __init__(self, regions: Union[str, List[Region]], backed: bool = False):
        """
        Instantiate a RegionSet object. This can be backed or not backed. It represents a set of genomic regions.

        If you specify `backed` as True, then the bed file will not be loaded into memory. This is useful for large
        bed files. You can still iterate over the regions, but you cannot index into them.

        :param regions: path, or url to bed file or list of Region objects
        :param backed: whether to load the bed file into memory or not [Default: False]
        """
        # load from file
        if isinstance(regions, str):
            self.backed = backed
            self.regions: List[Region] = []
            self.path = regions

            self.regions = None
            self.is_gzipped = False

            if backed:
                if is_url(regions):
                    raise BackedFileNotAvailableError()
                # Open function depending on file type
                if not is_gzipped(regions):
                    open_func = open
                    mode = "r"
                else:
                    self.is_gzipped = True
                    open_func = gzip.open
                    mode = "rt"

                # https://stackoverflow.com/a/32607817/13175187
                try:
                    with open_func(self.path, mode) as file:
                        self.length = sum(1 for line in file if line.strip())
                except UnicodeDecodeError:
                    self.is_gzipped = True
                    with gzip.open(self.path, "rt") as file:
                        self.length = sum(1 for line in file if line.strip())

            else:
                if is_gzipped(regions):
                    df = self._read_gzipped_file(regions)
                else:
                    df = self._read_file_pd(regions, sep="\t", header=None, engine="pyarrow")

                _regions = []
                df.apply(
                    lambda row: _regions.append(Region(row[0], row[1], row[2])),
                    axis=1,
                )

                self.regions = _regions
                self.length = len(self.regions)

        # load from list
        elif isinstance(regions, list) and all([isinstance(region, Region) for region in regions]):
            self.backed = False
            self.path = None
            self.regions = regions
            self.length = len(self.regions)
        else:
            raise ValueError("regions must be a path to a bed file or a list of Region objects")

        self._identifier = None

    def _read_gzipped_file(self, file_path: str) -> pd.DataFrame:
        """
        Read a gzipped file into a pandas dataframe

        :param file_path: path to gzipped file
        :return: pandas dataframe
        """
        return self._read_file_pd(
            file_path,
            sep="\t",
            compression="gzip",
            header=None,
            engine="pyarrow",
        )

    def _read_file_pd(self, *args, **kwargs) -> pd.DataFrame:
        """
        Read bed file into a pandas DataFrame, and skip header rows if needed

        :return: pandas dataframe
        """
        max_rows = 5
        row_count = 0
        while row_count <= max_rows:
            try:
                df = pd.read_csv(*args, **kwargs, skiprows=row_count)
                if row_count > 0:
                    _LOGGER.info(f"Skipped {row_count} rows while standardization. File: '{args}'")
                df = df.dropna(axis=1)
                return df
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as _:
                if row_count <= max_rows:
                    row_count += 1
            # if can't open file after 5 attempts try to open it with gzip
        return self._read_gzipped_file(*args)

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
            # Open function depending on file type
            if self.is_gzipped:
                open_func = gzip.open
                mode = "rt"
            else:
                open_func = open
                mode = "r"

            with open_func(self.path, mode) as f:
                skipped_lines = 0
                max_skipped_lines = 5
                for line in f:

                    try:
                        chr, start, stop = line.split("\t")[:3]
                    except ValueError as _:
                        if skipped_lines < max_skipped_lines:
                            skipped_lines += 1
                            continue
                        else:
                            raise BEDFileReadError(f"Could not read line bed file")
                    if skipped_lines > 0:
                        _LOGGER.info(
                            f"Skipped {skipped_lines} lines while opening file. File: '{self.path}'"
                        )
                    yield Region(chr, int(start), int(stop))
        else:
            for region in self.regions:
                yield region

    @property
    def identifier(self) -> str:
        return self.compute_bed_identifier()

    def to_granges(self) -> genomicranges.GenomicRanges:
        """
        Return GenomicRanges contained in this BED file

        :return: GenomicRanges object
        """

        seqnames, starts, ends = zip(
            *[(region.chr, region.start, region.end) for region in self.regions]
        )
        width_list = []
        for start, end in zip(starts, ends):
            width_list.append(end - start)
        ir = IRanges(start=starts, width=width_list)

        return genomicranges.GenomicRanges(seqnames, ir)

    def compute_bed_identifier(self) -> str:
        """
        Return bed file identifier. If it is not set, compute one

        :return: the identifier of BED file (str)
        """
        if self._identifier is not None:
            return self._identifier
        else:
            if not self.backed:
                # concate column values
                chrs = ",".join([region.chr for region in self.regions])
                starts = ",".join([str(region.start) for region in self.regions])
                ends = ",".join([str(region.end) for region in self.regions])

            else:
                open_func = open if not is_gzipped(self.path) else gzip.open
                mode = "r" if not is_gzipped(self.path) else "rt"
                with open_func(self.path, mode) as f:
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

            self._identifier = bed_digest

            return self._identifier


class BedSet:
    """
    BedSet object
    """

    def __init__(
        self,
        region_sets: Union[List[RegionSet], List[str], List[List[Region]], None] = None,
        file_path: str = None,
        identifier: str = None,
    ):
        """
        :param region_sets: list of BED file paths, RegionSet, or 2-dimension list of Region [Default: None - empty BedSet]
        :param file_path: path to the .txt file with identifier of all BED files in it
        :param identifier: the identifier of the BED set
        """

        if isinstance(region_sets, list):
            # init with a list of BED files
            if all(isinstance(region_set, RegionSet) for region_set in region_sets):
                self.region_sets = region_sets
            # init with a list of file paths or a 2d list of Region
            else:
                self.region_sets = []
                for r in region_sets:
                    self.region_sets.append(RegionSet(r))

        elif file_path is not None:
            if os.path.isfile(file_path):
                self.region_sets = [RegionSet(r) for r in read_bedset_file(file_path)]
            else:
                raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
        else:
            # create empty regionSet
            self.region_sets = []

        self._bedset_identifier = identifier

    def __len__(self):
        return len(self.region_sets)

    def __iter__(self):
        for region_set in self.region_sets:
            yield region_set

    def __getitem__(self, indx: int):
        return self.region_sets[indx]

    @property
    def identifier(self) -> str:
        return self._bedset_identifier or self.compute_bedset_identifier()

    def add(self, bedfile: RegionSet) -> NoReturn:
        """
        Add a BED file to the BED set

        !Warning: if new bedfile will be added, bedSet identifier will be changed!

        :param bedfile: RegionSet instance, that should be added to the bedSet
        :return: NoReturn
        """
        self.region_sets.append(bedfile)

        self._bedset_identifier = self.compute_bedset_identifier()

    def to_granges_list(self) -> genomicranges.GenomicRangesList:
        """
        Process a list of BED set identifiers and returns a GenomicRangesList object
        """
        gr_list = []
        for regionset in self.region_sets:
            gr_list.append(regionset.to_granges())

        return genomicranges.GenomicRangesList(ranges=gr_list)

    def compute_bedset_identifier(self) -> str:
        """
        Return the identifier. If it is not set, compute one

        :param bedset: BedSet object
        :return: the identifier of BED set
        """
        if self._bedset_identifier is not None:
            return self._bedset_identifier

        elif self._bedset_identifier is None:
            bedfile_ids = []
            for bedfile in self.region_sets:
                bedfile_ids.append(bedfile.compute_bed_identifier())
            self._bedset_identifier = md5(
                ";".join(sorted(bedfile_ids)).encode("utf-8")
            ).hexdigest()

            return self._bedset_identifier


class SNP:
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


class Maf:
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
            raise ValueError("mafs must be a path to a maf file")

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
