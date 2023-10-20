import gzip
import os
from typing import Dict, List, Union, Any


from .const import *

# from .io import BedSet, RegionSet


def is_gzipped(file: str) -> bool:
    """
    Check if a file is gzipped.

    :param file: path to file
    :return: True if file is gzipped, else False
    """
    _, file_extension = os.path.splitext(file)
    return file_extension == ".gz"


def extract_maf_col_positions(file: str) -> Dict[MAF_COLUMN, Union[int, None]]:
    """
    Extract the column positions of the MAF file.

    :param file: path to .maf file
    :return: dictionary of column positions
    """

    def get_index_from_header(header: List[str], col_name: str) -> int:
        """
        Get the index of a column from a header.

        :param header: list of column names
        :param col_name: column name
        :return: index of column
        """
        try:
            return header.index(col_name)
        except ValueError:
            return None

    # detect open function
    open_func = open if not is_gzipped(file) else gzip.open
    mode = "r" if not is_gzipped(file) else "rt"

    with open_func(file, mode) as f:
        header = f.readline().strip().split(MAF_FILE_DELIM)
        col_positions = {
            MAF_HUGO_SYMBOL_COL_NAME: get_index_from_header(header, MAF_HUGO_SYMBOL_COL_NAME),
            MAF_ENTREZ_GENE_ID_COL_NAME: get_index_from_header(
                header, MAF_ENTREZ_GENE_ID_COL_NAME
            ),
            MAF_CENTER_COL_NAME: get_index_from_header(header, MAF_CENTER_COL_NAME),
            MAF_NCBI_BUILD_COL_NAME: get_index_from_header(header, MAF_NCBI_BUILD_COL_NAME),
            MAF_CHROMOSOME_COL_NAME: get_index_from_header(header, MAF_CHROMOSOME_COL_NAME),
            MAF_START_COL_NAME: get_index_from_header(header, MAF_START_COL_NAME),
            MAF_END_COL_NAME: get_index_from_header(header, MAF_END_COL_NAME),
            MAF_STRAND_COL_NAME: get_index_from_header(header, MAF_STRAND_COL_NAME),
        }
    return col_positions


def read_bedset_file(file_path: str) -> List[str]:
    """Load a bedset from a text file"""
    bed_identifiers = []

    with open(file_path, "r") as f:
        for line in f:
            bed_identifiers.append(line.strip())
    return bed_identifiers


# def compute_bed_identifier(bedfile: Any) -> str:
#     """
#     Return bed file identifier. If it is not set, compute one
#
#     :param bedfile: RegionSet object (Representation of bed_file)
#     :return: the identifier of BED file (str)
#     """
#     if bedfile.identifier is not None:
#         return bedfile.identifier
#     else:
#         if not bedfile.backed:
#             # concate column values
#             chrs = ",".join([region.chr for region in bedfile.regions])
#             starts = ",".join([str(region.start) for region in bedfile.regions])
#             ends = ",".join([str(region.end) for region in bedfile.regions])
#
#         else:
#             open_func = open if not is_gzipped(bedfile.path) else gzip.open
#             mode = "r" if not is_gzipped(bedfile.path) else "rt"
#             with open_func(bedfile.path, mode) as f:
#                 # concate column values
#                 chrs = []
#                 starts = []
#                 ends = []
#                 for row in f:
#                     chrs.append(row.split("\t")[0])
#                     starts.append(row.split("\t")[1])
#                     ends.append(row.split("\t")[2].replace("\n", ""))
#                 chrs = ",".join(chrs)
#                 starts = ",".join(starts)
#                 ends = ",".join(ends)
#
#         # hash column values
#         chr_digest = md5(chrs.encode("utf-8")).hexdigest()
#         start_digest = md5(starts.encode("utf-8")).hexdigest()
#         end_digest = md5(ends.encode("utf-8")).hexdigest()
#         # hash column digests
#         bed_digest = md5(
#             ",".join([chr_digest, start_digest, end_digest]).encode("utf-8")
#         ).hexdigest()
#
#         bedfile.identifier = bed_digest
#
#         return bedfile.identifier


# def compute_bedset_identifier(bedset: Any) -> str:
#     """
#     Return the identifier. If it is not set, compute one
#
#     :param bedset: BedSet object
#     :return: the identifier of BED set
#     """
#     if bedset.bedset_identifier is not None:
#         return bedset.bedset_identifier
#
#     elif bedset.bedset_identifier is None:
#         bedfile_ids = []
#         for bedfile in bedset.region_sets:
#             bedfile_ids.append(compute_bed_identifier(bedfile))
#         bedset.bedset_identifier = md5(";".join(sorted(bedfile_ids)).encode("utf-8")).hexdigest()
#
#         return bedset.bedset_identifier
