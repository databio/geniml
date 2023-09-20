import gzip
import os
from typing import Dict, List, Union

from .const import *


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
