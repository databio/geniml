from typing import Dict
import os

from .const import *


def is_gzipped(file: str) -> bool:
    """
    Check if a file is gzipped.

    :param file: path to file
    :return: True if file is gzipped, else False
    """
    _, file_extension = os.path.splitext(file)
    return file_extension == ".gz"


def extract_maf_col_positions(file: str) -> Dict[MAF_COLUMN, int]:
    """
    Extract the column positions of the MAF file.

    :param file: path to .maf file
    :return: dictionary of column positions
    """
    with open(file, "r") as f:
        header = f.readline().strip().split(MAF_FILE_DELIM)
        col_positions = {
            MAF_HUGO_SYMBOL_COL_NAME: header.index(MAF_HUGO_SYMBOL_COL_NAME),
            MAF_ENTREZ_GENE_ID_COL_NAME: header.index(MAF_ENTREZ_GENE_ID_COL_NAME),
            MAF_CENTER_COL_NAME: header.index(MAF_CENTER_COL_NAME),
            MAF_NCBI_BUILD_COL_NAME: header.index(MAF_NCBI_BUILD_COL_NAME),
            MAF_CHROMOSOME_COL_NAME: header.index(MAF_CHROMOSOME_COL_NAME),
            MAF_START_COL_NAME: header.index(MAF_START_COL_NAME),
            MAF_END_COL_NAME: header.index(MAF_END_COL_NAME),
            MAF_STRAND_COL_NAME: header.index(MAF_STRAND_COL_NAME),
        }
    return col_positions
