import gzip
import os
from io import BytesIO
from typing import Optional
from pathlib import Path

import genomicranges
import pandas as pd

from .const import DEFAULT_CACHE_FOLDER


class BedCacheManager:
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder
        self.create_cache_folder()

    def create_cache_folder(self, subfolder_path: Optional[str] = None) -> None:
        """
        Create cache folder if it doesn't exist

        :param subfolder_path: path to the subfolder
        """
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


def get_abs_path(path: str = DEFAULT_CACHE_FOLDER, create_folder: bool = True) -> str:
    """
    Get absolute path to the folder and create it if it doesn't exist

    :param path: path to the folder
    :param create_folder: create folder if it doesn't exist

    :return: absolute path to the folder
    """
    absolute_cache_folder = os.path.expandvars(path)
    Path(absolute_cache_folder).mkdir(parents=True, exist_ok=True)
    return absolute_cache_folder
