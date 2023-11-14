import gzip
import os
from io import BytesIO
from typing import Optional
import urllib.parse

import genomicranges
import pandas as pd


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


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid

    :param url: URL to be checked
    :return: True if the URL is valid, False otherwise
    """
    try:
        if urllib.parse.urlparse(url).hostname:
            return True
        return False
    except Exception:
        return False
