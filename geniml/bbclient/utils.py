import gzip
import requests
import genomicranges
import pandas as pd
from io import BytesIO
from typing import Optional
import os


class BedDownloader:
    def download_bed_data(self, bed_identifier: str) -> bytes:
        """Download BED file from BEDbase API and return the file content as bytes"""
        bed_url = f"http://bedbase.org/api/bed/{bed_identifier}/file/bed"
        response = requests.get(bed_url)
        response.raise_for_status()

        return response.content

    def download_bedset_data(self, bed_identifier: str) -> dict:
        """Download BEDset (List of bedfiles) from BEDbase API and return the file content as bytes"""
        bed_url = f"http://bedbase.org/api/bedset/{bed_identifier}/bedfiles?ids=md5sum"
        response = requests.get(bed_url)
        data = response.json()
        extracted_data = [entry[0] for entry in data["data"]]
        filename = f"bedset_{bed_identifier}.txt"
        folder_name = f"bedsets"
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, filename)

        with open(file_path, "w") as file:
            for value in extracted_data:
                file.write(value + "\n")

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

    def process_local_bed_data(self, file_path: str) -> genomicranges.GenomicRanges:
        """Process a local BED file and return the file content as bytes"""
        with open(file_path, "rb") as local_file:
            file_content = local_file.read()

        gr_bed_local = self.decompress_and_convert_to_genomic_ranges(file_content)

        return gr_bed_local

    def decompress_and_convert_to_genomic_ranges(
        self, content: bytes
    ) -> genomicranges.GenomicRanges:
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