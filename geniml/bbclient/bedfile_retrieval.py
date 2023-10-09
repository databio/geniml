import os
from typing import List, Union

import genomicranges
import requests

from ..io import Region, RegionSet
from .const import DEFAULT_BEDBASE_URI
from .utils import (BedCacheManager, BedFile, BedSet, bedset_to_grangeslist,
                    read_bedset_file)


class BBClient(BedCacheManager):
    def __init__(self, cache_folder: str, bedbase_api: str = DEFAULT_BEDBASE_URI):
        super().__init__(cache_folder)
        self.bedbase_uri = bedbase_api

    def download_and_process_bed_region_data(
        self, bed_identifier: str, chr_num: str, start: int, end: int
    ) -> genomicranges.GenomicRanges:
        """Download regions of a BED file from BEDbase API and return the file content as bytes"""
        bed_url = f"{self.bedbase_uri}/{bed_identifier}/regions/{chr_num}?start={start}&end={end}"
        response = requests.get(bed_url)
        response.raise_for_status()
        response_content = response.content
        gr_bed_regions = self.decompress_and_convert_to_genomic_ranges(response_content)

        return gr_bed_regions

    def download_bed_data(self, bed_identifier: str) -> bytes:
        """Download BED file from BEDbase API and return the file content as bytes"""
        bed_url = f"http://bedbase.org/api/bed/{bed_identifier}/file/bed"
        print(bed_url)
        response = requests.get(bed_url)
        response.raise_for_status()

        return response.content

    def load_bedset(self, bedset_identifier: str) -> BedSet:
        """Download BEDset (List of bedfiles) from BEDbase API and return the file content as BedSet"""
        bed_url = f"http://bedbase.org/api/bedset/{bedset_identifier}/bedfiles?ids=md5sum"
        response = requests.get(bed_url)
        data = response.json()
        extracted_data = [entry[0] for entry in data["data"]]
        filename = f"bedset_{bedset_identifier}.txt"
        folder_name = os.path.join(
            self.cache_folder, "bedsets", bedset_identifier[0], bedset_identifier[1]
        )
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, filename)
        print(file_path)
        # cache the file
        with open(file_path, "w") as file:
            for value in extracted_data:
                file.write(value + "\n")

        return BedSet(
            [self.load_bed(bed_file_id) for bed_file_id in extracted_data],
            identifier=bedset_identifier,
        )

    # def load_bed(self, bed_file_identifier: str) -> RegionSet:
    def load_bed(self, bed_file_identifier: str) -> BedFile:
        """Loads a BED file from cachce, or downloads and caches it if it doesn't exist"""
        cached_file_path_existing = os.path.join(
            self.cache_folder, bed_file_identifier[0], bed_file_identifier[1], bed_file_identifier
        )

        if os.path.exists(cached_file_path_existing):
            print("File already exists in cache.")
        else:
            bed_data = self.download_bed_data(bed_file_identifier)
            subfolder_path = os.path.join(
                self.cache_folder, "bedfiles", bed_file_identifier[0], bed_file_identifier[1]
            )
            self.create_cache_folder(subfolder_path=subfolder_path)
            cached_file_path = os.path.join(subfolder_path, f"{bed_file_identifier}.bed.gz")

            with open(cached_file_path, "wb") as f:
                f.write(bed_data)
            print("File downloaded and cached successfully.")

            return BedFile(regions=cached_file_path, identifier=bed_file_identifier)
