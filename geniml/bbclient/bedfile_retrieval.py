import os
import requests
import genomicranges
from typing import List, Union
from .utils import BedDownloader, BedCacheManager


class BedFetch(BedCacheManager, BedDownloader):
    def __init__(self, cache_folder: str):
        super().__init__(cache_folder)

    def download_and_process_bed_region_data(
        self, bed_identifier: str, chr_num: str, start: int, end: int
    ) -> genomicranges.GenomicRanges:
        """Download regions of a BED file from BEDbase API and return the file content as bytes"""
        bed_url = f"http://bedbase.org/api/bed/{bed_identifier}/regions/{chr_num}?start={start}&end={end}"
        response = requests.get(bed_url)
        response.raise_for_status()
        response_content = response.content
        gr_bed_regions = self.decompress_and_convert_to_genomic_ranges(response_content)

        return gr_bed_regions

    def process_bed_file(self, input_identifier: str) -> genomicranges.GenomicRanges:
        """Process a BED file and return the file content as bytes"""
        cached_file_path_existing = os.path.join(
            self.cache_folder, input_identifier[0], input_identifier[1], input_identifier
        )

        if os.path.exists(cached_file_path_existing):
            print("File already exists in cache.")
        else:
            bed_data = self.download_bed_data(input_identifier)
            subfolder_path = os.path.join(
                self.cache_folder, input_identifier[0], input_identifier[1]
            )
            self.create_cache_folder(subfolder_path=subfolder_path)
            cached_file_path = os.path.join(subfolder_path, input_identifier)

            with open(cached_file_path, "wb") as f:
                f.write(bed_data)
            print("File downloaded and cached successfully.")

        with open(cached_file_path_existing, "rb") as f:
            bed_data = f.read()
            gr = self.decompress_and_convert_to_genomic_ranges(bed_data)

            return gr

    def read_bed_identifiers_from_file(self, file_path: str) -> List[str]:
        """Read BED identifiers from a text file"""
        bed_identifiers = []

        with open(file_path, "r") as f:
            for line in f:
                bed_identifiers.append(line.strip())

        return bed_identifiers
    
    def process_identifier(self, input_identifier: Union[str, List[str]]) -> genomicranges.GenomicRanges:
        """Process a BED file identifier or a list of BED file identifiers"""
        gr = self.process_bed_file(input_identifier)

        return gr
        
    def process_identifiers(self, input_identifier: Union[str, List[str]]) -> genomicranges.GenomicRangesList:
            """Process a list of BED file identifiers and returns a GenomicRangesList object"""
            gr_dict = {}  # Create empty dict to store GenomicRanges objects
            
            bed_identifiers = self.read_bed_identifiers_from_file(input_identifier)

            for bed_identifier in bed_identifiers:
                gr = self.process_bed_file(bed_identifier)
                gr_dict[bed_identifier] = gr
                print(f"Processed {bed_identifier}")
                print(gr)

            # Create a GenomicRangesList object from the dictionary (Which is a bedset object...)
            grl = genomicranges.GenomicRangesList(**gr_dict)
            print(grl)
            
            # return this bedset object
            return grl
        

