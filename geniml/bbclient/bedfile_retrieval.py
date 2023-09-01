import os
import gzip
import requests
import genomicranges
import pandas as pd
from io import BytesIO
from typing import List, Optional, Union

class BedDownloader:
    def download_bed_data(self, bed_identifier: str) -> bytes:
        """ Download BED file from BEDbase API and return the file content as bytes """
        bed_url = f"http://bedbase.org/api/bed/{bed_identifier}/file/bed"
        response = requests.get(bed_url)
        response.raise_for_status()
        
        return response.content
    
    def download_bedset_data(self, bed_identifier: str) -> dict:
        """ Download BEDset (List of bedfiles) from BEDbase API and return the file content as bytes """
        bed_url = f"http://bedbase.org/api/bedset/{bed_identifier}/bedfiles?ids=md5sum"
        response = requests.get(bed_url)
        data = response.json()
        extracted_data = [entry[0] for entry in data['data']] 
        filename = f"bedset_{bed_identifier}.txt"
        folder_name = f"bedsets"
        os.makedirs(folder_name, exist_ok=True) 
        file_path = os.path.join(folder_name, filename)
        
        with open(file_path, 'w') as file:
            for value in extracted_data:
                file.write(value + '\n')

        return data

class BedCacheManager:
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder
        self.create_cache_folder()

    def create_cache_folder(self, subfolder_path: Optional[str] = None) -> None:
        """ Create cache folder if it doesn't exist """
        if subfolder_path is None:
            subfolder_path = self.cache_folder
        
        full_path = os.path.join(subfolder_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    def process_local_bed_data(self, file_path: str) -> genomicranges.GenomicRanges:
        """ Process a local BED file and return the file content as bytes """
        with open(file_path, 'rb') as local_file:
            file_content = local_file.read()
            
        gr_bed_local = self.decompress_and_convert_to_genomic_ranges(file_content)
        
        return gr_bed_local

    def decompress_and_convert_to_genomic_ranges(self, content: bytes) -> genomicranges.GenomicRanges:
        """ Decompress a BED file and convert it to a GenomicRanges object """
        is_gzipped = content[:2] == b'\x1f\x8b'

        if is_gzipped:
            with gzip.GzipFile(fileobj=BytesIO(content), mode='rb') as f:
                df = pd.read_csv(f, sep='\t', header=None, engine='pyarrow')
        else:
            df = pd.read_csv(BytesIO(content), sep='\t', header=None, engine='pyarrow')

        header = ['seqnames', 'starts', 'ends', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount']
        df.columns = header[:len(df.columns)]
        gr = genomicranges.from_pandas(df)

        return gr

class BedProcessor(BedCacheManager, BedDownloader):
    def __init__(self, cache_folder: str):
        super().__init__(cache_folder)

    def download_and_process_bed_region_data(self, bed_identifier: str, chr_num: str, start: int, end: int) -> genomicranges.GenomicRanges:
        """ Download regions of a BED file from BEDbase API and return the file content as bytes """
        bed_url = f"http://bedbase.org/api/bed/{bed_identifier}/regions/{chr_num}?start={start}&end={end}"
        response = requests.get(bed_url)
        response.raise_for_status()
        response_content = response.content
        gr_bed_regions = self.decompress_and_convert_to_genomic_ranges(response_content)

        return gr_bed_regions
    
    def process_bed_file(self, input_identifier: str) -> genomicranges.GenomicRanges:
        """ Process a BED file and return the file content as bytes """
        cached_file_path_existing = os.path.join(self.cache_folder, input_identifier[0], input_identifier[1], input_identifier)

        if os.path.exists(cached_file_path_existing):
            print("File already exists in cache.")
        else:
            bed_data = self.download_bed_data(input_identifier)
            subfolder_path = os.path.join(self.cache_folder, input_identifier[0], input_identifier[1])
            self.create_cache_folder(subfolder_path=subfolder_path)
            cached_file_path = os.path.join(subfolder_path, input_identifier)

            with open(cached_file_path, 'wb') as f:
                f.write(bed_data)
            print("File downloaded and cached successfully.")

        with open(cached_file_path_existing, 'rb') as f:
            bed_data = f.read()
            gr = self.decompress_and_convert_to_genomic_ranges(bed_data)

            return gr

    def read_bed_identifiers_from_file(self, file_path: str) -> List[str]:
        """ Read BED identifiers from a text file """
        bed_identifiers = []

        with open(file_path, 'r') as f:
            for line in f:
                bed_identifiers.append(line.strip())

        return bed_identifiers

    def process_identifiers(self, input_identifier: Union[str, List[str]]) -> List[genomicranges.GenomicRanges]:
        """ Process a BED file identifier or a list of BED file identifiers """
        if os.path.isfile(input_identifier):
            bed_identifiers = self.read_bed_identifiers_from_file(input_identifier)
            results_list = []
            for bed_identifier in bed_identifiers:
                gr = self.process_bed_file(bed_identifier)
                results_list.append(gr)
                print(f"Processed {bed_identifier}")
                print(gr)

            return results_list
        else:
            gr = self.process_bed_file(input_identifier)

            return [gr]
