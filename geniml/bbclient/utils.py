import gzip
import requests
import genomicranges
import pandas as pd
from io import BytesIO
from typing import Optional
import os



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
    

def bedset_to_grangeslist(bedset: BedSet) -> genomicranges.GenomicRangesList:
        """Convert a bedset into a GenomicRangesList object"""
        gr_dict = {}  # Create empty dict to store GenomicRanges objects
        
        bed_identifiers = self.read_bed_identifiers_from_file(bedset_identifier)

        for bed_identifier in bed_identifiers:
            gr = self.process_bed_file(bed_identifier)
            gr_dict[bed_identifier] = gr
            print(f"Processed {bed_identifier}")
            print(gr)

            # Create a GenomicRangesList object from the dictionary
            grl = genomicranges.GenomicRangesList(**gr_dict)
            return grl

# QUESTION: should this move to the RegionSet object?
def regionset_to_granges(regionset: RegionSet) -> genomicranges.GenomicRanges:
    """Convert a regionset into a GenomicRanges object"""
    with open(regionset.path, "rb") as f:
        bed_data = f.read()
        gr = self.decompress_and_convert_to_genomic_ranges(bed_data)

        return gr

def read_bedset_file(file_path: str) -> List[str]:
    """Load a bedset from a text file"""
    bed_identifiers = []

    with open(file_path, "r") as f:
        for line in f:
            bed_identifiers.append(line.strip())
    return bed_identifiers
