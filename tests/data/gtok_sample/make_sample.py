import os

from genimtools.utils import write_tokens_to_gtok
from geniml.io import RegionSet
from geniml.tokenization import ITTokenizer

path = "tests/data/hg38_sample/"
out_path = "tests/data/gtok_sample/"
universe = "tests/data/universe.bed"
tokenizer = ITTokenizer(universe)

for bed in os.listdir(path):
    if bed.endswith(".bed"):
        gtok_file = bed.replace(".bed", ".gtok")
        bed_path = os.path.join(path, bed)
        region_set = RegionSet(bed_path)
        tokens = tokenizer.tokenize(region_set)
        write_tokens_to_gtok(os.path.join(out_path, gtok_file), tokens.ids)
