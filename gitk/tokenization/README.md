# Tokenization
## Introduction
In training word embeddings, we need to first tokenize each word such that words in different forms are represented by one word. For example, "orange", "oranges" and "Orange" are all mapped to "orange" since they essentially convey the same meaning. This can reduce the vocabulary size and also improve the quality of learned embeddings.<br>
Similary, before running region2vec, we need to run tokenization on regions. First, we need to provide a universe, the "vocabulary" in the setting of genomic regions. The universe is a BED file, containing representative regions. With the given universe, we represent (tokenize) raw regions with the regions in the universe.

For hard tokenization, if the overlap between a raw region in a bed file and a region in the universe exceeds a certain amount, then we use the region in the universe to represent this raw region; otherwise, we ignore this raw region. This is a "zeor or one" process. After hard tokenization, each bed file will contain regions all from the universe, and the number of regions will be smaller or equal to the original number.


## Usage
For hard tokenization, run
```
from gitk.tokenization import hard_tokenization

src_folder = '/path/to/raw/bed/files'
dst_folder = '/path/to/tokenized_files'
universe_file = '/path/to/universe_file'
hard_tokenization(src_folder, dst_folder, universe_file, 1e-9)

```
Note that we use the `intersect` function of `bedtools` to do tokenization. If you want to switch to different tools, you can override the `bedtool_tokenization` function in `hard_tokenization_batch.py` and provide the path to your tool by specifying the input argument `bedtools_path`. The `fraction` argument specifies the minimum overlap required as a fraction of some region in the universe (default: 1E-9,i.e. 1bp; maximum 1.0). A raw region will be mapped into a universe region when an overlap is above the threshold.

The bedtools (version 2.30.0) will be automatically downloaded from https://github.com/arq5x/bedtools2/releases to the `bedtools` folder.
