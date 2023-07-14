# Tokenization

## Introduction

In NLP, training word embeddings requires first tokenizing words such that words in different forms are represented by one word. For example, "orange", "oranges" and "Orange" are all mapped to "orange" since they essentially convey the same meaning. This reduces the vocabulary size and improves the quality of learned embeddings. Similary, many `gitk` modules (such as `region2vec`) require first tokenizating regions.

To tokenize reigons, we need to provide a universe, which specifies the "vocabulary" of genomic regions. The universe is a BED file, containing representative regions. With the given universe, we represent (tokenize) raw regions into the regions in the universe.

Different strategies can be used to tokenize. The simplest case we call *hard tokenization*, which means if the overlap between a raw region in a BED file and a region in the universe exceeds a certain amount, then we use the region in the universe to represent this raw region; otherwise, we ignore this raw region. This is a "zero or one" process. After hard tokenization, each BED file will contain only regions from the universe, and the number of regions will be smaller or equal to the original number.

## Usage

For hard tokenization, run
```
from gitk.tokenization import hard_tokenization

src_folder = '/path/to/raw/bed/files/'
dst_folder = '/path/to/tokenized_files/'
universe_file = '/path/to/universe_file.bed'
hard_tokenization(src_folder, dst_folder, universe_file, 1e-9)

```
Note that we use the `intersect` function of `bedtools` to do tokenization. If you want to switch to different tools, you can override the `bedtool_tokenization` function in `hard_tokenization_batch.py` and provide the path to your tool by specifying the input argument `bedtools_path`. The `fraction` argument specifies the minimum overlap required as a fraction of some region in the universe (default: 1E-9,i.e. 1bp; maximum 1.0). A raw region will be mapped into a universe region when an overlap is above the threshold.

By default, the code assumes the binary `bedtools` exists and can be called via command line. If `bedtools` does not exists, the code will raise an exception. To solve this, please specify `bedtools_path` which points to a bedtools binary.

Command line usage
```bash
gitk tokenize --data-folder /folder/with/raw/BED/files --token-folder ./tokens --universe /universe/file --bedtools-path bedtools
```

For more details, type `gitk tokenize --help`.
