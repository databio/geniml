# How to tokenize a BED file


For hard tokenization, run

```python
from geniml.tokenization import hard_tokenization

src_folder = '/path/to/raw/bed/files/'
dst_folder = '/path/to/tokenized_files/'
universe_file = '/path/to/universe_file.bed'
hard_tokenization(src_folder, dst_folder, universe_file, 1e-9)
```

Note that we use the `intersect` function of `bedtools` to do tokenization. If you want to switch to different tools, you can override the `bedtools_tokenization` function in `hard_tokenization_batch.py` and provide the path to your tool by specifying the input argument `bedtools_path`. The `fraction` argument specifies the minimum overlap required as a fraction of some region in the universe (default: 1E-9,i.e. 1bp; maximum 1.0). A raw region will be mapped into a universe region when an overlap is above the threshold.

By default, the code assumes the binary `bedtools` exists and can be called via command line. If `bedtools` does not exists, the code will raise an exception. To solve this, please specify `bedtools_path` which points to a bedtools binary.

Command line usage
```bash
geniml tokenize --data-folder /folder/with/raw/BED/files --token-folder ./tokens --universe /universe/file --bedtools-path bedtools
```

For more details, type `geniml tokenize --help`.
