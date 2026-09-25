# Pre-tokenize data for training
Before doing any training with Atacformer, you need to pre-tokenize your genomic interval data. This process converts your data into a format that the Atacformer model can understand, specifically input ids that represent genomic regions.

For easy reading, writing, and manipulation of pre-tokenized data, we use the Parquet format. This format is efficient and works well with the huggingface ecosystem; namely, the [`datasets`](https://huggingface.co/docs/datasets/index) library.

## Prerequisites

Ensure you have `geniml` and `gtars` installed. `gtars` is our companion library for genomic interval data processing -- it contains the tokenizers. `geniml` provides the Atacformer model.
```bash
pip install geniml[ml] datasets gtars
```

Next, ensure you have a universe file that defines the genomic regions you want to tokenize. This file is typically in BED format and contains the regions of interest.

Finally, you need your data in the `AnnData` format or as `.fragments.tsv.gz` files. If you have your data in a different format, you may need to convert it first. If using `AnnData`, please ensure that you have a `chr`, `start`, and `end` column in the `.var` dataframe.

## Tokenization Process

To pre-tokenize your data, you can use the `gtars` tokenizer. This tokenizer will read your genomic interval data and convert it into the Atacformer input format.

```python
import scanpy as sc
import polars as pl

from geniml.tokenization import tokenize_anndata
from gtars.tokenizers import Tokenizer

# read in data; create a tokenizer
adata = sc.read("path/to/your/anndata.h5ad")
tokenizer = Tokenizer.from_pretrained("databio/atacformer-base-hg38")

# tokenize the data
tokens = tokenize_anndata(adata, tokenizer)
input_ids = [t["input_ids"] for t in tokens]

# optional: cutoff at context size
CONTEXT_SIZE = 8192
input_ids = [ids[:CONTEXT_SIZE] for ids in input_ids]

# convert to a DataFrame
df = pl.DataFrame({
    "input_ids": input_ids,
})

df.write_parquet("path/to/tokenized_data.parquet")
```