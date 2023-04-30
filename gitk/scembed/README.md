# scEmbed
`scEmbed` is a single-cell implementation of `Region2Vec`: a method to represent genomic region sets as vectors, or embeddings, using an adapted word2vec approach. `scEmbed` allows for dimensionality reduction and feature selection of single-cell ATAC-seq data; a notoriously sparse and high-dimensional data type. We intend for `scEmbed` to be used with the [`scanpy`](https://scanpy.readthedocs.io/en/stable/) package. As such, it natively accepts `AnnData` objects as input and returns `AnnData` objects as output.

## Installation
Simply install the parent package `gitk` from PyPi:

```bash
pip install gitk
```

Then import `scEmbed` from `gitk`:

```python
from gitk.scembed import SCEmbed
```

## Usage
`scEmbed` is simple to use. Import your `AnnData` object and pass it to `SCEmbed`. Note: your `AnnData` object **must** have a `.var` attribute with three columns: `chr`, `start`, and `end`. These columns should be the chromosome, start position, and end position of each genomic region, respectively. `scEmbed` will automatically import these and use them for training.

```python
import scanpy as sc
from gitk.scembed import SCEmbed

adata = sc.read_h5ad("path/to/adata.h5ad")

scEmbed = SCEmbed(adata)
scEmbed.train(
    epochs=100,
)

cell_embeddings = scEmbed.get_cell_embeddings()
```