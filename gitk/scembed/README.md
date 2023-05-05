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
`scEmbed` is simple to use. Import your `AnnData` object and pass it to `SCEmbed`. `scEmbed` will work regardless of any `var` or `obs` annotations, but it is recommended that you use `scEmbed` after you have already performed some basic filtering and quality control on your data. Further. If you'd like to maintain information about region embeddings, it is recommended that you attach `chr`, `start`, adn `end` annotations to your `AnnData` object before passing it to `scEmbed`.

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