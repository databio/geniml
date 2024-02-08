# How to train a single-cell model with scEmbed

This example walks you through training an `scembed` region2vec model on a single-cell dataset. We start with data preparation, then train the model, and finally use the model to cluster the cells.

For this example we are using the [10x Genomics PBMC 10k dataset](https://www.10xgenomics.com/resources/datasets/10k-human-pbmcs-atac-v2-chromium-controller-2-standard). The dataset contains 10,000 peripheral blood mononuclear cells (PBMCs) from a healthy donor.


## Installation

Simply install the parent package `geniml` from PyPi:

```bash
pip install geniml
```

Then import `scEmbed` from `geniml`:

```python
from geniml.scembed import ScEmbed
```

## Data preparation
`scembed` requires that the input data is in the [AnnData](https://anndata.readthedocs.io/en/latest/) format. Moreover, the `.var` attribute of this object must have `chr`, `start`, and `end` values. The reason is two fold: 1) we can track which vectors belong to which genmomic regions, and 2) region vectors are now reusable. We ned three files: 1) The `barcodes.txt` file, 2) the `peaks.bed` file, and 3) the `matrix.mtx` file. These will be used to create the `AnnData` object. To begin, download the data from the 10x Genomics website:

```bash
wget https://cf.10xgenomics.com/samples/cell-atac/2.1.0/10k_pbmc_ATACv2_nextgem_Chromium_Controller/10k_pbmc_ATACv2_nextgem_Chromium_Controller_raw_peak_bc_matrix.tar.gz
tar -xzf 10k_pbmc_ATACv2_nextgem_Chromium_Controller_raw_peak_bc_matrix.tar.gz
```

Your files will be inside `filtered_peak_bc_matrix/`. Assuming you've installed the proper dependencies, you can now use python to build the `AnnData` object:

```python
import pandas as pd
import scanpy as sc

from scipy.io import mmread
from scipy.sparse import csr_matrix

barcodes = pd.read_csv("barcodes.tsv", sep="\t", header=None, names=["barcode"])
peaks = pd.read_csv("peaks.bed", sep="\t", header=None, names=["chr", "start", "end"])
mtx = mmread("matrix.mtx")
mtx_sparse = csr_matrix(mtx)
mtx_sparse = mtx_sparse.T

adata = sc.AnnData(X=mtx_sparse, obs=barcodes, var=peaks)
adata.write_h5ad("pbmc.h5ad")
```

We will use the `pbmc.h5ad` file for downstream work.

## Training

Training an `scEmbed` model requires two key steps: 1) pre-tokenizing the data, and 2) training the model.

### Pre-tokenizing the data
To learn more about pre-tokenizing the data, see the [pre-tokenization tutorial](./pre-tokenization.md). Pre-tokenization offers many benefits, the two most important being 1) speeding up training, and 2) lower resource requirements. The pre-tokenization process is simple and can be done with a combination of `geniml` and `genimtools` utilities. Here is an example of how to pre-tokenize the 10x Genomics PBMC 10k dataset:

```python
from genimtools.utils import write_tokens_to_gtok
from geniml.tokenization import ITTokenizer

adata = sc.read_h5ad("path/to/adata.h5ad")
tokenizer = ITTokenizer("peaks.bed")

tokens = tokenizer(adata)

for i, t in enumerate(tokens):
    file = f"tokens{i}.gtok"
    write_tokens_to_gtok(t, file)
```

### Training the model

Now that the data is pre-tokenized, we can train the model. The `scEmbed` model is designed to be used with `scanpy`. Here is an example of how to train the model:

```python

from geniml.region2vec.utils import Region2VecDataset

dataset = Region2VecDataset("path/to/tokens")

model = ScEmbed(
    tokenizer=tokenizer,
)
model.train(
    dataset,
    epochs=100,
)
```

We can then export the model for upload to huggingface:

```python
model.export("path/to/model")
```

## Get embeddings of single-cells
`scEmbed` is simple to use and designed to be used with `scanpy`. Here is a simple example of how to train a model and get cell embeddings:

```python
model = ScEmbed.from_pretrained("path/to/model")
model = ScEmbed("databio/scembed-pbmc10k")

adata = sc.read_h5ad("path/to/adata.h5ad")
embeddings = model.encode(adata)

adata.obsm["scembed_X"] = embeddings
```

## Clustering the cells
With the model now trained, and cell-embeddings obtained, we can get embeddings of our individual cells. You can use `scanpy` utilities to cluster the cells:

```python
sc.pp.neighbors(adata, use_rep="scembed_X")
sc.tl.leiden(adata) # or louvain
```

And visualize with UMAP

```python
sc.tl.umap(adata)
sc.pl.umap(adata, color="leiden")
```
