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
from geniml.scembed import SCEmbed
```

## Usage
`scEmbed` is simple to use. Import your `AnnData` object and pass it to `SCEmbed`. `scEmbed` will work regardless of any `var` or `obs` annotations, but it is recommended that you use `scEmbed` after you have already performed some basic filtering and quality control on your data. Further. If you'd like to maintain information about region embeddings, it is recommended that you attach `chr`, `start`, adn `end` annotations to your `AnnData` object before passing it to `scEmbed`.

```python
import scanpy as sc
from geniml.scembed import SCEmbed

adata = sc.read_h5ad("path/to/adata.h5ad")

scEmbed = SCEmbed(adata)
scEmbed.train(
    epochs=100,
)

cell_embeddings = scEmbed.get_cell_embeddings()
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

## Training the model
To train an `scembed` model, just create an instance of the `SCEmbed` model class, define your hyperparamters, and call the `train` method. For this example, we will use the default hyperparameters. The only thing we need to specify is the number of epochs to train for. We will train for 10 epochs:

```python
import logging

import scanpy as sc
from geniml.scembed import SCEmbed

# if you want to see the training progress
logging.basicConfig(level=logging.INFO)

model = SCEmbed(
    use_default_region_names=False  # this is to specify that we want to use chr, start, end.
)
model.train(adata, epochs=3)  # we recomend increasing this to 100
```

Thats it!

## Clustering the cells
With the model now trained, we can get embeddings of our cells. This occurs in two steps: 1) tokenize the data and 2) encode the cells.

**Tokenize:**

```python
from geniml.tokenization import HardTokenizer


tokenizer = HardTokenizer("peaks.bed")  # consensus peak set

region_sets = tokenizer(adata)
```

**Encode:**
```python
cell_embeddings = []
for region_set in region_sets:
    embedding = np.mean([model(r) for r in region_set if r in model.region2vec, axis=0])
    cell_embeddings.append(embedding)

cell_embeddings = np.array(cell_embeddings)

adata.obsm['scembed_X'] = cell_embeddings
```

Now you can use `scanpy` utilities to cluster the cells:

```python
sc.pp.neighbors(adata, use_rep="scembed_X")
sc.tl.leiden(adata)
```

And visualize with UMAP

```python
sc.tl.umap(adata)
sc.pl.umap(adata, color="leiden")
```