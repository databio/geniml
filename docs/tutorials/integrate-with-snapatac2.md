# Using our models with SnapATAC2
## Overview

[SnapATAC2](https://github.com/kaizhang/SnapATAC2) is a flexible, versatile, and scalable single-cell omics analysis framework. It is designed to process and analyze single-cell ATAC-seq data. SnapATAC2 is written in Rust with Python bindings. It seemlessly integrates with `scanpy` and `anndata` objects. Therefore, it is extremely easy to use `geniml` models with SnapATAC2. Here's how you can do it:

## Install tools

Ensure that you have `geniml` and `SnapATAC2` installed. You can install both using `pip`:
```bash
pip install geniml snapatac2
```

## Download some data

To get started, let's download some single-cell ATAC-seq data. We will use the [10x Genomics PBMC 10k dataset](https://www.10xgenomics.com/resources/datasets/10k-human-pbmcs-atac-v2-chromium-controller-2-standard). The dataset contains 10,000 peripheral blood mononuclear cells (PBMCs) from a healthy donor.

You can easily grab the fragment files like so:
```bash
wget "https://cf.10xgenomics.com/samples/cell-atac/2.1.0/10k_pbmc_ATACv2_nextgem_Chromium_Controller/10k_pbmc_ATACv2_nextgem_Chromium_Controller_fragments.tsv.gz" -O pbmc_fragments.tsv.gz
```

## Pre-process with SnapATAC2

Lets start by pre-processing the data with SnapATAC2. We will closely follow the [[SnapATAC2 tutorial](https://snapatac2.readthedocs.io/en/latest/tutorial.html](https://kzhang.org/SnapATAC2/tutorials/pbmc.html)) to get the data into an `anndata` object.

### Import the data

Lets import the data into `snapatac2`:
```python
from pathlib import Path
import snapatac2 as snap

fragment_file = Path("pbmc_fragments.tsv.gz")
data = snap.pp.import_data(
    fragment_file,
    chrom_sizes=snap.genome.hg38,
    file="pbmc.h5ad",  # Optional
    sorted_by_barcode=False,
)
```
### Run some basic quality control

Using the `snapatac2` quality control functions, we can quickly assess the quality of the data:

```python
snap.pl.frag_size_distr(data, interactive=False)
fig = snap.pl.frag_size_distr(data, show=False)
fig.update_yaxes(type="log")
fig.show()

snap.metrics.tsse(data, snap.genome.hg38)
snap.pl.tsse(data, interactive=False)

snap.pp.filter_cells(data, min_counts=5000, min_tsse=10, max_counts=100000)
```

Next, we can add a tile matrix to the data, select features, and run `scrublet` which is a doublet detection algorithm:
```python
snap.pp.add_tile_matrix(data)
snap.pp.select_features(data, n_features=250000)
snap.pp.scrublet(data)

# actually filter the cells
snap.pp.filter_doublets(data)
```

With this, we have a clean `anndata` object that we can use with `geniml`.

### Analyze with geniml

We will use a Region2Vec model to cluster the cells by generating embeddings. This PBMC data comes from peripheral blood mononuclear cells (PBMCs) from a healthy donor. As such. we will use the `databio/r2v-luecken2021-hg38-v2` model to generate embeddings because it contains embeddings for the Luecken2021 dataset, a first-of-its-kind multimodal benchmark dataset of 120,000 single cells from the human bone marrow of 10 diverse donors measured with two commercially-available multi-modal technologies: nuclear GEX with joint ATAC, and cellular GEX with joint ADT profiles.

```python
import numpy as np
import scanpy as sc
from geniml.scembed import ScEmbed

adata = sc.read_h5ad("pbmc.h5ad")
model = ScEmbed("databio/r2v-luecken2021-hg38-v2")

adata.obsm['scembed_X'] = np.array(model.encode(adata))
```

With the embeddings, we can run a usual workflow like UMAP, clustering, and visualization:
```python
sc.pp.neighbors(adata, use_rep="scembed_X")
sc.tl.umap(adata)

sc.tl.leiden(adata)
sc.pl.umap(adata, color="leiden")
```

And that's it! You've now used `geniml` with SnapATAC2. You can use the embeddings to annotate cell types, or perform other analyses. If you want to learn more about this, check out the [cell-type annotation](./cell-type-annotation-with-knn.md) tutorial.
