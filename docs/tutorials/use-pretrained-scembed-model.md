# How to use a pre-trained scEmbed model
One advantage of scEmbed is the ability to use pre-trained models. This is useful for quickly getting embeddings of new data without having to train a new model. In this tutorial, we will show how to use a pre-trained model to get embeddings of new data.

I will be using the `databio/luecken2021` model. It was trained on the [Luecken2021](https://openreview.net/forum?id=gN35BGa1Rt) dataset, a first-of-its-kind multimodal benchmark dataset of 120,000 single cells from the human bone marrow of 10 diverse donors measured with two commercially-available multi-modal technologies: nuclear GEX with joint ATAC, and cellular GEX with joint ADT profiles.

This model will work best on PBMC-like data. It also requires your fragments be aligned to the GRCh38 genome.

## Example data preparation
Grab a fresh set of PBMC data from 10X genomics: https://www.10xgenomics.com/resources/datasets/10k-human-pbmcs-atac-v2-chromium-controller-2-standard

You need the [Peak by cell matrix (filtered)](https://cf.10xgenomics.com/samples/cell-atac/2.1.0/10k_pbmc_ATACv2_nextgem_Chromium_Controller/10k_pbmc_ATACv2_nextgem_Chromium_Controller_filtered_peak_bc_matrix.tar.gz). This  contains the binary accessibility matrix, the peaks, and the barcodes. Pre-trained models also requires that the data be in a `scanpy.AnnData` format and the `.var` attribute contain `chr`, `start`, and `end` values. For details on how to make this, see [data preparation](./train-scembed-model.md#data-preparation).

Once your data is ready, you can load it into python and get embeddings.

## Encoding cells

Encoding cells is as easy as:

```python
import scanpy as sc

from gitk.scembed import ScEmbed

adata = sc.read_h5ad("path/to/adata.h5ad")
model = ScEmbed("databio/luecken2021")

embeddings = model.encode(adata)
adata.obsm['scembed_X'] = embeddings
```

And, thats it! You can now cluster your cells using the `scembed_X` embeddings.