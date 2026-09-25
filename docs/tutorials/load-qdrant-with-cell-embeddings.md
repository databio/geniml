# How to load a vector database with cell embeddings
## Overview

In this tutorial, we will show how to load a vector database with cell embeddings. There are many benefits to storing cell-embeddings in a vector database:
1. **Speed**: Loading a vector database is much faster than re-encoding cells.
2. **Reproducibility**: You can share your cell embeddings with others.
3. **Flexibility**: You can use the same cell embeddings for many different analyses.
4. **Interoperability**: You can use the same cell embeddings with many different tools.

In a subsequent tutorial, we will show how to use a vector database to query cell embeddings and annotate cells with cell-type labels using a KNN classification algorithm.

## Preqrequisites
There are two core components to this tutorial: 1) the pre-trained model, and 2) the vector database.

**Pre-trained model:**
I will be using the `databio/luecken2021` model. It was trained on the [Luecken2021](https://openreview.net/forum?id=gN35BGa1Rt) dataset, a first-of-its-kind multimodal benchmark dataset of 120,000 single cells from the human bone marrow of 10 diverse donors measured with two commercially-available multi-modal technologies: nuclear GEX with joint ATAC, and cellular GEX with joint ADT profiles.

**Vector database:**
Vector databases are a new and exciting technology that allow you to store and query high-dimensional vectors very quickly. This tutorial will use the `qdrant` vector database. As a lab, we really like `qdrant` because it is fast, easy to use, and has a great API. You can learn more about `qdrant` [here](https://qdrant.com/). For `qdrant` setup, please refer to the [qdrant documentation](https://qdrant.com/docs/). In the end, you should have a running `qdrant` instance at `http://localhost:6333`.

## Data preparation

Grab a fresh copy of the Luecken2021 data from the [geo accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122). We want the `multiome` data. This dataset contains the binary accessibility matrix, the peaks, and the barcodes. It also conveniently contains the cell-type labels. Pre-trained models also requires that the data be in a `scanpy.AnnData` format and the `.var` attribute contain `chr`, `start`, and `end` values.

```python
import scanpy as sc

adata = sc.read_h5ad("path/to/adata.h5ad")
adata = adata[:, adata.var['feature_types'] == 'ATAC']
```

## Getting embeddings
We can easily get embeddings of the dataset using the pre-trained model:

```python
import scanpy as sc

from geniml.scembed import ScEmbed

adata = sc.read_h5ad("path/to/adata.h5ad")

model = ScEmbed("databio/r2v-luecken2021-hg38-v2")
embeddings = model.encode(adata)

adata.obsm['scembed_X'] = np.array(embeddings)
```

## Loading the vector database
With the embeddings, we can now upsert them to `qdrant`. Ensure you have `qdrant_client` installed:

```bash
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="luecken2021",
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.DOT),
)

embeddings, cell_types = adata.obsm['scembed_X'], adata.obs['cell_type']

points = []
for embedding, cell_type, i in zip(embeddings, cell_types, range(len(embeddings)):
    points.append(
        PointStruct(
            id=adata.obs.index[i],
            vector=embedding.tolist(),
            payload={"cell_type": cell_type}

    ))


client.upsert(collection_name="luecken2021", points=points, wait=True)
```

You should now have a vector database with cell embeddings. In the next tutorial, we will show how to use this vector database to query cell embeddings and annotate cells with cell-type labels using a KNN classification algorithm.
