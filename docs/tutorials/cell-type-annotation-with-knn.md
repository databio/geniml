# How to annotate cell-types with KNN
In the [previous tutorial](./load-qdrant-with-cell-embeddings.md), we loaded a vector database with cell embeddings. In this tutorial, we will show how to use this vector database to query cell embeddings and annotate cells with cell-type labels using a KNN classification algorithm.

If you have **not** completed the previous tutorial, you should ensure you have a vector database with cell embeddings.

## What is K-nearest-neighbors (KNN) classification?
According to [IBM](https://www.ibm.com/topics/knn), K-nearest-neighbors classification is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. Point more simply: KNN is a classification algorithm that uses the distance between an **unlabeled** data point and its **labed** neighbors to classify the new data point.

Assuming we have a vector-space of well-annotated cell embeddings, we can use KNN to classify new cell embeddings based on their proximity to the labeled cell embeddings.

## Querying the vector database
First, we need to generate new cell embeddings for the cells we want to annotate. **Note: it is imperative that the new cell embeddings are generated using the same model as the cell embeddings in the vector database.** The previous tutorial used `databio/r2v-luecken2021-hg38-v2` to generate cell embeddings. We will use the same model to generate new cell embeddings.

```python
import scanpy as sc

from geniml.scembed import ScEmbed

adata = sc.read_h5ad("path/to/adata_unlabeled.h5ad")

model = ScEmbed("databio/r2v-luecken2021-hg38-v2")
```

We can get embeddings of the dataset using the pre-trained model:

```python
embeddings = model.encode(adata)

adata.obsm['scembed_X'] = np.array(embeddings)
```

Now that we have the new cell embeddings, we can query the vector database to find the K-nearest-neighbors of each cell embedding.

```python
from collections import Counter
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Query the vector database
k = 5 # set to whatever value you want, this is a hyperparameter

for i, embedding in enumerate(embeddings):
    neighbors = client.search(
        collection_name="luecken2021", 
        query_vector=embedding.tolist(), 
        limit=k, 
        with_payload=True
    )
    cell_types = [neighbor.payload["cell_type"] for neighbor in neighbors]

    # get majority
    cell_type = Counter(cell_types).most_common(1)[0][0]
    adata.obs['cell_type'][i] = cell_type
```

And just like that, we've annotated our cells with cell-type labels using KNN classification. We can improve this methodology by first clustering the unlabeled cells and then using the cluster centroids to query the vector database. This will reduce the number of queries and improve the speed of the annotation process. Another approach would be to do a secondary consensus vote on each cluster and assign one label per cluster.

