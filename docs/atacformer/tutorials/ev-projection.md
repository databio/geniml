Reference mapping is a common technique in single-cell genomics. It involves representing both new data and a high-quality reference dataset in the same latent space, allowing for comparison and analysis of new embeddings against established ones.

Atacformer, because it can generate embeddings for any genomic region, is well-suited for this task. You can use it to project new embeddings into the latent space of a reference dataset, enabling you to visualize and analyze how new data relates to existing references.

## Prerequisites

Ensure you have `geniml` and `gtars` installed. `gtars` is our companion library for genomic interval data processing, and `geniml` provides the Atacformer model and tokenizer.

The steps we will follow include:

1. Load a pre-trained Atacformer model.
2. Load a reference dataset along side a query dataset.
3. Tokenize both datasets.
4. Generate embeddings for both datasets.
5. Create a UMAP projection of the reference dataset embeddings.
6. Visualize the new embeddings in the latent space of the reference dataset.
7. Interpret the results and draw conclusions about the relationship between the new and reference datasets.

## Loading the Pre-trained Model
To start, we need to load a pre-trained Atacformer model. This model will be used to generate embeddings for both the reference and new datasets.

```python
from gtars.tokenizers import Tokenizer
from geniml.atacformer import AtacformerForCellClustering

# craft performs well on blood datasets
model = AtacformerForCellClustering.from_pretrained("databio/atacformer-craft100k-hg38")
# model = model.to("cuda")  # or "mps" for Apple Silicon


tokenizer = Tokenizer.from_pretrained("databio/atacformer-craft100k-hg38")
```

## Loading the Datasets
TODO: how do we let users quickly load the Luecken2021 and pbmc datasets?

```python
import scanpy as sc

# load the reference dataset (e.g., Luecken2021)
reference_dataset = sc.read("path/to/reference_dataset.h5ad")

# load the query dataset (e.g., pbmc)
query_dataset = sc.read("path/to/query_dataset.h5ad")
```

## Tokenizing the Datasets
Next, we need to tokenize both the reference and new datasets. This step converts the genomic intervals into a format that the Atacformer model can understand, specifically input ids that represent genomic regions.

```python
from geniml.atacformer import tokenize_anndata

reference_inputs = tokenize_anndata(reference_dataset, tokenizer)
query_inputs = tokenize_anndata(query_dataset, tokenizer)

reference_input_ids = [t["input_ids"] for t in reference_inputs]
query_input_ids = [t["input_ids"] for t in query_inputs]
```

## Generating Embeddings
Now that we have tokenized both datasets, we can generate embeddings using the Atacformer model. These embeddings will represent the genomic regions in a high-dimensional space.

```python
batch_size = 4  # adjust based on your memory
reference_embeddings = model.encode_tokenized_cells(
    reference_input_ids,
    batch_size=batch_size,
)
query_embeddings = model.encode_tokenized_cells(
    query_input_ids,
    batch_size=batch_size,
)

# attach embeddings to the AnnData objects
reference_dataset.obsm["X_atacformer"] = reference_embeddings.cpu().numpy()
query_dataset.obsm["X_atacformer"] = query_embeddings.cpu().numpy()
```

## Visualize the embeddings
Before we project the new embeddings into the reference latent space, it can be helpful to visualize the embeddings of both datasets separately. This step allows us to see the distribution of the embeddings in their respective latent spaces.

We can also perform leiden clustering on the embeddings to identify clusters in the query dataset.

```python
import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP

# perform Leiden clustering on the query dataset (since we dont have labels)
sc.pp.neighbors(query_dataset, use_rep='X_atacformer', n_neighbors=10)
sc.tl.leiden(query_dataset)

# generate two UMAPS, one for the reference and one for the query dataset
umap_model = UMAP(n_neighbors=15, random_state=42)
reference_umap = umap_model.fit_transform(reference_dataset.obsm["X_atacformer"])

umap_model = UMAP(n_neighbors=15, random_state=42)
query_umap = umap_model.fit_transform(query_dataset.obsm["X_atacformer"])

# plot the UMAPs
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=reference_umap[:, 0],
    y=reference_umap[:, 1],
    hue=reference_dataset.obs['cell_type'],  # assuming you have cell type labels
    palette='tab20',
    ax=plt.subplot(1, 2, 1)
)
sns.scatterplot(
    x=query_umap[:, 0],
    y=query_umap[:, 1],
    hue=query_dataset.obs['leiden'],  # using leiden clusters
    palette='tab20',
    ax=plt.subplot(1, 2, 2)
)

plt.suptitle("UMAP of Reference and Query Datasets")
plt.show()
```

## Projecting New Embeddings into the Reference Latent Space
Now, we can project the new embeddings (from the query dataset) into the latent space of the reference dataset. This step allows us to visualize how the new data relates to the established reference.

```python
umap_model = UMAP(random_state=42)
reference_umap = umap_model.fit_transform(reference_dataset.obsm["X_atacformer"])
query_umap = umap_model.transform(query_dataset.obsm["X_atacformer"])

# plot the projected query embeddings in the reference latent space
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=reference_umap[:, 0],
    y=reference_umap[:, 1],
    color='gray',
    label='Reference',
    alpha=0.5
)
sns.scatterplot(
    x=query_umap[:, 0],
    y=query_umap[:, 1],
    hue=query_dataset.obs['leiden'],  # using leiden clusters
    palette='tab20',
    label='Query',
    alpha=0.8
)
plt.title("Projected Query Embeddings in Reference Latent Space")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.show()
```