# %%
# import libraries
import logging
import scanpy as sc
from gitk import scembed

# set to DEBUG to see more info
logging.basicConfig(level=logging.INFO)

# %%
adata = sc.read_h5ad("../data/buenrostro2018.h5ad", backed="r")

# for visualization at the end
COLORMAP = {
    "CLP": "#bbe7f3",
    "CMP": "#fecd84",
    "GMP": "#fba102",
    "HSC": "#093e12",
    "LMPP": "#35c3ad",
    "MEP": "#eb2d35",
    "momo": "#eb5000",
    "MPP": "#41a713",
    "pDC": "#da9de2",
    "UNK": "#2c2625",
}

# %%
from multiprocessing import cpu_count

chunker = scembed.AnnDataChunker(adata, chunk_size=500)
model = scembed.SCEmbed(
    threads=cpu_count() - 1, min_count=2, use_default_region_names=False
)

# %%
from tqdm import tqdm

# train in chunks
i = 1
for chunk in tqdm(chunker):
    print(f"Training chunk {i} of {len(chunker)} | {chunk.shape[0]} cells")
    model.train(chunk, epochs=100)
    i += 1

# %%
# get embeddings out of the adata object
model.save_model("buenrostro2018.model")

# %%
model.load_model("buenrostro2018.model")

# %%
adata = sc.read_h5ad("../data/buenrostro2018.h5ad")
documents = scembed.convert_anndata_to_documents(adata)

# %%
import numpy as np


def get_embedding(r: str) -> np.ndarray:
    return model.get_embedding(r)


cell_embeddings = []
for cell in tqdm(documents, total=len(documents)):
    cell_embedding = np.mean(
        [get_embedding(r) for r in cell if r in r in model.region2vec], axis=0
    )
    cell_embeddings.append(cell_embedding)

# attach embeddings to the AnnData object
adata.obs["embedding"] = cell_embeddings

# %%
# add cell type labels using the metadata
metadata = pd.read_csv(
    "../data/buenrostro_metadata.tsv", sep="\t", skiprows=1, names=["id", "label"]
)

adata.obs["label"] = adata.obs["id"].map(metadata.set_index("id")["label"])


# %%
import pandas as pd

embeddings_df = pd.DataFrame(adata.obs["embedding"].tolist())

# %%
import pandas as pd
from umap import UMAP

reducer = UMAP(n_components=2, random_state=42)
umap_embeddings = reducer.fit_transform(embeddings_df)

# %%
adata.obs["umap1"] = umap_embeddings[:, 0]
adata.obs["umap2"] = umap_embeddings[:, 1]

# %%


# attach cell types
embeddings_df["color"] = embeddings_df["id"].map(metadata.set_index("id")["label"])

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=adata.obs, x="umap1", y="umap2", ax=ax, s=10, alpha=1)
ax.set_title("UMAP of Buenrostro 2018 data")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")

# increase font
for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontsize(20)

# arial
for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontname("Arial")
