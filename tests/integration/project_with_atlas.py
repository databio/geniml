# %%
import os
import scanpy as sc
from gitk import scembed

folder = "/Users/nathanleroy/Desktop/pbmc"

pbmc = sc.read_h5ad(os.path.join(folder, "pbmc.h5ad"))

# %%
projector = scembed.Projector("databio/scatlas")
pbmc = projector.project(pbmc)


# %%
import numpy as np
import umap

reducer = umap.UMAP(
    n_components=2,
    random_state=42,
)

embeddings = np.array(pbmc.obs["embedding"].tolist())
umap_embedding = reducer.fit_transform(embeddings)

# %%
# create dataframe with umap embeddings and original labels
import pandas as pd

umap_df = pd.DataFrame(umap_embedding, columns=["umap1", "umap2"])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

_, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=umap_df,
    x="umap1",
    y="umap2",
    palette="tab20",
    linewidth=0,
    s=10,
    alpha=0.8,
    ax=ax,
)

# increase font size
ax.legend(fontsize=10)
ax.set_xlabel("UMAP 1", fontsize=16)
ax.set_ylabel("UMAP 2", fontsize=16)
ax.set_title("PBMCs", fontsize=20)
ax.tick_params(labelsize=14)
