# %%
import os
import scanpy as sc
from gitk.scembed import ScEmbed

# %%
# Load the pretrained model
model = ScEmbed("databio/r2v-luecken2021-hg38-small")

# %%
# Load the data
data_path = os.path.expandvars("$DATA/genomics/scembed/pbmc_hg38/pbmc.h5ad")
adata = sc.read_h5ad(data_path)

# %%
# Embed the data
embeddings = model.encode(adata)
adata.obsm["embedding"] = embeddings

# %%
print(embeddings.shape)

# %%
from umap import UMAP

reducer = UMAP(n_components=2, random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

adata.obsm["umap"] = umap_embeddings
adata.obsm["UMAP1"] = umap_embeddings[:, 0]
adata.obsm["UMAP2"] = umap_embeddings[:, 1]

# %%
sc.pp.neighbors(adata, use_rep="embedding")
sc.tl.leiden(adata, resolution=0.5)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.dpi"] = 300

_, ax = plt.subplots(figsize=(5, 5))

sns.scatterplot(
    data=adata.obsm,
    x="UMAP1",
    y="UMAP2",
    hue=adata.obs["leiden"],
    palette="tab10",
    linewidth=0,
    s=2,
    ax=ax,
)

# add title
ax.set_title("scATAC 10X PBMCs analyzed with ChIP-atlas ATAC", fontsize=14)


# increase font size
for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
):
    item.set_fontsize(12)

# remove legend
ax.get_legend().remove()
