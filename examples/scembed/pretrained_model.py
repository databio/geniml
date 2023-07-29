# %%
import scanpy as sc
from gitk.scembed import ScEmbed

# %%
# Load the pretrained model
model = ScEmbed("databio/r2v-luecken2021-hg38-small")

# %%
# Load the data
adata = sc.read_h5ad("../../tests/data/pbmc_hg38.h5ad")

# %%
# Embed the data
embeddings = model.encode(adata)

# %%
print(embeddings.shape)
