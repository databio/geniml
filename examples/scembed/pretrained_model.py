# %%
import scanpy as sc
from gitk.models import PretrainedScembedModel

# %%
# Load the pretrained model
model = PretrainedScembedModel("nleroy917/luecken2021")

# %%
# Load the data
adata = sc.read_h5ad("../../tests/data/pbmc_hg38.h5ad")

# %%
# Embed the data
embeddings = model.encode(adata)

# %%
print(embeddings.shape)
