# %%
import os

import numpy as np
import scanpy as sc

# %%
path_to_pbmc = "$DATA/genomics/10X/10kpbmcsV2_atac/pbmc.h5ad"

# exapand the path
path_to_pbmc = os.path.expandvars(path_to_pbmc)
adata = sc.read_h5ad(path_to_pbmc)

# %%
sc.pp.subsample(adata, n_obs=20)

# %%
n_rows, n_cols = adata.shape

# randomly select 1000 regions to keep
np.random.seed(0)
keep = np.random.choice(range(n_cols), size=1000, replace=False)
adata = adata[:, keep]

# %%
adata.write_h5ad("pbmc_hg38.h5ad")
