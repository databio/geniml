# %%
import os
from gitk import scembed

folder = "/Users/nathanleroy/Desktop/pbmc"

path_to_barcodes = os.path.join(folder, "barcodes.tsv")
path_to_mtx = os.path.join(folder, "matrix.mtx")
path_to_peaks = os.path.join(folder, "peaks.bed")


# %%
adata = scembed.utils.barcode_mtx_peaks_to_anndata(
    path_to_barcodes, path_to_mtx, path_to_peaks
)

# %%

adata.write_h5ad(os.path.join(folder, "pbmc.h5ad"))
