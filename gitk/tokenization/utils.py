import os
from typing import List

import numpy as np
import scanpy as sc
from tqdm import tqdm
from ..io import Region


def anndata_to_regionsets(adata: sc.AnnData) -> List[List[str]]:
    """
    Converts an AnnData object to a list of lists of regions. This
    is done by taking each cell and creating a list of all regions
    that have a value greater than 0.

    This function is already pretty optimized. To speed this up
    further we'd have to parallelize it or reach for a lower-level
    language.

    *Note: this method requires that the sc.AnnData object have
    chr, start, and end in `.var` attributes*
    """
    if not isinstance(adata, sc.AnnData):
        raise ValueError("The input must be a scanpy AnnData object.")

    if not all(
        ["chr" in adata.var.columns, "start" in adata.var.columns, "end" in adata.var.columns]
    ):
        raise ValueError(
            "The AnnData object must have chr, start, and end in the `.var` attribute."
        )

    # Extract the arrays for chr, start, and end
    chr_values = adata.var["chr"].values
    start_values = adata.var["start"].values
    end_values = adata.var["end"].values

    # Perform the comparison using numpy operations
    positive_values = adata.X > 0

    if not isinstance(positive_values, np.ndarray):
        positive_values = positive_values.toarray()

    regions = []
    for i in tqdm(range(adata.shape[0]), total=adata.shape[0], desc="Tokenizing"):
        regions.append(
            [
                Region(chr_values[j], start_values[j], end_values[j])
                for j in np.where(positive_values[i])[0]
            ]
        )
    return regions
