import os
from typing import Union, List

import numpy as np
import scanpy as sc
from tqdm import tqdm
from ..io import Region, RegionSet


# better name?
def validate_and_standardize_regions(regions: Union[str, List[Region], RegionSet]) -> List[Region]:
    """
    Validate the input for the regions. A universe accepts a lot of forms of input,
    so we need to check what the user has passed. It also standardizes the input to a list
    of tuples of chr, start, end.

    Users are allowed to pass a list of regions, a path to a BED file, or a RegionSet object.

    :param regions: The regions to use for tokenization. This can be thought of as a vocabulary.
                    This can be either a list of regions or a path to a BED file containing regions.
    :return list: A list of regions or AnnData.
    """
    # check for bedfile
    if isinstance(regions, str):
        # ensure that the file exists
        if not os.path.exists(regions):
            raise FileNotFoundError(f"Could not find file {regions} containing regions.")
        file = regions  # rename for clarity
        regions: List[Region] = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                chr, start, end = line.strip().split("\t")
                regions.append(Region(chr, int(start), int(end)))
        return regions

    # check for list of Region objects
    elif isinstance(regions, list) and all([isinstance(r, Region) for r in regions]):
        return regions

    elif isinstance(regions, RegionSet):
        return regions.regions

    else:
        raise ValueError(
            "The regions must be either a list of regions or a path to a BED file containing regions."
        )


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
