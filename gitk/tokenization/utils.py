import os
from typing import Union, List, Tuple, Dict, TYPE_CHECKING

import numpy as np
import scanpy as sc
from tqdm import tqdm

if TYPE_CHECKING:
    from .main import Universe
from ..io import Region


# better name?
def extract_regions_from_bed_or_list(regions: Union[str, List[str], List[Region]]) -> List[Region]:
    """
    Validate the input for the regions. A universe accepts a lot of forms of input,
    so we need to check what the user has passed. It also standardizes the input to a list
    of tuples of chr, start, end.

    Users are allowed to pass a list of regions, a path to a BED file, or an AnnData object.

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

    # check for list of chr_start_end strings
    elif isinstance(regions, list) and all([isinstance(r, str) for r in regions]):
        regions_parsed: List[Region] = []
        for region in regions:
            chr, start, end = region.split("_")
            regions_parsed.append(Region(chr, int(start), int(end)))
        return regions_parsed

    # check for list of Region objects
    elif isinstance(regions, list) and all([isinstance(r, Region) for r in regions]):
        return regions

    else:
        raise ValueError(
            "The regions must be either a list of regions or a path to a BED file containing regions."
        )


def generate_var_conversion_map(
    a: List[Region],
    b: "Universe",
    fraction: float = 1.0e-9,  # not used
) -> Dict[str, Union[str, None]]:
    """
    Create a conversion map to convert regions from a to b. This is used to convert the
    consensus peak set of one AnnData object to another.

    For each region in a, we will either find a matching region in b, or None. If a matching
    region is found, we will store the region in b. If no matching region is found, we will
    store `None`.

    Intuitively, think of this as converting `A` --> `B`. If a region in `A` is found in `B`,
    we will change the region in `A` to the region in `B`. If a region in `A` is not found in
    `B`, we will drop that region in `A` altogether.

    :param List[tuple[str, int, int]] a: the first list of regions
    :param Universe: the second list of regions as a Universe object
    :param float fraction: the fraction of the region that must overlap to be considered an overlap. Not used.
    """

    conversion_map = dict()

    for region in tqdm(a, total=len(a), desc="Generating conversion map"):
        overlaps = b.query(region)
        region_str = f"{region.chr}_{region.start}_{region.end}"
        if len(overlaps) > 0:
            olap = overlaps[0]  # take the first overlap for now, we can change this later
            olap_str = f"{olap.chr}_{olap.start}_{olap.end}"
            conversion_map[region_str] = olap_str
        else:
            conversion_map[region_str] = None

    return conversion_map


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
