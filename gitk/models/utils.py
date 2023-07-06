import os
import subprocess
from typing import List, Dict, Union, TYPE_CHECKING

import numpy as np
import scanpy as sc
from tqdm import tqdm

if TYPE_CHECKING:
    from .tokenization import Universe
from .const import CACHE_DIR


def get_path_to_home_dir():
    """
    Get the path to the home directory.
    """
    return os.path.expanduser("~")


def get_cache_dir():
    """
    Get the path to the cache directory.
    """
    _home = get_path_to_home_dir()
    return os.path.join(_home, CACHE_DIR)


def make_cache_dir():
    """
    Create cache directory in the current working directory.
    """
    _home = get_path_to_home_dir()
    _cache_dir = os.path.join(_home, CACHE_DIR)
    if not os.path.exists(_cache_dir):
        os.mkdir(_cache_dir)


def validate_region_input(
    regions: Union[str, List[str], List[tuple[str]]]
) -> List[tuple[str, int, int]]:
    """
    Validate the input for the regions. this universe accepts a lot of forms of input,
    so we need to check what the user has passed. It also standardizes the input to a list
    of tuples of chr, start, end.

    Users are allowed to pass a list of regions, a list of tuples with chr, start, and end, or
    a path to a BED file.

    :param regions: The regions to use for tokenization. This can be thought of as a vocabulary.
                    This can be either a list of regions or a path to a BED file containing regions.
    :return A list of tuples of chr, start, end.
    """
    # check for bedfile
    if isinstance(regions, str):
        # ensure that the file exists
        if not os.path.exists(regions):
            raise FileNotFoundError(
                f"Could not find file {regions} containing regions."
            )
        file = regions  # rename for clarity
        regions = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                regions.append(tuple(line.strip().split("\t")))
        return regions

    # check for list of chr_start_end strings
    elif isinstance(regions, list) and all([isinstance(r, str) for r in regions]):
        regions = [tuple(r.split("_")) for r in regions]
        return regions

    # check for list of tuples of chr, start, end
    elif isinstance(regions, list) and all(
        [(isinstance(r, tuple) or isinstance(r, list)) and len(r) == 3 for r in regions]
    ):
        return regions
    else:
        raise ValueError(
            "The regions must be either a list of regions or a path to a BED file containing regions."
        )


def generate_var_conversion_map(
    a: List[tuple[str, int, int]],
    b: "Universe",
    fraction: float = 1.0e-9,
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
    :param float fraction: the fraction of the region that must overlap to be considered an overlap
    """

    conversion_map = dict()

    for region in tqdm(a, total=len(a)):
        overlaps = b.query(region)
        region_str = f"{region[0]}_{region[1]}_{region[2]}"
        if len(overlaps) > 0:
            olap = overlaps[
                0
            ]  # take the first overlap for now, we can change this later
            olap_str = f"{olap[0]}_{olap[1]}_{olap[2]}"
            conversion_map[region_str] = olap_str
        else:
            conversion_map[region_str] = None

    return conversion_map


def convert_to_universe(
    adata: sc.AnnData, universe_set: List[tuple[str, int, int]]
) -> sc.AnnData:
    """
    Converts the consensus peak set (.var) attributes of the AnnData object
    to a universe representation. This is done through interval overlap
    analysis with bedtools.

    For each region in the `.var` attribute of the AnnData object, we
    either 1) map it to a region in the universe, or 2) map it to `None`.
    If it is mapped to `None`, it is not in the universe and will be dropped
    from the AnnData object. If it is mapped to a region, it will be updated
    to the region in the universe for downstream analysis.
    """
    # ensure adata has chr, start, and end
    if not all([x in adata.var.columns for x in ["chr", "start", "end"]]):
        raise ValueError(
            "AnnData object must have `chr`, `start`, and `end` columns in .var"
        )
    universe_set = [f"{region[0]}_{region[1]}_{region[2]}" for region in universe_set]

    # create list of regions from adata
    query_set: List[str] = adata.var.apply(
        lambda x: f"{x['chr']}_{x['start']}_{x['end']}", axis=1
    ).tolist()

    # generate conversion map
    _map = generate_var_conversion_map(query_set, universe_set)

    # create a new DataFrame with the updated values
    updated_var = adata.var.copy()

    # find the regions that overlap with the universe
    # use dynamic programming to create a boolean mask of columns to keep
    columns_to_keep = []
    for i, row in tqdm(adata.var.iterrows(), total=adata.var.shape[0]):
        region = f"{row['chr']}_{row['start']}_{row['end']}"
        if _map[region] is None:
            columns_to_keep.append(False)
            continue

        # if it is, change the region to the universe region,
        # grab the first for now
        # TODO - this is a simplification, we should be able to handle multiple
        universe_region = _map[region]
        chr, start, end = universe_region.split("_")

        updated_var.at[i, "chr"] = chr
        updated_var.at[i, "start"] = start
        updated_var.at[i, "end"] = end

        columns_to_keep.append(True)

    # update adata with the new DataFrame and filtered columns
    adata = adata[:, columns_to_keep]
    adata.var = updated_var[columns_to_keep]

    return adata


def anndata_to_regionsets(adata: sc.AnnData) -> List[List[str]]:
    """
    Converts an AnnData object to a list of lists of regions. This
    is done by taking each cell and creating a list of all regions
    that have a value greater than 0.

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
    for i in tqdm(range(adata.shape[0]), total=adata.shape[0]):
        regions.append(
            [
                f"{chr_values[j]}_{start_values[j]}_{end_values[j]}"
                for j in np.where(positive_values[i])[0]
            ]
        )
    return regions
