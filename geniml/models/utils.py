import os
from typing import TYPE_CHECKING, Dict, List, Union, Tuple

import numpy as np
import scanpy as sc
from tqdm import tqdm

if TYPE_CHECKING:
    from .atac.tokenization import Universe

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
    regions: Union[str, List[str], List[Tuple[str]]]
) -> List[Tuple[str, int, int]]:
    """
    Validate the input for the regions. this universe accepts a lot of forms of input,
    so we need to check what the user has passed. It also standardizes the input to a list
    of tuples of chr, start, end.

    Users are allowed to pass a list of regions, a list of tuples with chr, start, and end, or
    a path to a BED file.

    :param regions: The regions to use for tokenization. This can be thought of as a vocabulary.
                    This can be either a list of regions or a path to a BED file containing regions.
    :return list: A list of tuples of chr, start, end.
    """
    # check for bedfile
    if isinstance(regions, str):
        # ensure that the file exists
        if not os.path.exists(regions):
            raise FileNotFoundError(f"Could not find file {regions} containing regions.")
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
    a: List[Tuple[str, int, int]],
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

    for region in tqdm(a, total=len(a), desc="Generating conversion map"):
        overlaps = b.query(region)
        region_str = f"{region[0]}_{region[1]}_{region[2]}"
        if len(overlaps) > 0:
            olap = overlaps[0]  # take the first overlap for now, we can change this later
            olap_str = f"{olap[0]}_{olap[1]}_{olap[2]}"
            conversion_map[region_str] = olap_str
        else:
            conversion_map[region_str] = None

    return conversion_map


def convert_to_universe(adata: sc.AnnData, universe: "Universe") -> sc.AnnData:
    """
    Convert an AnnData object to a new universe. This is useful for converting the consensus
    peak set of one AnnData object to another. This is usually used as a preprocessing step
    before tokenization.

    This function is already pretty optimized. To speed it up even more, we could use
    multiprocessing to parallelize the conversion. Or, we could use Rust library to speed
    up the conversion. However, this is not necessary at the moment.

    :param adata: the AnnData object to convert
    :param Universe universe: the universe to convert to
    :return: the converted AnnData object
    """
    # ensure adata has chr, start, and end
    if not all([x in adata.var.columns for x in ["chr", "start", "end"]]):
        raise ValueError("AnnData object must have `chr`, `start`, and `end` columns in .var")

    # create list of regions from adata
    adata.var["region"] = (
        adata.var["chr"].astype(str)
        + "_"
        + adata.var["start"].astype(str)
        + "_"
        + adata.var["end"].astype(str)
    )
    query_set: List[tuple[str, int, int]] = list(
        zip(adata.var["chr"], adata.var["start"].astype(int), adata.var["end"].astype(int))
    )

    # generate conversion map
    _map = generate_var_conversion_map(query_set, universe)

    # map regions to new universe
    adata.var["new_region"] = adata.var["region"].map(_map)

    # drop rows where new_region is None
    adata.var.dropna(subset=["new_region"], inplace=True)

    # split new_region into chr, start, end
    adata.var[["chr", "start", "end"]] = adata.var["new_region"].str.split("_", expand=True)

    # drop 'region' and 'new_region' columns
    adata.var.drop(columns=["region", "new_region"], inplace=True)

    # update adata with the new DataFrame
    adata = adata[:, adata.var.index]

    return adata


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
                f"{chr_values[j]}_{start_values[j]}_{end_values[j]}"
                for j in np.where(positive_values[i])[0]
            ]
        )
    return regions
