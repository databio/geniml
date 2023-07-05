import os
import subprocess
from typing import List, Dict, Union

import numpy as np
import scanpy as sc
from tqdm import tqdm

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


def generate_var_conversion_map(
    a: List[str], b: List[str], path_to_bedtools: str = None, fraction: float = 1.0e-9
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

    :param List[str] a: the first list of regions
    :param List[str] b: the second list of regions
    :param str path_to_bedtools: the path to the bedtools executable
    :param float fraction: the fraction of the region that must overlap to be considered an overlap
    """
    # write a and b to temp files in cache
    a_file = os.path.join(get_cache_dir(), "a.bed")
    b_file = os.path.join(get_cache_dir(), "b.bed")

    # write a and b to temp files
    with open(a_file, "w") as f:
        # split each region into chr start end
        a_parsed = [region.split("_") for region in a]
        a_parsed = [f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in a_parsed]
        f.writelines(a_parsed)
    with open(b_file, "w") as f:
        b_parsed = [region.split("_") for region in b]
        b_parsed = [f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in b_parsed]
        f.writelines(b_parsed)

    # sort both files
    cmd = f"sort -k1,1 -k2,2n {a_file} -o {a_file}"
    subprocess.run(cmd, shell=True)

    cmd = f"sort -k1,1 -k2,2n {b_file} -o {b_file}"
    subprocess.run(cmd, shell=True)

    # run bedtools
    bedtools_cmd = f"intersect -a {a_file} -b {b_file} -wa -wb -f {fraction}"

    # add path to bedtools if provided
    if path_to_bedtools is not None:
        cmd = f"{path_to_bedtools} {bedtools_cmd}"
    else:
        cmd = f"bedtools {bedtools_cmd}"

    # target file
    target_file = os.path.join(get_cache_dir(), "olaps.bed")
    with open(target_file, "w") as f:
        subprocess.run(cmd, shell=True, stdout=f)

    # bedtools reports overlaps like this:
    # chr1 100 200 chr1 150 250
    # we want to convert this to a map like this:
    # {chr1_100_200: chr1_150_250}
    # we will use a dictionary to do this
    # if a region in A overlaps with multiple regions in B, we will
    # take the first one. as such we need to check if a region in A
    # has already been mapped to a region in B
    conversion_map = dict()
    with open(target_file, "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            a_region = f"{line[0]}_{line[1]}_{line[2]}"
            b_region = f"{line[3]}_{line[4]}_{line[5]}"
            if a_region not in conversion_map:
                conversion_map[a_region] = b_region

    # add `None` mappings for regions in A that did not overlap with any regions in B
    for region in a:
        if region not in conversion_map:
            conversion_map[region] = None

    # remove temp files
    os.remove(a_file)
    os.remove(b_file)
    os.remove(target_file)

    return conversion_map


def convert_to_universe(adata: sc.AnnData, universe_file: str) -> sc.AnnData:
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

    # read in universe from file into universe_set
    universe_set = []
    with open(universe_file, "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            universe_set.append(f"{line[0]}_{line[1]}_{line[2]}")

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
