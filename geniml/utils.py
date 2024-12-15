from time import time
from typing import Dict, List, Optional

import numpy as np
import pyBigWig


def natural_chr_sort(a, b):
    ac = a.replace("chr", "")
    ac = ac.split("_")[0]
    bc = b.replace("chr", "")
    bc = bc.split("_")[0]
    if bc.isnumeric() and ac.isnumeric() and bc != ac:
        if int(bc) < int(ac):
            return 1
        elif int(bc) > int(ac):
            return -1
        else:
            return 0
    else:
        if b < a:
            return 1
        elif a < b:
            return -1
        else:
            return 0


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1)/60:.4f}min")
        return result

    return wrap_func


def read_chromosome_from_bw(file, chrom):
    bw = pyBigWig.open(file)
    chrom_size = bw.chroms(chrom)
    if pyBigWig.numpy:
        cove = bw.values(chrom, 0, chrom_size, numpy=True)
    else:
        cove = bw.values(chrom, 0, chrom_size)
        cove = np.array(cove)
    cove[np.isnan(cove)] = 0
    return cove.astype(np.uint16)


def find_path(hierarchy: Dict[str, Dict], path: List[str], cell_type: str) -> Optional[List[str]]:
    """
    Find the path from the root to a given cell type in a hierarchy.

    :param hierarchy: A dictionary representing the hierarchy.
    :param path: The current path.
    :param cell_type: The cell type to find.

    :return: The path from the root to the cell type. (a list of strings, ... or None)
    """
    if cell_type in hierarchy:
        return path + [cell_type]

    for key in hierarchy:
        sub_path = find_path(hierarchy[key], path + [key], cell_type)
        if sub_path:
            return sub_path

    return None


def find_lca(path1: List[str], path2: List[str]) -> int:
    """
    Find the lowest common ancestor (LCA) of two paths.

    :param path1: The first path.
    :param path2: The second path.
    """
    min_length = min(len(path1), len(path2))
    for i in range(min_length):
        if path1[i] != path2[i]:
            return i - 1
    return min_length - 1


def compute_cell_hierarchy_distance(
    hierarchy: Dict[str, Dict], cell1: str, cell2: str
) -> Optional[int]:
    """
    Compute the distance between two cell types in a hierarchy.

    The distance is the number of edges between the two cells in the hierarchy.

    :param hierarchy: A dictionary representing the hierarchy.
    :param cell1: The first cell type.
    :param cell2: The second cell type.

    :return: The distance between the two cell types. (an integer, ... or None)
    """
    # Find paths from root to both cells
    path1 = find_path(hierarchy, [], cell1)
    path2 = find_path(hierarchy, [], cell2)

    if not path1 or not path2:
        return None  # One of the cells doesn't exist in the hierarchy

    # Find the lowest common ancestor (LCA)
    lca_index = find_lca(path1, path2)

    # Distance is the sum of the lengths from LCA to both nodes
    distance = (len(path1) - lca_index - 1) + (len(path2) - lca_index - 1)

    return distance
