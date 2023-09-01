from time import time
from typing import List

import numpy as np
import pyBigWig

from .io.io import Region


def wordify_region(region: Region):
    return f"{region.chr}_{region.start}_{region.end}"


def unwordify_region(word: str):
    chr, start, end = word.split("_")
    return Region(chr, int(start), int(end))


def wordify_regions(regions: List[Region]):
    return [wordify_region(r) for r in regions]


def unwordify_regions(words: List[str]):
    return [unwordify_region(w) for w in words]


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
