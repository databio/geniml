from time import time
import pyBigWig
import numpy as np


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
    return cove.astype(np.uint16), chrom_size
