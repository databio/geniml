import scanpy as sc
from time import time


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
    return cove.astype(np.uint16)


class AnnDataChunker:
    def __init__(self, adata: sc.AnnData, chunk_size: int = 10000):
        """
        Simple class to chunk an AnnData object into smaller pieces. Useful for
        training on large datasets.

        :param adata: AnnData object to chunk. Must be in backed mode. See: https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html
        :param chunk_size: Number of cells to include in each chunk
        """
        self.adata = adata
        self.chunk_size = chunk_size
        self.n_chunks = len(adata) // chunk_size + 1

    def __iter__(self):
        for i in range(self.n_chunks):
            yield self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :]

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, item):
        return self.adata[item * self.chunk_size : (item + 1) * self.chunk_size, :]

    def __repr__(self):
        return f"<AnnDataChunker: {self.n_chunks} chunks of size {self.chunk_size}>"
