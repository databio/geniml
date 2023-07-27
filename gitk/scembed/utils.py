import scanpy as sc

from .const import DEFAULT_CHUNK_SIZE


class AnnDataChunker:
    def __init__(self, adata: sc.AnnData, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Simple class to chunk an AnnData object into smaller pieces. Useful for
        training on large datasets.

        :param sc.AnnData adata: AnnData object to chunk. Must be in backed mode. See: https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html
        :param int chunk_size: Number of cells to include in each chunk
        """
        self.adata = adata
        self.chunk_size = chunk_size
        self.n_chunks = len(adata) // chunk_size + 1

    def __iter__(self):
        for i in range(self.n_chunks):
            # check for shape = 0
            if self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :].shape[0] == 0:
                return
            yield self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :]

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, item: int):
        """
        Get a chunk of the AnnData object.

        :param int item: The chunk index to get.
        """
        return self.adata[item * self.chunk_size : (item + 1) * self.chunk_size, :]

    def __repr__(self):
        return f"<AnnDataChunker: {self.n_chunks} chunks of size {self.chunk_size}>"
