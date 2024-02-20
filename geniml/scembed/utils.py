import logging
import os
from glob import glob
from typing import Tuple, List

import scanpy as sc

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from rich.progress import track
from genimtools.utils import read_tokens_from_gtok

from .const import DEFAULT_CHUNK_SIZE

_LOGGER = logging.getLogger(__name__)


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


class BatchCorrectionDataset(Dataset):
    def __init__(self, batches: list, backed: bool = True):
        """
        Dataset for batch correction. This dataset takes in pre-tokenized
        cells and their batch of origin and then yields them out for training.

        :param batches list: a list of paths that point to pre-tokenized cells (.gtok files).
        :param bool backed: Whether to load the data in backed mode. If True, the data will be loaded in backed mode. If False, the data will be loaded into memory.
        """
        self.num_batches = len(batches)
        self.backed = backed

        # create tuples of (gtok_file, batch)
        self.data = []
        for i, batch in enumerate(batches):
            for gtok_file in glob(os.path.join(batch, "*.gtok")):
                self.data.append((gtok_file, i))

        if not self.backed:
            _LOGGER.info("Loading data into memory...")
            self.data = [
                (torch.tensor(read_tokens_from_gtok(gtok_file)), torch.tensor(batch))
                for gtok_file, batch in track(
                    self.data, total=len(self.data), description="Loading data into memory..."
                )
            ]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.

        :param idx: The index of the item to get.
        """
        if not self.backed:
            return self.data[idx]
        else:
            gtok_file, batch = self.data[idx]
            tokens = read_tokens_from_gtok(gtok_file)
            return torch.tensor(tokens), torch.tensor(batch)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (
            f"<BatchCorrectionDataset: {len(self.data)} samples, and {self.num_batches} batches>"
        )


class BCBatchCollator:
    def __init__(self, pad_value: int = 0):
        self.pad_value = pad_value

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for the batch correction dataset. This function
        takes in a list of tuples of (tokens, batch) and returns a tuple of
        (padded_tokens, batches).

        :param batch: A list of tuples of (tokens, batch)
        """
        tokens, batches = zip(*batch)
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.pad_value)
        batches = torch.stack(batches)
        return tokens, batches
