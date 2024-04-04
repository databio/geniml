import contextlib
import random
from random import shuffle
from typing import List, Tuple

import scanpy as sc
import torch
from rich.progress import track
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..tokenization.main import ITTokenizer


@contextlib.contextmanager
def tempseed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def collate_finetuning_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], pad_token: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a batch of data into a single tensor.

    :param batch: A batch of data.
    :return: A tuple of tensors.
    """
    pairs, labels = zip(*batch)
    firsts = [p[0] for p in pairs]
    seconds = [p[1] for p in pairs]

    firsts_padded = pad_sequence(firsts, batch_first=True, padding_value=pad_token)
    seconds_padded = pad_sequence(seconds, batch_first=True, padding_value=pad_token)

    pairs_padded = (firsts_padded, seconds_padded)

    return pairs_padded, torch.tensor(labels)


class SingleCellClassificationDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, labels: torch.Tensor):
        """
        Initialize the dataset.

        :param sc.AnnData data: The data to use for training.
        :param str label_key: The key in the obs to use for the labels.
        :param float train_test_split: The fraction of the data to use for training.
        """
        self.tokens = tokens
        self.labels = labels

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.tokens[idx]), torch.tensor(self.labels[idx])


class FineTuningDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[List[int], List[int]]],
        labels: List[int],
    ):
        """
        Initialize the dataset.

        :param List[Tuple[List[int], List[int]]] pairs: The pairs - each pair is a tuple of two lists of tokens.
        :param List[int] labels: The labels for each pair (0 or 1).
        """
        self.pairs = [(torch.tensor(p[0]), torch.tensor(p[1])) for p in pairs]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self.pairs[idx], self.labels[idx]


def generate_fine_tuning_dataset(
    adata: sc.AnnData,
    tokenizer: ITTokenizer,
    cell_type_key: str = "cell_type",
    seed: int = 42,
    sample_size: int = None,
    negative_ratio: float = None,
) -> Tuple[
    List[Tuple[List[int], List[int]]], List[Tuple[List[int], List[int]]], List[int], List[int]
]:
    """
    Generates a dataset for fine tuning the region2vec model using siamese networks. These
    networks require a datasets of pairs. Specifically, pairs of cells (a region set). The
    pairs are labeled as either the same cell type or different cell types. These correspond
    to positive and negative examples, respectively.

    This function will take in an AnnData object and generate a dataset of pairs of cells
    and their entanglement labels.

    :param sc.AnnData adata: The AnnData object to use for generating the dataset.
    :param ITTokenizer tokenizer: The tokenizer to use for tokenizing the regions.
    :param str cell_type_key: The key in the obs that contains the cell type labels.
    :param int seed: The seed to use for generating the pairs.
    :param int sample_size: The number of pairs to sample. If None, all possible pairs will be used.
    :param float negative_ratio: The ratio of negative pairs to positive pairs.
    :return: A tuple of positive pairs, negative pairs, positive labels, and negative labels.
    """
    # generate positive pairs
    if cell_type_key not in adata.obs:
        raise ValueError(f"Cell type key {cell_type_key} not found in obs.")

    cell_types = adata.obs[cell_type_key].unique()

    # tokenize every cell first, so we don't have to do it multiple times
    tokens = tokenizer.tokenize(adata)
    tokens = [
        [t.id for t in subset]
        for subset in track(tokens, description="Converting to ids", total=len(tokens))
    ]
    adata.obs["tokens"] = tokens

    positive_pairs = []
    negative_pairs = []

    for ct in track(cell_types, description="Generating pairs", total=len(cell_types)):
        adata_ct = adata[adata.obs[cell_type_key] == ct]
        adata_not_ct = adata[adata.obs[cell_type_key] != ct]

        # generate pairwise combinations of the cells
        pos_indexes = list(range(len(adata_ct)))
        neg_indexes = list(range(len(adata_not_ct)))

        if sample_size:
            # shuffle then take the first n pairs
            with tempseed(seed):
                shuffle(pos_indexes)
                shuffle(neg_indexes)
                pos_indexes = pos_indexes[:sample_size]
                neg_indexes = neg_indexes[:sample_size]

        # positive pair generation
        pos = [
            (adata_ct.obs["tokens"].iloc[i], adata_ct.obs["tokens"].iloc[j])
            for i in pos_indexes
            for j in pos_indexes
            if i != j  # don't include the same cell
        ]
        positive_pairs.extend(pos)

        # negative pair generation
        neg = [
            (adata_ct.obs["tokens"].iloc[i], adata_not_ct.obs["tokens"].iloc[j])
            for i in pos_indexes
            for j in neg_indexes
        ]

        if negative_ratio:
            # shuffle then take the first n pairs
            with tempseed(seed):
                shuffle(neg)
                neg = neg[: int(len(pos) * negative_ratio)]

        negative_pairs.extend(neg)

    return positive_pairs, negative_pairs, [1] * len(positive_pairs), [-1] * len(negative_pairs)
