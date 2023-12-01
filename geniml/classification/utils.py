from typing import Tuple, List

import torch
import scanpy as sc

from rich.progress import track
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..tokenization.main import ITTokenizer


def collate_classification_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], pad_token: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a batch of data into a single tensor.

    :param batch: A batch of data.
    :return: A tuple of tensors.
    """
    tokens, labels = zip(*batch)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=pad_token)

    labels = torch.tensor(labels, dtype=torch.long)

    return tokens_padded, labels


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

    labels = torch.tensor(labels, dtype=torch.float)

    return pairs_padded, labels


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
        self.pairs = pairs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])), torch.tensor(
            self.labels[idx]
        )


class TrainingResult(BaseModel):
    """
    Results of a region2VecClassification training run.

    :param List[float] validation_loss: The validation loss for each epoch.
    :param List[float] epoch_loss: The training loss for each epoch.
    :param List[float] all_loss: The training loss for each batch.
    """

    validation_loss: List[float]
    epoch_loss: List[float]
    all_loss: List[float]
    training_accuracy: float
    validation_accuracy: float


class FineTuneTrainingResult(BaseModel):
    """
    Results of a region2VecClassification training run.

    :param List[float] validation_loss: The validation loss for each epoch.
    :param List[float] epoch_loss: The training loss for each epoch.
    :param List[float] all_loss: The training loss for each batch.
    """

    validation_loss: List[float]
    epoch_loss: List[float]
    all_loss: List[float]


def generate_fine_tuning_dataset(
    adata: sc.AnnData,
    tokenizer: ITTokenizer,
    cell_type_key: str = "cell_type",
    seed: int = 42,
    negative_ratio: float = 1.0,
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

        # generate number of positive and negative pairs
        # shaves off the remainder if odd, will never be more than 1,
        # so worst case is 1 pair lost
        npos_pairs = int(len(adata_ct) - len(adata_ct) % 2)
        nneg_pairs = int((npos_pairs * negative_ratio) - (npos_pairs * negative_ratio) % 2)

        # generate random permutations of the indices
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            pos_indices = torch.randperm(npos_pairs).tolist()
            neg_indices = torch.randperm(nneg_pairs).tolist()

        # generate pairs
        pos_pairs = torch.tensor(pos_indices).reshape(-1, 2).tolist()
        neg_pairs = torch.tensor(neg_indices).reshape(-1, 2).tolist()

        # extend the positive and negative pairs (dont convert to tensor yet)
        for ppair, npair in zip(pos_pairs, neg_pairs):
            positive_pairs.append(
                (adata_ct.obs["tokens"].iloc[ppair[0]], adata_ct.obs["tokens"].iloc[ppair[1]])
            )
            # notice that we are appending one from the cell type and one from the not cell type
            # this ensures that the negative pairs are actually negative, otherwise
            # its possible that the negative pairs are actually positive (we happen to select two cells from the same cell type)
            negative_pairs.append(
                (
                    adata_not_ct.obs["tokens"].iloc[npair[0]],
                    adata_ct.obs["tokens"].iloc[npair[1]],
                )
            )

    return positive_pairs, negative_pairs, [1] * len(positive_pairs), [-1] * len(negative_pairs)
