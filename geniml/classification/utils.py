from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_batch(
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

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx]), torch.tensor(self.labels[idx])
