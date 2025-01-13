import os
from glob import glob
from math import ceil
from typing import List, Tuple

import torch
from gtars.utils import read_tokens_from_gtok
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .const import KEEP_RATE, MASK_RATE, REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE


class AtacformerMLMCollator:
    """
    Collator for the MLM dataset. This will pad the tokens, masked_tokens, and labels.
    """

    def __init__(self, padding_token: int, ignore_index: int = -100):
        self.padding_token = padding_token
        self.ignore_index = ignore_index

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function for the MLM dataset. Pads tokens, masked_tokens, and labels.

        :param batch: List of tuples (tokens, masked_tokens, labels)
        :return: Tuple of (tokens, masked_tokens, labels, attention_mask)
        """
        tokens, masked_tokens, labels = zip(*batch)

        # Pad the sequences
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.padding_token)
        masked_tokens = pad_sequence(
            masked_tokens, batch_first=True, padding_value=self.padding_token
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

        # Create attention mask
        attention_mask = (tokens != self.padding_token).float()

        return tokens, masked_tokens, labels, attention_mask


class AtacformerMLMDataset(Dataset):
    def __init__(
        self,
        data: str,
        mask_token_id: int,
        vocab_size: int,
        mask_rate: float = MASK_RATE,
        random_seed: int = 42,
        shuffle: bool = True,
        context_size: int = 2048,
        seed: int = 42,
    ):
        """
        Initialize the MLM dataset. This is heavily based on the MLM dataset
        proposed in the original BERT paper (https://arxiv.org/abs/1810.04805)

        :param str data: Path to the dataset. This should be a file of .gtok files
        :param int mask_token_id: ID of the mask token
        :param float mask_rate: Probability of masking a token
        :param int vocab_size: Size of the vocabulary
        :param int random_seed: Random seed to use
        :param bool shuffle: Whether to shuffle the data
        """
        self.data = data
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.context_size = context_size

        # get list of all files
        self.files = glob(os.path.join(data, "*.gtok"), recursive=True)
        if len(self.files) == 0:
            # try recursive
            self.files = glob(os.path.join(data, "**/*.gtok"), recursive=True)

        self.probs = torch.tensor([REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE, KEEP_RATE])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: Original token IDs
            masked_tokens: Token IDs with some tokens masked/replaced
            labels: Original token IDs for masked positions and -100 for others
        """
        # Load the data into memory
        tokens = torch.tensor(read_tokens_from_gtok(self.files[idx]), dtype=torch.long)

        # Reduce the tokens to the context size
        if tokens.shape[0] > self.context_size:
            indices = torch.multinomial(
                torch.ones(tokens.shape[0]), self.context_size, replacement=False
            )
            tokens = tokens[indices]

        masked_tokens = tokens.clone()
        labels = torch.full_like(tokens, -100)  # Initialize labels with -100

        # Determine the number of tokens to mask
        num_mask = ceil(tokens.shape[0] * self.mask_rate)
        if num_mask == 0:
            return tokens, masked_tokens, labels  # No masking needed

        # select unique positions to mask
        mask_ids = torch.multinomial(torch.ones(tokens.shape[0]), num_mask, replacement=False)

        # assign labels for the masked positions
        labels[mask_ids] = tokens[mask_ids]

        # decide how to mask the selected positions
        random_vals = torch.multinomial(self.probs, num_mask, replacement=True)

        # indices where [MASK] will be used
        # this is where random_vals == 0
        mask_token_indices = mask_ids[random_vals == 0]

        # indices where a random token will replace
        # the original token, this is where random_vals == 1
        replace_random_indices = mask_ids[random_vals == 1]

        # indices where the original token is kept (no action needed)
        # this is where random_vals == 2
        # pass

        # perform the masking
        masked_tokens[mask_token_indices] = self.mask_token_id
        masked_tokens[replace_random_indices] = torch.randint(
            low=0, high=self.vocab_size, size=(replace_random_indices.shape[0],), dtype=torch.long
        )

        return tokens, masked_tokens, labels

    def __str__(self):
        return f"AtacformerMLMDataset({len(self)} files)"

    def __repr__(self):
        return f"AtacformerMLMDataset({len(self)} files)"


class AtacformerCellTypeFineTuningCollator:
    """
    Collator for the cell type fine-tuning dataset. This will pad the tokens, labels, and attention_mask
    """

    def __init__(self, padding_token: int):
        self.padding_token = padding_token

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function for the cell type fine-tuning dataset. This should take a batch of
        (tokens, labels) and return a tuple of (tokens, labels) that are padded

        :param list[tuple[torch.Tensor, torch.Tensor]] batch: Batch of (tokens, labels)
        :param int padding_token: Token to use for padding
        """
        cell1, cell2, labels = zip(*batch)

        # pad the tokens
        cell1 = pad_sequence(cell1, batch_first=True, padding_value=self.padding_token)
        cell2 = pad_sequence(cell2, batch_first=True, padding_value=self.padding_token)

        attention_mask1 = (cell1 != self.padding_token).float()
        attention_mask2 = (cell2 != self.padding_token).float()

        labels = torch.tensor(labels)

        return cell1, cell2, labels, attention_mask1, attention_mask2


class AtacformerCellTypeFineTuningDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        context_size: int = 2048,
        seed: int = 42,
    ):
        """
        Initialize the cell type fine-tuning dataset.

        :param str file_path: Path to the file that defines the dataset
        """
        # check file and not directory
        if os.path.isdir(file_path):
            raise ValueError(f"Expected a file, got a directory: {file_path}")

        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")

        # init params
        self.file_path = file_path
        self.root_dir = os.path.dirname(self.file_path)
        self.context_size = context_size
        self.seed = seed

        # read the file, line by line
        # format is a\tb\label
        self.pairs = []
        with open(file_path, "r") as f:
            for line in f:
                barcode1, barcode2, label, cell_type1, cell_type2 = line.strip().split("\t")
                barcode1_path = os.path.join(self.root_dir, barcode1)
                barcode2_path = os.path.join(self.root_dir, barcode2)
                self.pairs.append((barcode1_path, barcode2_path, int(label)))

        self.max_label = max([pair[2] for pair in self.pairs])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This should return a tuple of (tokens, label).
        """
        barcode1_path, barcode2_path, label = self.pairs[idx]

        # load the data into memory
        cell1 = torch.tensor(read_tokens_from_gtok(barcode1_path + ".gtok"))
        cell2 = torch.tensor(read_tokens_from_gtok(barcode2_path + ".gtok"))

        # reduce the tokens to the context size
        # randomly sample self.context_size tokens from the tokens
        if cell1.shape[0] > self.context_size:
            indices = torch.multinomial(
                torch.ones(cell1.shape[0]), self.context_size, replacement=False
            )
            cell1 = cell1[indices]

        if cell2.shape[0] > self.context_size:
            indices = torch.multinomial(
                torch.ones(cell2.shape[0]), self.context_size, replacement=False
            )
            cell2 = cell2[indices]

        label = torch.tensor(label)

        return cell1, cell2, label

    def __str__(self):
        return f"AtacformerCellTypeFineTuningDataset({len(self)} files)"

    def __repr__(self):
        return f"AtacformerCellTypeFineTuningDataset({len(self)} files)"
