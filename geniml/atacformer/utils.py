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
    Collator for the MLM dataset. This will pad the tokens, masked_tokens, and mask_ids
    """

    def __init__(self, padding_token: int):
        self.padding_token = padding_token

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function for the MLM dataset. This should take a batch of
        (tokens, masked_tokens, mask_ids) and return a tuple of (tokens, masked_tokens, mask_ids) that are padded

        :param list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] batch: Batch of (tokens, masked_tokens, mask_ids)
        :param int padding_token: Token to use for padding
        """
        tokens, masked_tokens, mask_ids = zip(*batch)

        # pad the tokens
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.padding_token)
        masked_tokens = pad_sequence(
            masked_tokens, batch_first=True, padding_value=self.padding_token
        )
        mask_ids = pad_sequence(mask_ids, batch_first=True, padding_value=self.padding_token)

        attention_mask = (tokens != self.padding_token).float()

        return tokens, masked_tokens, mask_ids, attention_mask


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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This should return a tuple of (tokens, masked_tokens, mask_ids).
        """
        # load the data into memory
        tokens = torch.tensor(read_tokens_from_gtok(self.files[idx]))

        # reduce the tokens to the context size
        # randomly sample self.context_size tokens from the tokens
        # but dont just slice it.... actually just
        # pick self.context_points without replacement
        if tokens.shape[0] > self.context_size:
            indices = torch.multinomial(
                torch.ones(tokens.shape[0]), self.context_size, replacement=False
            )
            tokens = tokens[indices]

        masked_tokens = tokens.clone()

        # select the tokens to mask
        #   -- each token has an **equal probability** of being masked ( i.e. torch.ones(tokens.shape[0]))
        #   -- we sample a certain percentage of tokens to mask (in BERT this is 15%) ( i.e. ceil(tokens.shape[0] * self.mask_rate) )
        #   -- a token can't be masked more than once (replacement=False)
        mask_ids = torch.multinomial(
            torch.ones(tokens.shape[0]), ceil(tokens.shape[0] * self.mask_rate), replacement=False
        )

        # perform the actual masking. there are three possible outcomes:
        #   1. mask the token
        #   2. replace the token with a random token
        #   3. keep the token the same as it was
        # each outcome has a different probability of happening (`REPLACE_WITH_MASK_RATE`, `REPLACE_WITH_RANDOM_RATE`, `KEEP_RATE` represented in `self.probs`)
        #   therefore, we need a decision made for each token selected to be masked. this means we set num_samples = mask_ids.shape[0] (total number of
        #   tokens selected to mask). we need to sample with replacement (replacement=True) because the same outcome can occur multiple times. e.g. multiple
        #   tokens can be replaced with the mask token.
        random_vals = torch.multinomial(self.probs, mask_ids.shape[0], replacement=True)

        # now actually mask the tokens. recall that we defined the distribution as:
        #   `[REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE, KEEP_RATE]`
        #   so if the random value is:
        #   - 0, we *mask the token*,
        #   - 1, we *replace it with a random token*
        #   - 2, we *keep it the same*
        mask_token_indices = mask_ids[random_vals == 0]
        replace_random_indices = mask_ids[random_vals == 1]
        # why is there no need to do anything for random_vals == 2? because we're keeping the token the same...

        # perform the masking
        masked_tokens[mask_token_indices] = self.mask_token_id
        masked_tokens[replace_random_indices] = torch.randint(
            self.vocab_size, (replace_random_indices.shape[0],)
        )

        # when training we need to pass the masked tokens to the model
        # but we also need to know which tokens were masked so we can calculate the loss
        # and we need to know what those tokens were replaced with so we can calculate the loss
        return tokens, masked_tokens, mask_ids

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
        tokens, labels = zip(*batch)

        # pad the tokens
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.padding_token)
        labels = torch.stack(labels)

        attention_mask = (tokens != self.padding_token).float()

        return tokens, labels, attention_mask


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
                barcode1, barcode2, label = line.strip().split("\t")
                barcode1_path = os.path.join(self.root_dir, barcode1)
                barcode2_path = os.path.join(self.root_dir, barcode2)
                self.pairs.append((barcode1_path, barcode2_path, int(label)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This should return a tuple of (tokens, label).
        """
        barcode1_path, barcode2_path, label = self.pairs[idx]

        # load the data into memory
        cell1 = torch.tensor(read_tokens_from_gtok(barcode1_path))
        cell2 = torch.tensor(read_tokens_from_gtok(barcode2_path))

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
