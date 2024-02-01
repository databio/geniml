import os
from glob import glob
from math import ceil

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from genimtools.utils import read_tokens_from_gtok

from .const import MASK_RATE, REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE, KEEP_RATE


class AtacformerMLMDataset(Dataset):
    def __init__(
        self,
        data: str,
        mask_token_id: int,
        vocab_size: int,
        mask_prob: float = MASK_RATE,
        random_seed: int = 42,
        shuffle: bool = True,
    ):
        """
        Initialize the MLM dataset. This is heavily based on the MLM dataset
        proposed in the original BERT paper (https://arxiv.org/abs/1810.04805)

        :param str data: Path to the dataset. This should be a file of .gtok files
        :param int mask_token_id: ID of the mask token
        :param float mask_prob: Probability of masking a token
        :param int vocab_size: Size of the vocabulary
        :param int random_seed: Random seed to use
        :param bool shuffle: Whether to shuffle the data
        """
        self.data = data
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.random_seed = random_seed
        self.shuffle = shuffle

        # get list of all files
        self.files = glob(os.path.join(data, "*.gtok"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This should return a tuple of (tokens, masked_tokens, mask_ids).
        """
        # load the data
        tokens = torch.tensor(read_tokens_from_gtok(self.files[idx]))
        masked_tokens = tokens.clone()

        # get the mask ids (select tokens.shape[0] * self.mask_prob tokens to mask)
        mask_ids = torch.multinomial(
            torch.ones(tokens.shape[0]), ceil(tokens.shape[0] * 0.15), replacement=False
        )

        # mask the tokens
        for i in mask_ids:
            val = torch.multinomial(
                torch.tensor([REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE, KEEP_RATE]), 1
            )
            if val == 0:
                masked_tokens[i] = self.mask_token_id
            elif val == 1:
                masked_tokens[i] = torch.randint(self.vocab_size, (1,))
            else:
                pass  # do nothing, keep the original token

        return tokens, masked_tokens, mask_ids


def mlm_batch_collator(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], padding_token: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for the MLM dataset. This should take a batch of
    (tokens, masked_tokens, mask_ids) and return a tuple of (tokens, masked_tokens, mask_ids) that are padded

    :param list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] batch: Batch of (tokens, masked_tokens, mask_ids)
    :param int padding_token: Token to use for padding
    """
    tokens, masked_tokens, mask_ids = zip(*batch)

    # pad the tokens
    tokens = pad_sequence(tokens, batch_first=True, padding_value=padding_token)
    masked_tokens = pad_sequence(masked_tokens, batch_first=True, padding_value=padding_token)
    mask_ids = pad_sequence(mask_ids, batch_first=True, padding_value=padding_token)

    return tokens, masked_tokens, mask_ids
