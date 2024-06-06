import os
from glob import glob
from math import ceil
from typing import List, Tuple

import torch
from genimtools.utils import read_tokens_from_gtok
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .const import KEEP_RATE, MASK_RATE, REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE


class AtacformerMLMCollator:
    """
    Collator for the MLM dataset. This will pad the tokens, masked_tokens, and mask_ids
    """

    def __init__(self, padding_token: int, context_size: int):
        self.padding_token = padding_token
        self.context_size = context_size

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

        attention_mask = tokens != self.padding_token

        # clip the tokens to the context size
        # actually, we should randomly sample a subset of the tokens
        tokens = tokens[:, : self.context_size]
        masked_tokens = masked_tokens[:, : self.context_size]
        mask_ids = mask_ids[:, : self.context_size]
        attention_mask = attention_mask[:, : self.context_size]

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

        # get list of all files
        self.files = glob(os.path.join(data, "**/*.gtok"))
        self.probs = torch.tensor([REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE, KEEP_RATE])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This should return a tuple of (tokens, masked_tokens, mask_ids).
        """
        # load the data into memory
        tokens = torch.tensor(read_tokens_from_gtok(self.files[idx]))
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
