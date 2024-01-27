import os
import random
from glob import glob

import torch
from torch.utils.data import Dataset
from genimtools.utils import read_tokens_from_gtok

from .const import MASK_RATE, REPLACE_WITH_MASK_RATE, REPLACE_WITH_RANDOM_RATE, KEEP_RATE


class AtacformerMLMDataset(Dataset):
    def __init__(
        self,
        data: str,
        mask_token_id: int,
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
        :param int random_seed: Random seed to use
        :param bool shuffle: Whether to shuffle the data
        """
        self.data = data
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.random_seed = random_seed
        self.shuffle = shuffle

        # get list of all files
        self.files = glob(os.path.join(data, "*.gtok"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        This should return a tuple of (input_ids, labels).
        """
        # load the data
        tokens = torch.tensor(read_tokens_from_gtok(self.files[idx]))
        labels = torch.tensor(tokens)

        # shuffle the data if necessary
        if self.shuffle:
            random.shuffle(tokens)

        # get the number of tokens to predict
        num_to_predict = int(self.mask_prob * len(tokens))

        # get the indices of the tokens to predict
        idxs = random.sample(range(len(tokens)), num_to_predict)

        # get the input ids
        for i in idxs:
            prob = random.random()
            # mask the token
            if prob < REPLACE_WITH_MASK_RATE:
                tokens[i] = self.mask_token_id
            # replace the token with a random token
            elif prob < REPLACE_WITH_RANDOM_RATE:
                tokens[i] = random.choice(tokens)
            # keep the token
            else:
                pass

        return tokens, labels
