import time

import numpy as np
import scanpy as sc
from tqdm import tqdm
from gtars.tokenizers import Tokenizer

from ..io import Region


class Timer:
    """Records the running time.

    Uses Timer.s() or Timer() to record the start time. Then, calls Timer.t() to get the
    elapsed time in seconds.
    """

    def __init__(self):
        """Initializes a Timer object and starts the timer."""
        self.v = time.time()

    def s(self):
        """Restarts the timer."""
        self.v = time.time()

    def t(self):
        """Gives the elapsed time.

        Returns:
            float: The elapsed time in seconds.
        """
        return time.time() - self.v


def time_str(t: float) -> str:
    """Converts time in float to a readable format.

    Converts time in float to hours, minutes, or seconds based on the value of
    t.

    Args:
        t (float): Time in seconds.

    Returns:
        str: Time in readable time.
    """
    if t >= 3600:
        return f"{t / 3600:.2f}h"
    if t >= 60:
        return f"{t / 60:.2f}m"
    return f"{t:.2f}s"


def tokenize_anndata(adata: sc.AnnData, tokenizer: Tokenizer):
    """
    Tokenize an AnnData object. This is more involved, so it gets its own function.

    Args:
        adata (sc.AnnData): The AnnData object to tokenize.
        tokenizer (Tokenizer): The tokenizer to use.
    """
    # extract regions from AnnData
    # its weird because of how numpy handle Intervals, the parent class of Region,
    # see here:
    # https://stackoverflow.com/a/43722306/13175187
    adata_features = [
        Region(chr, int(start), int(end))
        for chr, start, end in tqdm(
            zip(adata.var["chr"], adata.var["start"], adata.var["end"]),
            total=adata.var.shape[0],
            desc="Extracting regions from AnnData",
        )
    ]
    features = np.ndarray(len(adata_features), dtype=object)
    for i, region in enumerate(adata_features):
        features[i] = region
    del adata_features

    # tokenize
    tokenized = []
    for row in tqdm(
        range(adata.shape[0]),
        total=adata.shape[0],
        desc="Tokenizing",
    ):
        _, non_zeros = adata.X[row].nonzero()
        regions = features[non_zeros]
        tokenized.append(tokenizer(regions))

    return tokenized
