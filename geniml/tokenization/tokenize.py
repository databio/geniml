import numpy as np
import scanpy as sc
from tqdm import tqdm
from gtars.tokenizers import Tokenizer
from gtars.models import Region


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
    x = adata.X
    for row in tqdm(
        range(adata.shape[0]),
        total=adata.shape[0],
        desc="Tokenizing",
    ):
        _, non_zeros = x[row].nonzero()
        regions = features[non_zeros]
        tokenized.append(tokenizer(regions))

    return tokenized
