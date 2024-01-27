import torch.nn as nn

from .const import DEFAULT_HIDDEN_SIZE


class Atacformer(nn.Module):
    def __init__(self, vocab_size, hidden_size: int = DEFAULT_HIDDEN_SIZE):
        """
        Atacformer is a transformer-based model for ATAC-seq data. It closely follows
        the architecture of BERT, but with a few modifications:
        - positional embeddings set to 0 since ATAC-seq data is not sequential
        - no next sentence prediction task
        """
        pass
