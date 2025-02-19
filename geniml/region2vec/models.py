try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "Please install Machine Learning dependencies by running 'pip install geniml[ml]'"
    )

from .const import DEFAULT_EMBEDDING_DIM, POOLING_TYPES


class Word2Vec(nn.Module):
    """
    Word2Vec model.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = DEFAULT_EMBEDDING_DIM, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = kwargs.get("padding_idx", None)
        self.projection = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


class Region2Vec(Word2Vec):
    def __init__(self, vocab_size: int, embedding_dim: int = DEFAULT_EMBEDDING_DIM, **kwargs):
        super().__init__(vocab_size, embedding_dim, **kwargs)


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=1)


class RegionSet2Vec(nn.Module):
    def __init__(self, region2vec: Region2Vec, pooling: POOLING_TYPES = "mean"):
        """
        Initialize the RegionSet2Vec. RegionSet2Vec is a wrapper around the Region2Vec model that allows
        pooling over a set of regions. This is useful for classification tasks where the input is a set of
        regions, such as classifying a cell type based on the set of regions that are accessible.

        :param Union[Region2Vec, str] region2vec: Either a Region2Vec instance or a path to a huggingface model.
        :param POOLING_TYPES pooling: The pooling type to use. Either "mean" or "max".
        """
        super().__init__()

        self.region2vec: Region2Vec = region2vec
        # assign the pooling layer based on mean or max
        if pooling == "mean":
            self.pooling = MeanPooling()
        elif pooling == "max":
            self.pooling = MaxPooling()
        else:
            raise ValueError(f"Invalid pooling type {pooling} passed.")

    def forward(self, x) -> torch.Tensor:
        x = self.region2vec(x)
        x = self.pooling(x)
        return x
