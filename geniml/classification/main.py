import logging

import torch.nn as nn

from .const import (
    MODULE_NAME,
)

from ..nn.main import Attention
from ..region2vec.main import Region2Vec

_LOGGER = logging.getLogger(MODULE_NAME)


# TODO: eventually remove this
class Region2VecClassifier(nn.Module):
    def __init__(self, region2vec: Region2Vec, num_classes: int, freeze_r2v: bool = False):
        """
        Initialize the Region2VecClassifier.

        :param Union[Region2Vec, str] region2vec: Either a Region2Vec instance or a path to a huggingface model.
        :param int num_classes: Number of classes to classify.
        :param bool freeze_r2v: Whether or not to freeze the Region2Vec model.
        """
        super().__init__()

        self.region2vec: Region2Vec = region2vec
        self.attention = Attention(self.region2vec.embedding_dim)
        self.output_layer = nn.Linear(self.region2vec.embedding_dim, num_classes)
        self.num_classes = num_classes

        if freeze_r2v:
            for param in self.region2vec.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.region2vec(x)
        x = self.attention(x)
        x = self.output_layer(x)
        return x
