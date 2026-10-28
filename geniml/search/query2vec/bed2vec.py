from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import numpy as np

from ...const import PKG_NAME
from ...io import RegionSet
from .abstract import Query2Vec

if TYPE_CHECKING:
    from ...region2vec.main import Region2VecExModel

_LOGGER = logging.getLogger(PKG_NAME)


class BED2Vec(Query2Vec):
    """Embed a query region set into a vector"""

    def __init__(self, model: Union[str, Region2VecExModel]):
        """Initialize BED2Vec with a Region2VecExModel.

        Args:
            model: a Region2VecExModel instance or a model repository on Hugging Face
        """
        # imported here so that importing geniml.search does not load torch
        from ...region2vec.main import Region2VecExModel

        if isinstance(model, str):
            self.model = Region2VecExModel(model)
        elif isinstance(model, Region2VecExModel):
            self.model = model
        else:
            _LOGGER.error(
                "TypeError: Please give a Region2VecExModel or a model repository on Hugging Face"
            )

    def forward(self, query: Union[str, RegionSet]) -> np.ndarray:
        """Embed the query region set into a vector.

        Args:
            query: a RegionSet instance or the path to a BED file on disk

        Returns:
            The region set embedding vector.
        """
        # if query is a BED file name, read it as a RegionSet class
        if isinstance(query, str):
            query = RegionSet(query)

        region_embeddings = self.model.encode(query)
        # BED embedding: averaging region embeddings
        return np.mean(region_embeddings, axis=0)
