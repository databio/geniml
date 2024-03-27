import logging
from typing import Union

import numpy as np

from ...const import PKG_NAME
from ...io import RegionSet
from ...region2vec import Region2VecExModel
from .abstract import Query2Vec

_LOGGER = logging.getLogger(PKG_NAME)


class Bed2Vec(Query2Vec):
    """Embed a query region set into a vector"""

    def __init__(self, model: Union[str, Region2VecExModel]):
        """
        set the Region2VecExModel

        Parameters
        ----------
        model : a Region2VecExModel or a model repository on Hugging Face
        """
        if isinstance(model, str):
            self.model = Region2VecExModel(model)
        elif isinstance(model, Region2VecExModel):
            self.model = model
        else:
            _LOGGER.error(
                "TypeError: Please give a Region2VecExModel or a model repository on Hugging Face"
            )

    def forward(self, query: Union[str, RegionSet]) -> np.ndarray:
        """

        Parameters
        ----------
        query : a RegionSet, or the path to a BED file in disk

        Returns
        -------
        the region set embedding
        """
        # if query is a BED file name, read it as a RegionSet class
        if isinstance(query, str):
            query = RegionSet(query)

        region_embeddings = self.model.encode(query)
        # BED embedding: averaging region embeddings
        return np.mean(region_embeddings, axis=0)
