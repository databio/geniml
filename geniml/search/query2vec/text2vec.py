import logging
from typing import Union

import numpy as np

from ...const import PKG_NAME
from ...text2bednn import Vec2VecFNN
from ...text2bednn.embedder.abstract import TextEmbedder
from ...text2bednn.embedder.fastembedder import FastEmbedder
from .abstract import Query2Vec

_LOGGER = logging.getLogger(PKG_NAME)


class Text2Vec(Query2Vec):
    """Map a query string into a vector into the embedding space of region sets"""

    def __init__(self, text_embedder: Union[str, TextEmbedder], v2v: Union[str, Vec2VecFNN]):
        """
        :param text_embedder: a subclass of TextEmbedder, or a model repository on Hugging Face
        :param v2v: a Vec2VecFNN (see geniml/text2bednn/text2bednn.py) or a model repository on Hugging Face
        """
        # Set model that embed natural language
        if isinstance(text_embedder, TextEmbedder):
            self.text_embedder = text_embedder
        elif isinstance(text_embedder, str):
            self.text_embedder = FastEmbedder(text_embedder)
        else:
            _LOGGER.error(
                "TypeError: Please give a sub class of TextEmbedder or a model repository on Hugging Face"
            )
        # Set model that maps natural language embeddings into the embedding space of region sets
        if isinstance(v2v, Vec2VecFNN):
            self.v2v = v2v
        elif isinstance(v2v, str):
            self.v2v = Vec2VecFNN(v2v)
        else:
            _LOGGER.error(
                "TypeError: Please give a Vec2VecFNN or a model repository on Hugging Face"
            )

    def forward(self, query: str) -> np.ndarray:
        """
        Embed the query natural language string

        :param query: a natural language string

        :return: the embedding vector of query
        """
        # embed query string
        query_embedding = self.text_embedder.forward(query)
        # map the query string embedding into the embedding space of region sets
        return self.v2v.embedding_to_embedding(query_embedding)
