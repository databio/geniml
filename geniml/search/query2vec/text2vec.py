import logging
from typing import Union

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

from ...const import PKG_NAME
from ...text2bednn import Vec2VecFNN
from .abstract import Query2Vec

_LOGGER = logging.getLogger(PKG_NAME)


class Text2Vec(Query2Vec):
    """Map a query string into a vector into the embedding space of region sets"""

    def __init__(self, hf_repo: str, v2v: Union[str, Vec2VecFNN]):
        """
        :param text_embedder: a model repository on Hugging Face
        :param v2v: a Vec2VecFNN (see geniml/text2bednn/text2bednn.py) or a model repository on Hugging Face
        """
        # Set model that embed natural language
        self.text_embedder = HuggingFaceEmbeddings(model_name=hf_repo)
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
        query_embedding = np.array(self.text_embedder.embed_query(query))
        # map the query string embedding into the embedding space of region sets
        return self.v2v.embedding_to_embedding(query_embedding)
