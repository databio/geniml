import logging

import numpy as np

from ..const import *
from .abstract import TextEmbedder

_LOGGER = logging.getLogger(MODULE_NAME)

try:
    from sentence_transformers import SentenceTransformer, models

    DEP_SENTENCE_TRANSFORMERS = True
except ImportError:
    DEP_SENTENCE_TRANSFORMERS = False
    _LOGGER.error(
        "SentTranEmbedder requires sentence_transformers. Install sentence_transformers, or ignore this if you don't need SentTranEmbedder"
    )


if not DEP_SENTENCE_TRANSFORMERS:

    class SentTranEmbedder(TextEmbedder):
        pass

    class BioGPTEmbedder(TextEmbedder):
        pass

else:

    class SentTranEmbedder(TextEmbedder):
        """embedding model based on sentence-transformers"""

        def __init__(self, model_name: str):
            self.model = SentenceTransformer(model_name)

        def forward(self, input_text: str) -> np.ndarray:
            return self.model.encode(input_text)

    class BioGPTEmbedder(TextEmbedder):
        """sentence-transformers pulling biogpt model"""

        def __init__(self, model_name: str = BIO_GPT_REPO):
            """from https://github.com/UKPLab/sentence-transformers/issues/1824"""
            word_embedding_model = models.Transformer(model_name)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
            )
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        def forward(self, input_text: str) -> np.ndarray:
            return self.model.encode(input_text)
