import logging

import numpy as np
import torch

from ..const import BIO_BERT_REPO, MODULE_NAME
from .abstract import TextEmbedder

_LOGGER = logging.getLogger(MODULE_NAME)

try:
    from transformers import AutoModel, AutoTokenizer

    DEP_TRANSFORMERS = True
except ImportError:
    DEP_TRANSFORMERS = False
    _LOGGER.error("Install transformers, or ignore this if you don't need it")


if not DEP_TRANSFORMERS:

    class BioBertEmbedder(TextEmbedder):
        pass

else:

    class TransformerEmbedder(TextEmbedder):
        """sentence-transformers pulling biogpt model"""

        def __init__(self, model_name: str = BIO_BERT_REPO):
            """
            from https://github.com/UKPLab/sentence-transformers/issues/1824
            """
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

        def forward(self, input_text: str) -> np.ndarray:
            encoded_input = self.tokenizer(input_text, return_tensors="pt")

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = model_output.last_hidden_state
            pooled_embeddings = embeddings.mean(dim=1)

            return pooled_embeddings.detach().cpu().numpy().reshape(-1)
