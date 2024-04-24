import numpy as np
from fastembed import TextEmbedding

from .abstract import TextEmbedder


class FastEmbedder(TextEmbedder):
    """embedding model based on fastembed"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def forward(self, input_text: str) -> np.ndarray:
        return next(self.model.embed(input_text))
