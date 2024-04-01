from abc import ABC, abstractmethod

import numpy as np


class TextEmbedder(ABC):
    """
    An abstract class representing Text Embedder Models. This allows
    the model to be either from fastembed or from sentence-transformers
    """

    @abstractmethod
    def __init__(self, model_name: str) -> None:
        """
        initiated by giving a model repository (on Hugging Face)

        :param model_name: a model repository (on Hugging Face)
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_text: str) -> np.ndarray:
        """
        Embed the input natural language string

        :param input_text: a natural language string to embed

        :return: the text embedding vector
        """
        raise NotImplementedError
