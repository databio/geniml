from typing import Union

import scanpy as sc
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .. import scembed
from .tokenization import HardTokenizer
from .const import MODEL_FILE_NAME, UNIVERSE_FILE_NAME


class PretrainedScembedModel:
    """
    A class for loading pretrained models from the HuggingFace Hub.
    """

    def __init__(self, model: str = None, tokenizer: HardTokenizer = None):
        self.model_name = model
        self._model_path = None
        self._universe_path = None
        self._tokenizer: HardTokenizer = tokenizer
        self._model: scembed.SCEmbed = None

        # load if model is passed
        if model is not None:
            self._download_model_files(model)
            self._tokenizer = HardTokenizer(self._universe_path)
            self._model = self._load_model(self._model_path)

    def _download_model_files(
        self,
        model: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        **kwargs,
    ):
        """
        Download a pretrained model from the HuggingFace Hub. We need to download
        the actual model + weights, and the universe file.

        :param model: The name of the model to download (this is the same as the repo name).
        :param model_file_name: The name of the model file - this should almost never be changed.
        :param universe_file_name: The name of the universe file - this should almost never be changed.
        """
        model_path = hf_hub_download(model, model_file_name, **kwargs)
        universe_path = hf_hub_download(model, universe_file_name, **kwargs)

        # set the paths to the downloaded files
        self._model_path = model_path
        self._universe_path = universe_path

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def _load_model(self, path: str) -> scembed.SCEmbed:
        """
        Load a pretrained model from the HuggingFace Hub.

        :param path: The path to the model.
        :return: The loaded model.
        """
        return scembed.utils.load_scembed_model(path)

    def encode(
        self, data: Union[str, list[tuple[str, int, int]], sc.AnnData]
    ) -> np.ndarray:
        """
        Encode region sets data using the pretrained model.
        """
        if self._model is None:
            raise ValueError("No model loaded. Please load a model first.")

        # generate tokens
        tokenized = self._tokenizer.tokenize(data)

        embeddings = []
        for region_set in tqdm(tokenized, total=len(tokenized)):
            if len(region_set) == 0:
                raise ValueError(
                    "Encountered empty region set. Please check your input."
                )
            # compute embeddings using average pooling of all region embeddings
            embedding = np.mean(
                [
                    self._model.region2vec[region]
                    for region in region_set
                    if region in self._model.region2vec
                ],
                axis=0,
            )
            embeddings.append(embedding)
        return np.array(embeddings)
