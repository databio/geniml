from gensim.models import Word2Vec
from huggingface_hub import hf_hub_download

from .. import scembed
from .tokenization import Tokenizer, HardTokenizer
from .const import MODEL_FILE_NAME, UNIVERSE_FILE_NAME


class PretrainedScembedModel:
    """
    A class for loading pretrained models from the HuggingFace Hub.
    """

    def __init__(
        self, model: str = None, file_name: str = None, tokenizer: Tokenizer = None
    ):
        self.model_name = model
        self.__model_path = None
        self.__universe_path = None
        self.__tokenizer = tokenizer
        self.__model = None

        # load if model is passed
        if model is not None:
            self.__download_model_files(model, file_name)
            self.__tokenizer = HardTokenizer(self.__universe_path)
            self.__model = self.__load_model(self.__model_path)

    def __download_model_files(
        self,
        model: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        **kwargs,
    ):
        """
        Download a pretrained model from the HuggingFace Hub. We need to download
        the actual model + weights, and the universe file.
        """
        model_path = hf_hub_download(model, model_file_name, **kwargs)
        universe_path = hf_hub_download(model, universe_file_name, **kwargs)

        self.__model_path = model_path
        self.__universe_path = universe_path

    def __load_model(self, path: str) -> scembed.SCEmbed:
        """
        Load a pretrained model from the HuggingFace Hub.

        :param path: The path to the model.
        :return: The loaded model.
        """
        return scembed.utils.load_scembed_model(path)
