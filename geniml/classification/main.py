import torch
import torch.nn as nn

from ..region2vec.main import Region2Vec
from ..tokenization.main import ITTokenizer


class SingleCellTypeClassifier(nn.Module):
    def __init__(self, region2vec: Region2Vec, num_classes: int):
        super().__init__()
        self.region2vec = region2vec
        self.num_classes = num_classes
        self.output_layer = nn.Linear(self.region2vec.embedding_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.region2vec(x)
        x = x.sum(dim=1)
        x = self.output_layer(x)
        return x


class SingleCellTypeClassifierExModel:
    def __init__(
        self,
        model_path: str = None,
        tokenizer: ITTokenizer = None,
        device: str = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.device = device

        if model_path is not None:
            self.__init_from_huggingface(model_path)
        else:
            self.__init_model(**kwargs)

    def _init_from_huggingface(self, model_path: str):
        raise NotImplementedError("Not implemented yet.")

    def _init_model(self, **kwargs):
        raise NotImplementedError("Not implemented yet.")
