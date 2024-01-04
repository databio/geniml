import torch
import torch.nn as nn

from .const import DEFAULT_BATCH_SIZE

from ..models import ExModel
from ..tokenization import Tokenizer


class Trainer:
    """
    A trainer is a class that takes a model and a tokenizer and trains the
    model. It is responsible for all aspects of training, including
    preprocessing, batching, and optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        """
        Instantiate a trainer.

        :param nn.Module model: A model
        :param Tokenizer tokenizer: A tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_exmodel(cls, exmodel: ExModel, **kwargs):
        """
        Instantiate a trainer from an ExModel. An ExModel (extended-model) is
        a small wrapper around a model, tokenizer, and universe. This method
        extracts the model and tokenizer from the ExModel and instantiates a
        trainer.

        :param ExModel exmodel: An ExModel
        :param kwargs: Additional keyword arguments to pass to the trainer's constructor

        :return: A trainer
        """
        if not hasattr(exmodel, "model"):
            raise ValueError("ExModel must have a model attribute")
        if not hasattr(exmodel, "tokenizer"):
            raise ValueError("ExModel must have a tokenizer attribute")
        return cls(exmodel.model, exmodel.tokenizer, **kwargs)

    def add_optimizer(self, optimizer: torch.optim.Optimizer, **kwargs):
        """
        Add an optimizer to the trainer. This is useful if you want to
        change the optimizer after the trainer has been instantiated.

        :param torch.optim.Optimizer optimizer: An optimizer
        :param kwargs: Additional keyword arguments to pass to the optimizer's constructor
        """
        if not self.model:
            raise ValueError("Cannot add optimizer without a model")
        self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def add_loss_fn(self, loss_fn: nn.Module):
        """
        Add a loss function to the trainer. This is useful if you want to
        change the loss function after the trainer has been instantiated.

        :param nn.Module loss_fn: A loss function
        """
        self.loss_fn = loss_fn

    def compute_loss(self, model: nn.Module, inputs: dict):
        """
        Compute the loss for a batch of data. This method must be implemented
        by subclasses.

        :param nn.Module model: A model
        :param dict inputs: A batch of data (e.g. a batch of regions + labels)
        """
        raise NotImplementedError("compute_loss must be implemented by subclasses")

    def train(
        self,
        train_data: any,
        epochs: int = 3,
        batch_size: int = DEFAULT_BATCH_SIZE,
        optimizer=torch.optim.Adam,
    ):
        pass
