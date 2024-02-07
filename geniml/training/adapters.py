from typing import Tuple, Union
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import torch
import torch.nn as nn
import lightning as L

from ..region2vec import Region2VecExModel
from ..scembed import ScEmbed
from ..atacformer import AtacformerExModel


class CellTypeFineTuneAdapter(L.LightningModule):
    """
    An adapter for fine-tuning a model on cell type classification.
    """

    def __init__(
        self,
        model: Union[Region2VecExModel, ScEmbed],
        **kwargs,
    ):
        """
        Instantiate a fine-tuning trainer.

        :param Region2VecExModel model: The model to fine-tune.

        :param kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(**kwargs)
        self.loss_fn = nn.CosineEmbeddingLoss()
        self.nn_model = model._model
        self.tokenizer = model.tokenizer
        self._exmodel = model

    def forward(self, x):
        return self.nn_model(x)

    def training_step(
        self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """
        Perform a training step.

        :param batch: The batch
        :param batch_idx: The batch index

        :return: The loss
        """

        # move the batch to the device
        pair, target = batch
        t1, t2 = pair

        # forward pass for the batch
        u = self.nn_model(t1)
        v = self.nn_model(t2)

        # pool the embeddings using mean
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)

        # compute the loss
        loss = self.loss_fn(u, v, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        :param batch: The batch
        :param batch_idx: The batch index

        :return: The loss
        """
        # move the batch to the device
        pair, target = batch
        t1, t2 = pair

        # forward pass for the batch
        u = self.nn_model(t1)
        v = self.nn_model(t2)

        # pool the embeddings using mean
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)

        # compute the loss
        loss = self.loss_fn(u, v, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MLMAdapter(L.LightningModule):
    """
    An adapter for training Atacformer on masked language modeling.
    """

    def __init__(self, model: AtacformerExModel, **kwargs):
        """
        Instantiate a fine-tuning trainer.

        :param Atacformer model: The model to fine-tune.
        :param int pad_token_id: The token to use for padding
        :param int mask_token_id: The token to use for masking
        :param kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        # linear layer acts as a classification layer for training
        # but is not used during inference
        self.linear = nn.Linear(model._model.d_model, model._model.vocab_size)
        self.nn_model = model._model
        self.tokenizer = model.tokenizer

    def forward(self, x):
        token_embeddings = self.nn_model(x)
        logits = self.linear(token_embeddings)
        return logits

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        We use the AdamW optimizer with a learning rate of 1e-3.

        See here: https://arxiv.org/abs/2302.01107

        > By default, AdamW [62], a variant of Adam which decouples the L2 regularization and the weight decay, is the most widely used optimizer for Transformers.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """
        Perform a training step.

        The batch is a tuple of (tokens, masked_tokens, mask_ids). This step performs
        masked language modeling as described in the original BERT paper (https://arxiv.org/abs/1810.04805).

        :param batch: The batch
        :param batch_idx: The batch index

        """

        # move the batch to the device
        tokens, masked_tokens, mask_ids, attention_mask = batch

        # strip the padding
        tokens = tokens[attention_mask]
        masked_tokens = masked_tokens[attention_mask]
        mask_ids = mask_ids[attention_mask]

        # forward pass for the batch
        output = self.forward(masked_tokens)

        # get predictions + targets
        predictions = output[mask_ids]
        targets = tokens[mask_ids]

        # compute the loss
        loss = self.loss_fn(predictions, targets)
        self.log("train_loss", loss)

        return loss
