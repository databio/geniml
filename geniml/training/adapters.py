import logging
from typing import Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..atacformer.main import AtacformerExModel
from ..nn import GradientReversal
from ..region2vec import Region2VecExModel
from ..scembed import ScEmbed
from .const import BATCH_CORRECTION_ADVERSARIAL_TRAINING_MODES

_LOGGER = logging.getLogger(__name__)


class CellTypeFineTuneAdapter(L.LightningModule):
    """
    An adapter for fine-tuning a model on cell type classification.
    """

    def __init__(
        self,
        model: Union[Region2VecExModel, ScEmbed, AtacformerExModel],
        init_lr: float = 1e-5,
        **kwargs,
    ):
        """
        Instantiate a fine-tuning trainer.

        :param Region2VecExModel model: The model to fine-tune.

        :param kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(**kwargs)
        self.loss_fn = nn.CosineEmbeddingLoss()
        self.r2v_model = model._model
        self.tokenizer = model.tokenizer
        self._exmodel = model
        self.init_lr = init_lr

    def forward(self, x):
        return self.r2v_model(x)

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ):
        """
        Perform a training step.

        :param batch: The batch
        :param batch_idx: The batch index

        :return: The loss
        """

        # move the batch to the device
        cell1, cell2, target, attn1, attn2 = batch

        # forward pass for the batch
        u = self.r2v_model(cell1, mask=attn1)
        v = self.r2v_model(cell2, mask=attn2)

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
        cell1, cell2, target, attn1, attn2 = batch

        # forward pass for the batch
        u = self.r2v_model(cell1, mask=attn1)
        v = self.r2v_model(cell2, mask=attn2)

        # pool the embeddings using mean
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)

        # compute the loss
        loss = self.loss_fn(u, v, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        init_lr = self.init_lr or 1e-5
        optimizer = torch.optim.Adam(self.parameters(), lr=init_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10, verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }


class MLMAdapter(L.LightningModule):
    """
    An adapter for training Atacformer on a masked language modeling task.
    """

    def __init__(self, model: AtacformerExModel, **kwargs):
        """
        Instantiate the masked-language-modeling adapter.

        :param Atacformer model: The model to train.
        :param int pad_token_id: The token to use for padding
        :param int mask_token_id: The token to use for masking
        :param kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        # linear layer acts as a classification layer for training
        # but is not used during inference
        self.linear = nn.Linear(model._model.d_model, model._model.vocab_size)
        self.r2v_model = model._model
        self.tokenizer = model.tokenizer
        self.lr = kwargs.get("lr", 1e-5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        token_embeddings = self.r2v_model(x, mask=mask)
        logits = self.linear(token_embeddings)
        return logits

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        We use the AdamW optimizer with a learning rate of 1e-3.

        See here: https://arxiv.org/abs/2302.01107

        > By default, AdamW [62], a variant of Adam which decouples the L2 regularization and the weight decay, is the most widely used optimizer for Transformers.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        """
        Perform a training step.

        The batch is a tuple of (tokens, masked_tokens, mask_ids). This step performs
        masked language modeling as described in the original BERT paper (https://arxiv.org/abs/1810.04805).

        :param batch: The batch
        :param batch_idx: The batch index

        """

        # move the batch to the device
        tokens, masked_tokens, masked_token_indexes, attention_mask = batch

        # forward pass for the batch
        output = self.forward(masked_tokens, mask=attention_mask)

        # get predictions and targets
        # the predictions are the logits for the masked tokens
        # defined by the masked_token_indexes
        # not sure what to do here...
        # can we set the logits where tokens are **not** masked
        # to one-hot vectors of the original tokens?
        # that way, the loss contributed to tokens we didnt mask
        # would be zero
        #
        # we have indexes which is 2 x 300 - indexes we want to access
        # organize index matrix as a 600 x 2 tensor
        # convert original masked_token_index into row-col index
        predictions = output.view(-1, self.r2v_model.vocab_size)[masked_token_indexes]
        targets = tokens.view(-1)[masked_token_indexes]

        # reshape once more
        predictions = predictions.view(predictions.shape[0] * predictions.shape[1], -1)
        targets = targets.view(targets.shape[0] * targets.shape[1])

        # compute the loss
        # can I use ignore index here?
        loss = self.loss_fn(predictions, targets)
        self.log("train_loss", loss)

        return loss

    # this breaks everything -- and I have NO idea why...
    # def validation_step(
    #     self,
    #     batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #     batch_idx: int,
    # ):
    #     """
    #     Perform a training step.

    #     The batch is a tuple of (tokens, masked_tokens, mask_ids). This step performs
    #     masked language modeling as described in the original BERT paper (https://arxiv.org/abs/1810.04805).

    #     :param batch: The batch
    #     :param batch_idx: The batch index

    #     """

    #     # move the batch to the device
    #     tokens, masked_tokens, masked_token_indexes, attention_mask = batch

    #     # forward pass for the batch
    #     output = self.forward(masked_tokens, mask=attention_mask)

    #     # get predictions and targets
    #     # the predictions are the logits for the masked tokens
    #     # defined by the masked_token_indexes
    #     # not sure what to do here...
    #     # can we set the logits where tokens are **not** masked
    #     # to one-hot vectors of the original tokens?
    #     # that way, the loss contributed to tokens we didnt mask
    #     # would be zero
    #     #
    #     # we have indexes which is 2 x 300 - indexes we want to access
    #     # organize index matrix as a 600 x 2 tensor
    #     # convert original masked_token_index into row-col index
    #     predictions = output.view(-1, self.r2v_model.vocab_size)[masked_token_indexes]
    #     targets = tokens.view(-1)[masked_token_indexes]

    #     # reshape once more
    #     predictions = predictions.view(predictions.shape[0] * predictions.shape[1], -1)
    #     targets = targets.view(targets.shape[0] * targets.shape[1])

    #     # compute the loss
    #     # can I use ignore index here?
    #     loss = self.loss_fn(predictions, targets)
    #     self.log("val_loss", loss)

    #     return loss


class AdversarialBatchCorrectionAdapter(L.LightningModule):
    """
    An adapter for training a model through an adversarial batch correction approach.

    The idea is to train a model to predict the batch of origin for each cell, and then use
    this information to correct for batch effects via adversarial training.
    """

    def __init__(
        self,
        model: Union[Region2VecExModel, ScEmbed],
        mode: BATCH_CORRECTION_ADVERSARIAL_TRAINING_MODES,
        num_batches: int,
        grad_rev_alpha: float = 1.0,
        lr: float = 1e-3,
        **kwargs,
    ):
        """
        Instantiate a fine-tuning trainer.

        :param Union[Region2VecExModel, ScEmbed] model: The model to fine-tune.
        :param Literal["adversary", "batch_correction"] mode: The mode to use for training. this
            can be either "adversary" or "batch_correction". "adversary" trains the model to predict
            the batch of origin for each cell, while "batch_correction" trains the model to correct
            for batch effects. "adversary" should be used first to train the adversary, and then
            "batch_correction" should be used to train the batch correction model.

        :param int num_batches: The number of batches in the dataset.
        :param float grad_rev_alpha: The alpha value to use for the gradient reversal layer. This
            is used to control the strength of the adversarial training. A higher value will make the
            adversarial training stronger. For more information, see: https://arxiv.org/abs/1409.7495
        :param float lr: The learning rate to use for training the model.
        :param kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(**kwargs)
        if mode not in ["adversary", "batch_correction"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of {BATCH_CORRECTION_ADVERSARIAL_TRAINING_MODES}"
            )

        self.mode = mode
        self.num_batches = num_batches
        self.grad_rev_alpha = grad_rev_alpha
        self.lr = lr

        self.r2v_model = model.model
        self.tokenizer = model.tokenizer
        self._exmodel = model

        self.classifier = nn.Linear(model.model.embedding_dim, self.num_batches)
        self.loss_fn = nn.CrossEntropyLoss()
        # TODO: integrate this, but for now, we'll just use the default
        self.grad_rev = GradientReversal(self.grad_rev_alpha)

        self._update_models_for_mode()

    def _freeze_region2vec_model(self):
        """
        Freeze the Region2Vec model.
        """
        for param in self.r2v_model.parameters():
            param.requires_grad = False

    def _unfreeze_region2vec_model(self):
        """
        Unfreeze the Region2Vec model.
        """
        for param in self.r2v_model.parameters():
            param.requires_grad = True

    def _freeze_classifier(self):
        """
        Freeze the classifier.
        """
        for param in self.classifier.parameters():
            param.requires_grad = False

    def _unfreeze_classifier(self):
        """
        Unfreeze the classifier.
        """
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _update_models_for_mode(self):
        """
        Update the models based on the current mode.
        """
        if self.mode == "adversary":
            self._freeze_region2vec_model()
            self._unfreeze_classifier()
        else:
            self._unfreeze_region2vec_model()
            self._freeze_classifier()

    def set_learning_rate(self, lr: float):
        """
        Set the learning rate for the model.

        :param float lr: The learning rate to use.
        """
        self.lr = lr
        _LOGGER.info(f"Setting learning rate to {lr}")

    # setter for the mode to toggle back and forth
    def set_mode(self, mode: BATCH_CORRECTION_ADVERSARIAL_TRAINING_MODES):
        if mode not in ["adversary", "batch_correction"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of {BATCH_CORRECTION_ADVERSARIAL_TRAINING_MODES}"
            )
        self.mode = mode
        _LOGGER.info(f"Switching mode to {mode}")
        self._update_models_for_mode()

    def forward(self, x):
        embeddings = self.r2v_model(x)
        cell_embeddings = torch.mean(embeddings, dim=1)
        return self.classifier(cell_embeddings)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Perform a training step.

        :param batch: The batch. This should be a set of tokens and then the batch of origin for each cell.
        :param batch_idx: The batch index

        :return: The loss
        """
        x, y = batch

        # if we're in batch correction mode,
        # we need to flip the target
        if self.mode == "batch_correction":
            # this only works for binary classification
            y = torch.abs(y - 1)

        # forward pass for the batch
        output = self.forward(x)

        # compute the loss
        loss = self.loss_fn(output, y)

        # log the loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        :param batch: The batch. This should be a set of tokens and then the batch of origin for each cell.
        :param batch_idx: The batch index

        :return: The loss
        """
        x, y = batch

        # forward pass for the batch
        output = self.forward(x)

        # compute the loss
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
