import logging
from typing import Tuple, Union
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import torch
import torch.nn as nn
import lightning as L

from ..nn import GradientReversal
from ..region2vec import Region2VecExModel
from ..scembed import ScEmbed
from ..atacformer import AtacformerExModel

from .const import BATCH_CORRECTION_ADVERSARIAL_TRAINING_MODES

_LOGGER = logging.getLogger(__name__)


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
        self.r2v_model = model._model
        self.tokenizer = model.tokenizer
        self._exmodel = model

    def forward(self, x):
        return self.r2v_model(x)

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
        u = self.r2v_model(t1)
        v = self.r2v_model(t2)

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
        u = self.r2v_model(t1)
        v = self.r2v_model(t2)

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
        self.r2v_model = model._model
        self.tokenizer = model.tokenizer

    def forward(self, x):
        token_embeddings = self.r2v_model(x)
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

        self.r2v_model = model.model
        self.tokenizer = model.tokenizer
        self._exmodel = model

        self.classifier = nn.Linear(model.model.embedding_dim, self.num_batches)
        self.loss_fn = nn.CrossEntropyLoss()
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
        if self.mode == "adversary":
            cell_embeddings = self.grad_rev(cell_embeddings)
        return self.classifier(cell_embeddings)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Perform a training step.

        :param batch: The batch. This should be a set of tokens and then the batch of origin for each cell.
        :param batch_idx: The batch index

        :return: The loss
        """
        x, y = batch

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
