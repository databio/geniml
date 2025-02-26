from typing import Tuple

import lightning as L
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..atacformer.main import AtacformerExModel


class CellTypeFineTuneAdapter(L.LightningModule):
    """
    An adapter for fine-tuning a model on cell type classification.
    """

    def __init__(
        self,
        model: AtacformerExModel,
        num_classes: int,
        init_lr: float = 1e-5,
        adamw_weight_decay: float = 0.01,
        **kwargs,
    ):
        """
        Instantiate a fine-tuning trainer.

        :param Region2VecExModel | scEmbed | Atacformer model: The model to fine-tune.

        :param kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(**kwargs)

        self.r2v_model = model._model
        self.tokenizer = model.tokenizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity()
        self.ffn = nn.Linear(model._model.d_model * 3, num_classes)
        self._exmodel = model
        self.init_lr = init_lr
        self.adamw_weight_decay = adamw_weight_decay

    def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """
        Perform mean-pooling on the token embeddings.

        From: https://www.pinecone.io/learn/series/nlp/train-sentence-transformers-softmax/
        """
        # reshape attention_mask to cover 768-dimension embeddings
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        # perform mean-pooling but exclude padding tokens (specified by in_mask)
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool

    def forward(self, x, mask=None):
        return self.r2v_model(x, mask=mask)

    def compute_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        """
        Compute the loss for the batch.

        :param batch: The batch
        :param batch_idx: The batch index

        :return: The loss
        """
        cell1, cell2, target, attn1, attn2 = batch

        # forward pass for the batch
        u = self.r2v_model(cell1, mask=attn1)
        v = self.r2v_model(cell2, mask=attn2)

        # pool the embeddings using mean
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)

        uv = torch.sub(u, v)
        uv_abs = torch.abs(uv)

        # concatenate the embeddings
        uv_concat = torch.cat((u, v, uv_abs), dim=1)

        # pass through the linear layer
        output = self.ffn(uv_concat)

        # compute the loss
        loss = self.loss_fn(output, target)

        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        """
        Perform a training step.

        :param batch: The batch
        :param batch_idx: The batch index

        :return: The loss
        """
        loss = self.compute_loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def on_validation_start(self) -> None:
        """
        Perform any setup before validation starts.

        This needs to be here because otherwise the
        TransformerEncoder layers fail (for some reason).
        """
        self.eval()
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        :param batch: The batch
        :param batch_idx: The batch index

        :return: The loss
        """
        loss = self.compute_loss(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        init_lr = self.init_lr or 1e-6
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=init_lr, weight_decay=self.adamw_weight_decay
        )

        self.trainer.fit_loop.setup_data()

        total_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs

        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_steps // 10,  # Restart every 1/10th of total steps
                T_mult=1,  # Keep the same cycle length
                eta_min=1e-7,  # Minimum learning rate
            ),
            "interval": "step",
            "frequency": 1,
        }

        # return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.r2v_model = model._model
        # linear layer acts as a classification layer for training
        # the model on the masked language modeling task
        self.linear = nn.Linear(model._model.d_model, model._model.vocab_size)

        self.tokenizer = model.tokenizer
        self.init_lr = kwargs.get("init_lr", 1e-5)

        # tie the weights of the linear layer with the token embeddings
        # this is done primarily to reduce the number of parameters
        # during training significantly
        self.linear.weight = self.r2v_model.embedding.weight

        # Verify vocab_size consistency
        assert self.r2v_model.vocab_size == len(
            self.tokenizer
        ), f"Model vocab_size {self.r2v_model.vocab_size} does not match tokenizer vocab_size {len(self.tokenizer)}"

        self.masked_token_id = self.tokenizer.mask_token_id()
        self.passing_token_id = self.tokenizer.padding_token_id()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        token_embeddings = self.r2v_model(x, mask=mask)
        masked_positions = (x == self.masked_token_id).nonzero(as_tuple=True)
        masked_embeddings = token_embeddings[masked_positions[0], masked_positions[1]]=
        logits = self.linear(masked_embeddings)
        return logits

    def on_fit_start(self):
        # move positional embeddings to the device
        self.r2v_model.positional_encoding = self.r2v_model.positional_encoding.to(self.device)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        We use the AdamW optimizer with a learning rate of 1e-6.

        See here: https://arxiv.org/abs/2302.01107

        > By default, AdamW [62], a variant of Adam which decouples the L2 regularization and the
        weight decay, is the most widely used optimizer for Transformers.
        """
        init_lr = self.init_lr or 1e-6
        optimizer = torch.optim.AdamW(self.parameters(), lr=init_lr, weight_decay=0.01)

        self.trainer.fit_loop.setup_data()

        total_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs

        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_steps // 10,  # Restart every 1/10th of total steps
                T_mult=1,  # Keep the same cycle length
                eta_min=1e-7,  # Minimum learning rate
            ),
            "interval": "step",
            "frequency": 1,
        }

        # return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the batch.

        :param logits: Logits from the model of shape (batch_size, seq_length, vocab_size).
        :param labels: Labels tensor of shape (batch_size, seq_length).
        :return: Computed loss.
        """
        # Reshape logits and labels for loss computation
        logits = logits.view(-1, self.r2v_model.vocab_size)
        labels = labels.view(-1)

        loss = self.loss_fn(logits, labels)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a training step.

        :param batch: The batch containing (tokens, masked_tokens, labels, attention_mask).
        :param batch_idx: The batch index.
        :return: The computed loss.
        """
        _, masked_tokens, labels, attention_mask = batch

        # Forward pass
        logits = self.forward(masked_tokens, mask=attention_mask)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a validation step.

        :param batch: The batch containing (tokens, masked_tokens, labels, attention_mask).
        :param batch_idx: The batch index.
        :return: The computed loss.
        """
        _, masked_tokens, labels, attention_mask = batch

        # Forward pass
        logits = self.forward(masked_tokens, mask=attention_mask)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # this breaks everything -- and I have NO idea why...
    def on_validation_start(self) -> None:
        """
        Perform any setup before validation starts.

        This needs to be here because otherwise the
        TransformerEncoder layers fail (for some reason).
        """
        self.eval()
        torch.set_grad_enabled(True)

        self.r2v_model.positional_encoding = self.r2v_model.positional_encoding.to(self.device)

    def on_train_start(self):
        self.r2v_model.positional_encoding = self.r2v_model.positional_encoding.to(self.device)
