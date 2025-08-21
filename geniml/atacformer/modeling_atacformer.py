from typing import Optional, Union, Tuple, List

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from transformers import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import (
    MaskedLMOutput,
    TokenClassifierOutput,
    BaseModelOutput,
    SequenceClassifierOutput,
)

from .configuration_atacformer import AtacformerConfig
from .modeling_utils import freeze_except_last_n
from .functional import revgrad

logger = logging.get_logger(__name__)

# try to import cut cross entropy
try:
    from cut_cross_entropy.linear_cross_entropy import LinearCrossEntropy

    CCE_AVAILABLE = True
except ImportError:
    CCE_AVAILABLE = False


class AtacformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AtacformerConfig
    base_model_prefix = "atacformer"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AtacformerEmbeddings(nn.Module):
    """
    Simple embedding layer that includes learnable token and position embeddings.
    """

    def __init__(self, config: AtacformerConfig):
        """
        Args:
            config (AtacformerConfig): Configuration object for the model.
        """
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        bsz, seq_len = input_ids.size()

        # create [0,1,2,…,seq_len‑1] and expand to [bsz, seq_len]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(pos_ids)
        x = tok_emb
        if self.config.use_pos_embeddings:
            x = x + pos_emb
        return x


class AtacformerModel(AtacformerPreTrainedModel):
    """
    atacformer model with a simple embedding layer that skips positional encoding.
    """

    config_class = AtacformerConfig
    base_model_prefix = "atacformer"

    def __init__(self, config: AtacformerConfig):
        super().__init__(config)
        self.embeddings = AtacformerEmbeddings(config)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                batch_first=True,
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="relu",
            ),
            num_layers=config.num_hidden_layers,
        )

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.token_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.token_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Optional[dict],
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (`torch.LongTensor`):
                Input tensor of shape (batch_size, sequence_length).
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            return_dict (`bool`, *optional*):
                Whether to return the outputs as a dict or a tuple.
        Raises:
            ValueError: If `input_ids` is not provided.
        Returns:
            `torch.Tensor`: The output of the model. It will be a tensor of shape (batch_size, sequence_length, hidden_size).
        """
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        # get embeddings
        embeddings = self.embeddings(input_ids)

        # pass through encoder
        outputs = self.encoder(embeddings, src_key_padding_mask=~attention_mask)

        return outputs


class EncodeTokenizedCellsMixin:
    """
    Provides a default `encode_tokenized_cells` by delegating to `self.atacformer(...)`.
    Assumes any subclass has an attribute `atacformer` with a forward method
    that takes (input_ids, attention_mask, return_dict=False).
    Token pooling (mean) can be disabled to return per-token embeddings.
    """

    def encode_tokenized_cells(
        self,
        input_ids: List[List[int]],
        batch_size: int = 16,
        pool_tokens: bool = True,
        max_tokens_per_cell: int = None,
    ) -> torch.Tensor:
        """
        Loops internally over input_ids to produce a [N, D] matrix (if pooled) or [N, L, D] tensor (if not).
        Args:
            input_ids (Sequence[torch.LongTensor]):
                A sequence of tokenized input IDs, each of shape (sequence_length,).
            batch_size (int, *optional*, defaults to 16):
                The batch size to use for encoding.
            pool_tokens (bool, *optional*, defaults to True):
                Whether to mean-pool the token embeddings (True) or return per-token embeddings (False).
            max_tokens_per_cell (int, *optional*, defaults to None):
                The maximum number of tokens to consider per cell. You can use this to override the models
                built in max_position_embeddings value (or context size).
        """
        if not hasattr(self, "atacformer"):
            raise AttributeError(
                "This class must have an 'atacformer' attribute with a forward method."
            )
        if not hasattr(self.config, "pad_token_id") or not hasattr(
            self.config, "max_position_embeddings"
        ):
            raise AttributeError(
                "This class must have 'pad_token_id' and 'max_position_embeddings' in its config."
            )

        pad_id = self.config.pad_token_id
        max_ctx = max_tokens_per_cell or self.config.max_position_embeddings

        device = next(self.parameters()).device
        all_embs = []

        with torch.no_grad():
            for start in tqdm(range(0, len(input_ids), batch_size), desc="Encoding batches"):
                torch.cuda.empty_cache()
                batch_seqs = input_ids[start : start + batch_size]
                toks = [
                    torch.tensor(
                        (
                            np.random.choice(s, size=max_ctx, replace=len(s) < max_ctx)
                            if len(s) > max_ctx
                            else s
                        ),
                        dtype=torch.long,
                        device=device,
                    )
                    for s in batch_seqs
                ]
                padded = nn.utils.rnn.pad_sequence(
                    toks, batch_first=True, padding_value=pad_id
                ).to(device)
                mask = padded != pad_id
                last_h = self.atacformer(input_ids=padded, attention_mask=mask)
                if pool_tokens:
                    masked = last_h * mask.unsqueeze(-1)
                    summed = masked.sum(dim=1)
                    lengths = mask.sum(dim=1).unsqueeze(-1)
                    batch_emb = summed / lengths.clamp(min=1)
                else:
                    batch_emb = last_h
                all_embs.append(batch_emb)
        return torch.cat(all_embs, dim=0)


class AtacformerForMaskedLM(EncodeTokenizedCellsMixin, AtacformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    """
    atacformerModel for masked language modeling.
    """

    def __init__(self, config: AtacformerConfig):
        super().__init__(config)
        self.atacformer = AtacformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # use cut cross entropy if available
        if CCE_AVAILABLE:
            self.loss_fct = LinearCrossEntropy()
        else:
            logger.warning(
                "Cut cross entropy not found, please install it with `pip install cut-cross-entropy`."
            )
            self.loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.atacformer.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        Forward pass of the model.

        Args:
            input_ids (`torch.LongTensor`):
                Input tensor of shape (batch_size, sequence_length).
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.LongTensor`, *optional*):
                Labels for masked language modeling.
            return_dict (`bool`, *optional*):
                Whether to return the outputs as a dict or a tuple.
        Returns:
            `MaskedLMOutput`: The output of the model.
        """
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ensure attention mask is bool
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()

        # get embeddings
        # shape (batch_size, sequence_length, hidden_size)
        outputs = self.atacformer(
            input_ids, attention_mask=attention_mask, return_dict=return_dict
        )

        # compute loss if labels are provided
        loss = None
        if labels is not None:
            required_dtype = self.get_output_embeddings().weight.dtype  # should be torch.bfloat16
            if outputs.dtype != required_dtype:
                # logger.warning_once(f"casting hidden states from {outputs.dtype} to {required_dtype} before cce loss.") # Optional logging
                outputs = outputs.to(required_dtype)

            loss = self.loss_fct(
                e=outputs,
                c=self.get_output_embeddings().weight,
                targets=labels,
                bias=self.get_output_embeddings().bias,
            )

        if not return_dict:
            if loss is not None:
                return (loss, None) + (outputs,)

        return MaskedLMOutput(loss=loss, logits=None, hidden_states=outputs, attentions=None)


class AtacformerDiscriminatorHead(nn.Module):
    """
    2-layer token-wise classifier, copied from Electra-Small.
    Hidden size is usually the same as the backbone’s, but you
    can pass a smaller `disc_hidden_size` in the config if you like.
    """

    def __init__(self, config: AtacformerConfig):
        super().__init__()
        hidden_sz = getattr(config, "discriminator_hidden_size", config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, hidden_sz)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_sz, eps=config.norm_eps)
        self.classifier = nn.Linear(hidden_sz, 1)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        x = self.dense(sequence_output)
        x = self.act(x)
        x = self.norm(x)
        logits = self.classifier(x).squeeze(-1)  # (B, L)
        return logits


class AtacformerForReplacedTokenDetection(EncodeTokenizedCellsMixin, AtacformerPreTrainedModel):
    """
    Atacformer model for replaced token detection. This model uses the ELECTRA
    framework to train a discriminator (this model) to detect replaced tokens.

    https://arxiv.org/abs/2003.10555
    """

    def __init__(self, config: AtacformerConfig):
        super().__init__(config)
        self.atacformer = AtacformerModel(config)
        self.discriminator = AtacformerDiscriminatorHead(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.atacformer.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        Forward pass of the model.

        Args:
            input_ids (`torch.LongTensor`):
                Input tensor of shape (batch_size, sequence_length).
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.LongTensor`, *optional*):
                Labels for masked language modeling.
            return_dict (`bool`, *optional*):
                Whether to return the outputs as a dict or a tuple.
        Returns:
            `TokenClassifierOutput`: The output of the model.
        """
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ensure attention mask is bool
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()

        # get embeddings
        # shape (batch_size, sequence_length, hidden_size)
        backbone_out = self.atacformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = self.discriminator(backbone_out)  # (B, L)

        loss = None
        if labels is not None:
            # labels: 1=replaced, 0=original, -100=ignore (special tokens)
            active = labels != -100
            if active.any():
                loss = F.binary_cross_entropy_with_logits(logits[active], labels.float()[active])

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return TokenClassifierOutput(  # simple HF container
            loss=loss, logits=logits, hidden_states=backbone_out
        )


class AtacformerForCellClustering(EncodeTokenizedCellsMixin, AtacformerPreTrainedModel):
    """
    Atacformer model for cell clustering. It follows a similar learning framework
    to SentenceBERT (SBERT), where the model is trained to minimize the distance
    between positive pairs and maximize the distance between negative pairs.
    """

    def __init__(self, config: AtacformerConfig):
        super().__init__(config)
        self.atacformer = AtacformerModel(config)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

        self.post_init()

    def get_input_embeddings(self):
        return self.atacformer.get_input_embeddings()

    def _mean_pooling(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling for the embeddings.

        Args:
            embeddings (`torch.Tensor`):
                Embeddings tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (`torch.Tensor`):
                Attention mask tensor of shape (batch_size, sequence_length).
        Returns:
            `torch.Tensor`: The mean-pooled embeddings.
        """
        # apply attention mask
        attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size())
        sum_embeddings = torch.sum(embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids_anchor: torch.LongTensor,
        attention_mask_anchor: Optional[torch.Tensor] = None,
        input_ids_positive: torch.LongTensor = None,
        attention_mask_positive: Optional[torch.Tensor] = None,
        input_ids_negative: torch.LongTensor = None,
        attention_mask_negative: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:

        if attention_mask_anchor is None:
            attention_mask_anchor = torch.ones_like(input_ids_anchor, dtype=torch.bool)
        if attention_mask_positive is None:
            attention_mask_positive = torch.ones_like(input_ids_positive, dtype=torch.bool)
        if attention_mask_negative is None:
            attention_mask_negative = torch.ones_like(input_ids_negative, dtype=torch.bool)

        # encode and pool
        emb_anchor = self._mean_pooling(
            self.atacformer(
                input_ids_anchor, attention_mask=attention_mask_anchor, return_dict=False
            ),
            attention_mask_anchor,
        )
        emb_positive = self._mean_pooling(
            self.atacformer(
                input_ids_positive, attention_mask=attention_mask_positive, return_dict=False
            ),
            attention_mask_positive,
        )
        emb_negative = self._mean_pooling(
            self.atacformer(
                input_ids_negative, attention_mask=attention_mask_negative, return_dict=False
            ),
            attention_mask_negative,
        )

        # Triplet margin loss
        loss = self.triplet_loss(emb_anchor, emb_positive, emb_negative)

        if not return_dict:
            return (loss, emb_anchor, emb_positive, emb_negative)

        return BaseModelOutput(
            loss=loss,
            last_hidden_state=None,
            hidden_states=(emb_anchor, emb_positive, emb_negative),
            attentions=None,
        )


class AtacformerPairwiseInteractionHead(nn.Module):
    """
    Pairwise interaction head that will build pairwise interaction]
    scores for embeddings output by AtacformerModel. (self-interaction)
    """

    def __init__(self, config: AtacformerConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the pairwise interaction head with a batch dimension.

        Args:
            embeddings (torch.Tensor): The input embeddings of shape (batch_size, n, hidden_size).

        Returns:
            torch.Tensor: The pairwise interaction scores of shape (batch_size, n, n).
        """
        B, n, d = embeddings.size()

        # create tensor pairs for each batch sample
        emb1 = embeddings.unsqueeze(2).expand(B, n, n, d)
        emb2 = embeddings.unsqueeze(1).expand(B, n, n, d)

        pairwise_features = torch.cat(
            [emb1, emb2, emb1 * emb2, torch.abs(emb1 - emb2)], dim=-1
        )  # shape: (B, n, n, 4*d)

        scores = self.mlp(pairwise_features).squeeze(-1)  # shape: (B, n, n)
        return scores


class AtacformerForPairwiseInteraction(AtacformerPreTrainedModel):
    """
    Atacformer model for pairwise interaction prediction. It follows a similar learning framework
    to the one used in protein-protein interaction prediction, where the model is trained to predict
    whether two proteins interact or not based on their sequence embeddings.
    """

    def __init__(self, config: AtacformerConfig):
        super().__init__(config)
        self.atacformer = AtacformerModel(config)
        self.pairwise_head = AtacformerPairwiseInteractionHead(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.atacformer.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        """
        Forward pass of the model.

        Args:
            input_ids (`torch.LongTensor`):
                Input tensor of shape (batch_size, sequence_length).
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor`, *optional*):
                Labels for pairwise interaction prediction.
            return_dict (`bool`, *optional*):
                Whether to return the outputs as a dict or a tuple.
        Returns:
            `BaseModelOutput`: The output of the model.
        """
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ensure attention mask is bool
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()

        # get embeddings
        # shape (batch_size, sequence_length, hidden_size)
        backbone_out = self.atacformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # get pairwise interaction scores
        pairwise_scores = self.pairwise_head(backbone_out)
        if not return_dict:
            return (pairwise_scores,)

        return BaseModelOutput(
            loss=None,
            last_hidden_state=backbone_out.last_hidden_state,
            hidden_states=None,
            attentions=None,
        )


class GRL(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


class AtacformerForUnsupervisedBatchCorrection(
    EncodeTokenizedCellsMixin, AtacformerPreTrainedModel
):
    """
    Atacformer model for batch correction. It follows a similar learning framework
    to the one used in domain adaptation, where the model is trained to correct
    batch effects in the embeddings.
    """

    def __init__(self, config: AtacformerConfig):
        super().__init__(config)
        self.atacformer = AtacformerModel(config)
        self.discriminator = AtacformerDiscriminatorHead(config)
        self.grl = GRL(alpha=config.grl_alpha)  # gradient reversal layer
        self.batch_prediction_head = nn.Linear(config.hidden_size, config.num_batches)

        self.lambda_adversarial = config.lambda_adv  # weight for adversarial loss

        self.post_init()
        freeze_except_last_n(self.atacformer, config.bc_unfreeze_last_n_layers)

    def get_input_embeddings(self):
        return self.atacformer.get_input_embeddings()

    def _mean_pooling(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling for the embeddings.

        Args:
            embeddings (`torch.Tensor`):
                Embeddings tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (`torch.Tensor`):
                Attention mask tensor of shape (batch_size, sequence_length).
        Returns:
            `torch.Tensor`: The mean-pooled embeddings.
        """
        # apply attention mask
        attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size())
        sum_embeddings = torch.sum(embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        batch_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass of the model.

        Args:
            input_ids (`torch.LongTensor`):
                Input tensor of shape (batch_size, sequence_length).
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor`, *optional*):
                Labels for masked language modeling (ELECTRA).
            batch_labels (`torch.Tensor`, *optional*):
                Labels for batch prediction. Should be of shape (batch_size,).
            return_dict (`bool`, *optional*):
                Whether to return the outputs as a dict or a tuple.
        Returns:
            `BaseModelOutput`: The output of the model.
        """
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ensure attention mask is bool
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()

        # get embeddings
        # shape (batch_size, sequence_length, hidden_size)
        backbone_out = self.atacformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cell_embeddings = self._mean_pooling(backbone_out, attention_mask)

        # 1) MLM-ELECTRA loss
        logits_mlm = self.discriminator(backbone_out)  # (B, L)
        loss_mlm = None
        if labels is not None:
            # labels: 1=replaced, 0=original, -100=ignore (special tokens)
            active = labels != -100
            if active.any():
                loss_mlm = F.binary_cross_entropy_with_logits(
                    logits_mlm[active], labels.float()[active]
                )

        # 2) Adversarial loss for batch prediction
        logits_adv = self.batch_prediction_head(self.grl(cell_embeddings))  # (B, num_batches)
        loss_adv = None
        if batch_labels is not None:
            loss_adv = F.cross_entropy(logits_adv, batch_labels, ignore_index=-100)

        # total
        loss = None
        if loss_mlm is not None and loss_adv is not None:
            loss = loss_mlm + self.lambda_adversarial * loss_adv
        elif loss_mlm is not None:
            loss = loss_mlm
        elif loss_adv is not None:
            loss = self.lambda_adversarial * loss_adv

        # keep losses for logging
        # if self.keep_losses:
        #     self.loss_mlm = loss_mlm
        #     self.loss_adv = loss_adv
        #     self.loss = loss

        if not return_dict:
            return (loss, logits_mlm, logits_adv, cell_embeddings)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_mlm,
            hidden_states=None,
            attentions=None,
        )
