from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput, BaseModelOutput
from transformers.utils import logging

from atacformer import AtacformerModel
from geneformer import GeneformerModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_craft import CraftConfig

logger = logging.get_logger(__name__)


@dataclass
class CraftOutput(ModelOutput):
    """
    Args:
        loss (torch.FloatTensor of shape (1,), optional):
            Contrastive loss measuring the similarity between gene and chromatin accessibility representations.
        logits_per_atac (torch.FloatTensor of shape (gene_batch_size, atac_batch_size)):
            Scaled dot-product scores between the gene embeddings and the ATAC embeddings, representing gene-to-ATAC similarity.
        logits_per_genes (torch.FloatTensor of shape (atac_batch_size, gene_batch_size)):
            Scaled dot-product scores between the ATAC embeddings and the gene embeddings, representing ATAC-to-gene similarity.
        geneformer_output (BaseModelOutput, optional):
            Output from the gene encoder containing hidden states and additional information.
        atacformer_output (BaseModelOutput, optional):
            Output from the ATAC encoder containing hidden states and additional information.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_atac: Optional[torch.FloatTensor] = None
    logits_per_genes: Optional[torch.FloatTensor] = None
    geneformer_output: Optional[BaseModelOutput] = None
    atacformer_output: Optional[BaseModelOutput] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["geneformer_output", "atacformer_output"]
                else getattr(self, k).to_tuple()
            )
            for k in self.keys()
        )


@dataclass
class CraftGeneActivityOutput(ModelOutput):
    """
    Args:
        loss (torch.FloatTensor of shape (1,), optional):
            Loss value for the gene activity prediction task.
        gene_activity_predictions (torch.FloatTensor of shape (batch_size, n_genes)):
            Predicted gene activity scores for each gene in the batch.
    """

    loss: Optional[torch.FloatTensor] = None
    gene_activity_predictions: Optional[torch.FloatTensor] = None


class CraftModel(PreTrainedModel):
    """
    CRAFT Model with a masked language modeling head.
    """

    config_class = CraftConfig
    base_model_prefix = "craft"

    def __init__(self, config: CraftConfig):
        super().__init__(config)
        self.config = config
        self.geneformer_config = config.geneformer_config
        self.atacformer_config = config.atacformer_config

        self.gene_encoder = GeneformerModel(self.geneformer_config)
        self.atac_encoder = AtacformerModel(self.atacformer_config)

        self.projection_dim = config.projection_dim
        self.atac_embed_dim = self.atacformer_config.hidden_size
        self.gene_embed_dim = self.geneformer_config.hidden_size

        self.gene_projection = torch.nn.Linear(
            self.gene_embed_dim, self.projection_dim, bias=False
        )
        self.atac_projection = torch.nn.Linear(
            self.atac_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.post_init()

    def _pool_embeddings(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool the embeddings using the attention mask and mean pooling.

        Args:
            hidden_states (torch.Tensor): The hidden states from the transformer model.
            attention_mask (torch.Tensor): The attention mask to apply.
        """
        attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = (hidden_states * attention_mask).sum(1)
        sum_mask = attention_mask.sum(1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        gene_input_ids: torch.Tensor = None,
        gene_attention_mask: torch.Tensor = None,
        gene_token_type_ids: torch.Tensor = None,
        atac_input_ids: torch.Tensor = None,
        atac_attention_mask: torch.Tensor = None,
        return_dict: bool = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], CraftOutput]:
        """
        Forward pass through the CRAFT model.

        Args:
            gene_input_ids (torch.Tensor): Input IDs for the gene encoder.
            gene_attention_mask (torch.Tensor): Attention mask for the gene encoder.
            gene_token_type_ids (torch.Tensor): Token type IDs for the gene encoder.
            atac_input_ids (torch.Tensor): Input IDs for the ATAC encoder.
            atac_attention_mask (torch.Tensor): Attention mask for the ATAC encoder.
            return_dict (bool): Whether to return a dictionary or tuple.
        Returns:
            torch.Tensor: The logits from the CRAFT model.
        """
        # gene encoding
        gene_outputs = self.gene_encoder.bert(
            input_ids=gene_input_ids,
            attention_mask=gene_attention_mask,
            token_type_ids=gene_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        gene_hidden_states = gene_outputs.last_hidden_state  # last layer hidden states
        gene_pooled_output = self._pool_embeddings(gene_hidden_states, gene_attention_mask)

        # atac encoding
        atac_outputs = self.atac_encoder(
            input_ids=atac_input_ids,
            attention_mask=atac_attention_mask,
        )
        atac_pooled_output = self._pool_embeddings(atac_outputs, atac_attention_mask)

        # project into same space
        gene_projs = self.gene_projection(gene_pooled_output)
        atac_projs = self.atac_projection(atac_pooled_output)

        # normalize the projections
        gene_projs = F.normalize(gene_projs, dim=-1)
        atac_projs = F.normalize(atac_projs, dim=-1)

        # scaled pairwise cosine similarities
        cos_sims = torch.matmul(gene_projs, atac_projs.T) * self.logit_scale.exp()

        n = gene_input_ids.shape[0]
        labels = torch.arange(n, device=cos_sims.device, dtype=torch.long)
        loss_i = F.cross_entropy(cos_sims, labels)  # image→text
        loss_t = F.cross_entropy(cos_sims.T, labels)  # text→image
        loss = (loss_i + loss_t) / 2

        if not return_dict:
            return (loss, cos_sims, cos_sims.T, gene_outputs, atac_outputs)

        return CraftOutput(
            loss=loss,
            logits_per_atac=cos_sims,
            logits_per_genes=cos_sims.T,
            geneformer_output=gene_outputs,
            atacformer_output=atac_outputs,
        )


class CraftForContrastiveLearning(PreTrainedModel):
    """
    CRAFT model for contrastive learning between gene and ATAC embeddings. While this
    looks redudant with the CraftModel, it makes it easier to use the model
    for further tasks like gene activity prediction without needing to
    instantiate the CraftModel directly.

    Mostly used for pre-training tasks
    """

    config_class = CraftConfig
    base_model_prefix = "craft_for_contrastive_learning"

    def __init__(self, config: CraftConfig):
        super().__init__(config)
        self.craft = CraftModel(config)

    def forward(
        self,
        gene_input_ids: torch.Tensor,
        gene_attention_mask: torch.Tensor,
        gene_token_type_ids: torch.Tensor,
        atac_input_ids: torch.Tensor,
        atac_attention_mask: torch.Tensor,
    ) -> CraftOutput:
        """
        Forward pass through the model.

        Args:
            gene_input_ids (torch.Tensor): Input IDs for the gene encoder.
            gene_attention_mask (torch.Tensor): Attention mask for the gene encoder.
            gene_token_type_ids (torch.Tensor): Token type IDs for the gene encoder.
            atac_input_ids (torch.Tensor): Input IDs for the ATAC encoder.
            atac_attention_mask (torch.Tensor): Attention mask for the ATAC encoder.

        Returns:
            CraftOutput: The output of the CRAFT model containing loss and logits.
        """
        return self.craft(
            gene_input_ids=gene_input_ids,
            gene_attention_mask=gene_attention_mask,
            gene_token_type_ids=gene_token_type_ids,
            atac_input_ids=atac_input_ids,
            atac_attention_mask=atac_attention_mask,
            return_dict=True,
        )


class GeneActivityPredictionHead(nn.Module):
    """
    A head for computing gene activity scores from the shared latent space
    of the CRAFT model.

    Mostly used for scATAC-seq data, where we want to predict gene activity
    scores from the ATAC-seq embeddings.
    """

    def __init__(self, config: CraftConfig):
        super().__init__()
        self.projection_dim = config.projection_dim
        self.n_genes = config.geneformer_config.vocab_size - 2  # Exclude <pad> and <mask> tokens
        self.gene_activity_head = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.n_genes, bias=False),
        )

    def forward(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute gene activity scores.

        Args:
            latent_embeddings (torch.Tensor): The latent embeddings from the CRAFT model.

        Returns:
            torch.Tensor: The computed gene activity scores.
        """
        return self.gene_activity_head(latent_embeddings)


class CraftForGeneActivityPrediction(PreTrainedModel):
    """
    CRAFT model for gene activity prediction.
    """

    config_class = CraftConfig
    base_model_prefix = "craft_for_gene_activity_prediction"

    def __init__(self, config: CraftConfig):
        super().__init__(config)
        self.craft = CraftModel(config)
        self.gene_activity_head = GeneActivityPredictionHead(config)

    def forward(
        self,
        atac_input_ids: torch.Tensor,
        atac_attention_mask: torch.Tensor,
        gene_activity: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor | None, Any], CraftGeneActivityOutput]:
        """
        Forward pass through the model.

        Args:
            atac_input_ids (torch.Tensor): Input IDs for the ATAC encoder.
            atac_attention_mask (torch.Tensor): Attention mask for the ATAC encoder.
            gene_activity (Optional[torch.Tensor]): Optional gene activity scores for computing loss.
        Returns:
            torch.Tensor: The predicted gene activity scores.
        """
        atac_latent_embeddings = self.craft.atac_encoder(
            input_ids=atac_input_ids, attention_mask=atac_attention_mask
        )
        # pool the ATAC embeddings to get cell-level representations
        atac_latent_embeddings = self.craft._pool_embeddings(
            atac_latent_embeddings, atac_attention_mask
        )
        # project the ATAC embeddings to the shared latent space
        atac_latent_embeddings = self.craft.atac_projection(atac_latent_embeddings)

        # normalize the embeddings
        atac_latent_embeddings = F.normalize(atac_latent_embeddings, dim=-1)

        # compute gene activity predictions
        gene_activity_predictions = self.gene_activity_head(atac_latent_embeddings)

        loss = None
        if gene_activity is not None:
            loss = F.mse_loss(gene_activity_predictions, gene_activity)

        if not return_dict:
            return (loss, gene_activity_predictions)

        return CraftGeneActivityOutput(
            loss=loss, gene_activity_predictions=gene_activity_predictions
        )
