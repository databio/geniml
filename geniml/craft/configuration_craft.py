from transformers import PretrainedConfig

from atacformer.configuration_atacformer import AtacformerConfig
from geneformer.configuration_geneformer import GeneformerConfig

class CraftConfig(PretrainedConfig):
    """
    Configuration for the CRAFT model, a contrastive RNA-ATAC transformer that attempts
    to learn leverage Geneformer and Atacformer to learn a joint representation of RNA and ATAC data.
    """
    def __init__(
        self,
        geneformer_config: GeneformerConfig = None,
        atacformer_config: AtacformerConfig = None,
        projection_dim: int = 512,
        logit_scale_init_value: float = 2.6592,
        **kwargs,
    ):
        """
        Joint configuration for the CRAFT model.

        Args:
            geneformer_config (GeneformerConfig): Configuration for the Geneformer model.
            atacformer_config (AtacformerConfig): Configuration for the Atacformer model.
            projection_dim (int): Dimension of the projection layer.
            logit_scale_init_value (float): Initial value for the logit scale parameter.
        """
        super().__init__(**kwargs)

        if isinstance(geneformer_config, dict):
            geneformer_config = GeneformerConfig.from_dict(geneformer_config)
        if isinstance(atacformer_config, dict):
            atacformer_config = AtacformerConfig.from_dict(atacformer_config)

        self.geneformer_config = geneformer_config
        self.atacformer_config = atacformer_config
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value