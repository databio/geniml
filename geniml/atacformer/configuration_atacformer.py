from transformers import PretrainedConfig


class AtacformerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of an `AtacformerModel`.
    it inherits from [`ModernBertConfig`] and expands it for Atacformer specific settings.
    instantiating a configuration with the defaults will yield a similar configuration to that of the
    modernbert base configuration.

    Args:
        use_pos_embeddings (`bool`, *optional*, defaults to `True`):
            whether to use positional embeddings.
        vocab_size (`int`, *optional*, defaults to 890711):
            vocabulary size tailored for genomic regions.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            the maximum sequence length that this model might ever be used with.
        hidden_size (`int`, *optional*, defaults to 384):
            the size of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            the size of the "intermediate" (often named feed-forward) layer in the transformer.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            the number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            the number of attention heads in each attention layer.
        pad_token_id (`int`, *optional*, defaults to 890705):
            the id of the token used for padding.
        eos_token_id (`int`, *optional*, defaults to 890708):
            the id of the token used for the end of a sequence.
        bos_token_id (`int`, *optional*, defaults to 890709):
            the id of the token used for the beginning of a sequence.
        cls_token_id (`int`, *optional*, defaults to 890707):
            the id of the token used for classification tasks.
        sep_token_id (`int`, *optional*, defaults to 890710):
            the id of the token used to separate segments in a sequence.
        sparse_prediction (`bool`, *optional*, defaults to `True`):
            whether to use sparse prediction for the output layer.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            the epsilon value used for layer normalization.
        embedding_dropout (`float`, *optional*, defaults to 0.0):
            the dropout probability for the embedding layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            the standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
            the cutoff factor for the truncated normal initializer.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            whether to tie the word embeddings with the output layer.
        num_batches (`int`, *optional*, defaults to 1):
            the number of batches when doing batch correction training.
        lambda_adv: float = 1.0,
            the weight for the adversarial loss.
        grl_alpha: float = 1.0,
            the alpha value for the gradient reversal layer.
        bc_unfreeze_last_n_layers (`int`, *optional*, defaults to 0):
            the number of last layers to unfreeze during training for batch correction.
        **kwargs: (additional keyword arguments, *optional*):
            additional configuration parameters.
    """

    model_type = "atacformer"

    def __init__(
        self,
        use_pos_embeddings: bool = True,
        vocab_size: int = 890711,
        max_position_embeddings: int = 8192,
        hidden_size: int = 384,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        pad_token_id: int = 890705,
        eos_token_id: int = 890708,
        bos_token_id: int = 890709,
        cls_token_id: int = 890707,
        sep_token_id: int = 890710,
        sparse_prediction: bool = True,
        norm_eps: float = 1e-5,
        embedding_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_cutoff_factor: float = 2.0,
        tie_word_embeddings: bool = True,
        num_batches: int = None,
        lambda_adv: float = 1.0,
        grl_alpha: float = 1.0,
        bc_unfreeze_last_n_layers: int = 2,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            sparse_prediction=sparse_prediction,
            norm_eps=norm_eps,
            embedding_dropout=embedding_dropout,
            initializer_range=initializer_range,
            initializer_cutoff_factor=initializer_cutoff_factor,
            tie_word_embeddings=tie_word_embeddings,
            num_batches=num_batches,
            lambda_adv=lambda_adv,
            grl_alpha=grl_alpha,
            bc_unfreeze_last_n_layers=bc_unfreeze_last_n_layers,
            **kwargs,
        )
        self.use_pos_embeddings = use_pos_embeddings
        self.num_batches = num_batches
        self.lambda_adv = lambda_adv
        self.grl_alpha = grl_alpha
        self.bc_unfreeze_last_n_layers = bc_unfreeze_last_n_layers

__all__ = ["AtacformerConfig"]
