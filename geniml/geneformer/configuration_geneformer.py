from transformers import BertConfig


class GeneformerConfig(BertConfig):
    """
    Configuration for Geneformer model, a BERT-like transformer for gene tokens.
    """
    model_type = "geneformer"

    def __init__(
        self,
        vocab_size: int = 20275,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 12,
        attention_probs_dropout_prob: float = 0.02,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.02,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 4096,
        pad_token_id: int = 0,
        classifier_dropout: float = None,        
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            classifier_dropout=classifier_dropout,
            **kwargs
        )