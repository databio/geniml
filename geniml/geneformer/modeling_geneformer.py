# re-written after introspecting the original code: https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/pretraining_new_model/pretrain_geneformer_w_deepspeed.py

from transformers import BertForMaskedLM

from .configuration_geneformer import GeneformerConfig


class GeneformerModel(BertForMaskedLM):
    """
    Geneformer Model with a masked language modeling head.
    """

    config_class = GeneformerConfig
    base_model_prefix = "geneformer"
    _tied_weights_keys = [
        "cls.predictions.bias",
        "cls.predictions.decoder.bias",
        "cls.predictions.decoder.weight",
        "bert.embeddings.word_embeddings.weight",
    ]

    def __init__(self, config: GeneformerConfig):
        super().__init__(config)

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the model.
        """
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings of the model.
        """
        self.bert.embeddings.word_embeddings = value

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs
