import torch
import torch.nn as nn
import lightning as L


class ATACFormer(L.LightningModule):
    def __init__(self, vocab_size: int):
        super().__init__()
        # BERT-like transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.cls = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.transformer_encoder(x)
        x = self.cls(x)
        return x
