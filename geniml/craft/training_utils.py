from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

class DataCollatorForCraft:
    """
    Pads + builds masks for gene/ATAC pairs used by CraftModel.
    """

    def __init__(
        self,
        gene_pad: int,
        atac_pad: int,
        gene_max_len: int | None = None,
        atac_max_len: int | None = None,
    ):
        self.gene_pad, self.atac_pad = gene_pad, atac_pad
        self.gene_max_len, self.atac_max_len = gene_max_len, atac_max_len

    @staticmethod
    def _truncate(seq: List[int], max_len: int | None) -> List[int]:
        return seq[:max_len] if max_len is not None else seq

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        gene_ids  = [torch.tensor(self._truncate(f["gene_input_ids"],  self.gene_max_len),
                                  dtype=torch.long)
                     for f in features]
        atac_ids  = [torch.tensor(self._truncate(f["atac_input_ids"],  self.atac_max_len),
                                  dtype=torch.long)
                     for f in features]

        gene_batch = pad_sequence(gene_ids, batch_first=True,
                                  padding_value=self.gene_pad)
        atac_batch = pad_sequence(atac_ids, batch_first=True,
                                  padding_value=self.atac_pad)

        gene_mask  = (gene_batch != self.gene_pad).long()
        atac_mask  = (atac_batch != self.atac_pad).bool() # atacformer needs a bool mask (using nn.TransformerEncoder)

        if "gene_token_type_ids" in features[0]:
            tt_ids = [torch.tensor(self._truncate(f["gene_token_type_ids"],
                                                  self.gene_max_len),
                                   dtype=torch.long)
                      for f in features]
            gene_tt = pad_sequence(tt_ids, batch_first=True, padding_value=0)
        else:
            gene_tt = torch.zeros_like(gene_batch)

        return {
            "gene_input_ids"      : gene_batch,
            "gene_attention_mask" : gene_mask,
            "gene_token_type_ids" : gene_tt,
            "atac_input_ids"      : atac_batch,
            "atac_attention_mask" : atac_mask,
        }
    
class DataCollatorForCraftGeneActivityPrediction:
    """
    Pads + builds masks for ATAC-gene pairs used by CraftForGeneActivityPrediction.

    Gene activity is always the same shape for everything, its just a set of floats
    representing the activity of each gene in the genome, so we don't pad it.
    """

    def __init__(
        self,
        atac_pad: int,
        atac_max_len: int | None = None,
    ):
        self.atac_pad = atac_pad
        self.atac_max_len = atac_max_len

    @staticmethod
    def _truncate(seq: List[int], max_len: int | None) -> List[int]:
        return seq[:max_len] if max_len is not None else seq

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        atac_ids  = [torch.tensor(self._truncate(f["atac_input_ids"],  self.atac_max_len),
                                  dtype=torch.long)
                     for f in features]

        gene_activity = torch.stack([torch.tensor(f["gene_activity"], dtype=torch.float)
                         for f in features])

        atac_batch = pad_sequence(atac_ids, batch_first=True,
                                  padding_value=self.atac_pad)

        atac_mask  = (atac_batch != self.atac_pad).bool() # atacformer needs a bool mask (using nn.TransformerEncoder)

        return {
            "atac_input_ids": atac_batch,
            "atac_attention_mask": atac_mask,
            "gene_activity": gene_activity
        }