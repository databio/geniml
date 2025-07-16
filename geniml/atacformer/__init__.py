from transformers import AutoConfig

from ._version import VERSION
from .data_processing import TrainingTokenizer
from .modeling_atacformer import AtacformerModel, AtacformerForMaskedLM, AtacformerForReplacedTokenDetection, AtacformerForCellClustering, AtacformerForUnsupervisedBatchCorrection
from .configuration_atacformer import AtacformerConfig
from .modeling_utils import freeze_except_last_n
from .training_utils import (
    DataCollatorForReplacedTokenDetection,
    DataCollatorForTripletLoss,
    DataCollatorForUnsupervisedBatchCorrection,
    ModelParameterChangeCallback,
    AdjustedRandIndexCallback,
    get_git_hash,
    get_decaying_cosine_with_hard_restarts_schedule_with_warmup,
    tokenize_anndata,
)

AutoConfig.register("atacformer", AtacformerConfig)

__all__ = [
    "AtacformerConfig",
    "AtacformerModel",
    "AtacformerForMaskedLM",
    "AtacformerForReplacedTokenDetection",
    "AtacformerForCellClustering",
    "AtacformerForUnsupervisedBatchCorrection",
    "DataCollatorForReplacedTokenDetection",
    "DataCollatorForTripletLoss",
    "DataCollatorForUnsupervisedBatchCorrection",
    "ModelParameterChangeCallback",
    "AdjustedRandIndexCallback",
    "TrainingTokenizer",
    "tokenize_anndata",
    "get_decaying_cosine_with_hard_restarts_schedule_with_warmup",
    "get_git_hash",
    "freeze_except_last_n",
]
__version__ = VERSION
__author__ = "Nathan LeRoy"
