from transformers import AutoConfig

from ._version import VERSION
from .configuration_atacformer import AtacformerConfig
from .data_processing import TrainingTokenizer
from .modeling_atacformer import (
    AtacformerForCellClustering,
    AtacformerForMaskedLM,
    AtacformerForReplacedTokenDetection,
    AtacformerForUnsupervisedBatchCorrection,
    AtacformerModel,
)
from .modeling_utils import freeze_except_last_n, patch_atacformer_model_for_mps
from .training_utils import (
    AdjustedRandIndexCallback,
    DataCollatorForReplacedTokenDetection,
    DataCollatorForTripletLoss,
    DataCollatorForUnsupervisedBatchCorrection,
    ModelParameterChangeCallback,
    get_decaying_cosine_with_hard_restarts_schedule_with_warmup,
    get_git_hash,
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
    "patch_atacformer_model_for_mps",
]
__version__ = VERSION
__author__ = "Nathan LeRoy"
