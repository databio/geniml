import os
import pytest

import torch
import lightning as L

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from geniml.atacformer.main import Atacformer, AtacformerExModel
from geniml.atacformer.utils import AtacformerMLMDataset, mlm_batch_collator
from geniml.tokenization.main import ITTokenizer
from geniml.training.adapters import MLMAdapter
