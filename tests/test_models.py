import os
import sys

import pytest
import scanpy as sc

# add parent directory to path
sys.path.append("../")

from gitk import models


@pytest.fixture
def adata():
    return sc.read_h5ad("tests/data/buenrostro.h5ad")


@pytest.fixture
def pretrained_model():
    return models.PretrainedScembedModel("nleroy917/luecken2021")
