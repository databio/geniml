import logging
import os
import sys

import pytest

# add parent directory to path
sys.path.append("../")

from gitk import scembed


def test_model_exists():
    """
    Test that the model exists on the model-hub.
    """
    exists = scembed.utils.check_model_exists_on_hub("databio/scatlas")
    assert exists


def test_model_doesnt_exist():
    """
    Test that the model exists on the model-hub.
    """
    exists = scembed.utils.check_model_exists_on_hub("databio/scatlas_doesnt_exist")
    assert not exists


@pytest.mark.skip
def test_download_model():
    registry = "databio/scatlas"
    scembed.utils.download_remote_model(registry, scembed.const.MODEL_CACHE_DIR)
    config_file = os.path.join(
        scembed.const.MODEL_CACHE_DIR, registry, scembed.const.MODEL_CONFIG_FILE_NAME
    )
    assert os.path.exists(config_file)


@pytest.mark.skip
def test_load_local_model():
    registry = "databio/scatlas"
    config_file = os.path.join(
        scembed.const.MODEL_CACHE_DIR, registry, scembed.const.MODEL_CONFIG_FILE_NAME
    )
    projector = scembed.Projector(config_file)
    assert isinstance(projector.model_config, scembed.models.ModelCard)
    assert isinstance(projector.model, dict)
    assert isinstance(projector.universe, scembed.models.Universe)


@pytest.mark.skip
def test_init_projector():
    registry = "databio/scatlas"
    projector = scembed.Projector(registry)
    assert isinstance(projector.model_config, scembed.models.ModelCard)
    assert isinstance(projector.model, dict)
    assert isinstance(projector.universe, scembed.models.Universe)


@pytest.mark.skip
def test_tokenization():
    """
     Tokenize the top into the bottom

        chr1: 1000     2000     3000     4000     5000     6000
    A         |--------|        |--------|        |--------|
    B              |--------|        |--------|  |----| |--------|

        chr1: 1000     2000     3000     4000     5000     6000
    A         |--------|        |--------|        |--------|
    C              |--------|        |---------------|

    """
    regions_a = [
        "chr1_1000_2000",
        "chr1_3000_4000",
        "chr1_5000_6000",
    ]
    regions_b = [
        "chr1_1500_2500",
        "chr1_3500_4500",
        "chr1_4800_5050",
        "chr1_5500_6050",
    ]
    regions_c = ["chr1_1500_2500", "chr1_3500_5500"]

    var_map_ab = scembed.utils.generate_var_conversion_map(regions_a, regions_b)
    var_map_ac = scembed.utils.generate_var_conversion_map(regions_a, regions_c)
    assert len(var_map_ab) == 3
    assert len(var_map_ac) == 3


@pytest.mark.skip
def test_convert_to_universe():
    registry = "databio/scatlas"
    projector = scembed.Projector(registry)

    path_to_data = "/Users/nathanleroy/uva/lab/code/scEmbed-benchmarking/input/buenrostro2018/buenrostro2018_annotated.h5ad"
    adata = scembed.load_scanpy_data(path_to_data)

    # convert to universe
    adata_converted = projector.convert_to_universe(adata)
