import os
import sys

import pytest
import scanpy as sc
import numpy as np

# add parent directory to path
sys.path.append("../")

from gitk import scembed


@pytest.fixture
def adata():
    return sc.read_h5ad("tests/data/buenrostro.h5ad")


@pytest.fixture
def projector():
    return scembed.Projector("databio/scatlas")


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


@pytest.mark.skip  # projector is too large to test in CI
def test_download_model():
    registry = "databio/scatlas_v2"
    scembed.utils.download_remote_model(registry, scembed.const.MODEL_CACHE_DIR)
    config_file = os.path.join(
        scembed.const.MODEL_CACHE_DIR, registry, scembed.const.MODEL_CONFIG_FILE_NAME
    )
    assert os.path.exists(config_file)


@pytest.mark.skip  # projector is too large to test in CI
def test_load_local_model():
    registry = "databio/scatlas_v2"
    config_file = os.path.join(
        scembed.const.MODEL_CACHE_DIR, registry, scembed.const.MODEL_CONFIG_FILE_NAME
    )
    projector = scembed.Projector(config_file)
    assert isinstance(projector.model_config, scembed.models.ModelCard)
    assert isinstance(projector.model, dict)
    assert isinstance(projector.universe, scembed.models.Universe)


@pytest.mark.skip  # projector is too large to test in CI
def test_init_projector(projector):
    assert isinstance(projector.model_config, scembed.models.ModelCard)
    assert isinstance(projector.model, dict)
    assert isinstance(projector.universe, scembed.models.Universe)


@pytest.mark.skip(reason="bedtools isnt installed in CI")
def test_tokenization():
    """
    Test tokenization for three scenarios:
    1. One overlap in B per region in A
    2. At least one region in A doesn't overlap with any region in B
    3. A region in A overlaps with multiple regions in B

    Scenario 1:
    A: |-----------|        |-----------|           |-----------|
    B:     |-----------|        |-----------|           |-----------|

    Scenario 2:
    A: |-----------|        |-------|              |-----------|
    B:     |-----------|                 |-------|          |-----------|

    Scenario 3:
    A: |-----------|        |-----------|           |-----------|
    B:     |-----------|  |--------| |-----------|           |-----------|

    """
    # scenario 1
    with open("tests/data/s1_a.bed", "r") as f:
        lines = f.readlines()
        a = ["_".join(l.strip().split("\t")) for l in lines]
    with open("tests/data/s1_b.bed", "r") as f:
        lines = f.readlines()
        b = ["_".join(l.strip().split("\t")) for l in lines]

    conversion_map = scembed.utils.generate_var_conversion_map(a, b)
    assert conversion_map == {
        "chr1_10_30": "chr1_20_40",
        "chr1_110_130": "chr1_120_140",
        "chr1_210_230": "chr1_220_240",
    }

    # scenario 2
    with open("tests/data/s2_a.bed", "r") as f:
        lines = f.readlines()
        a = ["_".join(l.strip().split("\t")) for l in lines]
    with open("tests/data/s2_b.bed", "r") as f:
        lines = f.readlines()
        b = ["_".join(l.strip().split("\t")) for l in lines]

    conversion_map = scembed.utils.generate_var_conversion_map(a, b)
    assert conversion_map == {
        "chr1_10_30": "chr1_20_40",
        "chr1_110_130": None,
        "chr1_210_230": "chr1_220_240",
    }

    # scenario 3
    with open("tests/data/s3_a.bed", "r") as f:
        lines = f.readlines()
        a = ["_".join(l.strip().split("\t")) for l in lines]
    with open("tests/data/s3_b.bed", "r") as f:
        lines = f.readlines()
        b = ["_".join(l.strip().split("\t")) for l in lines]

    conversion_map = scembed.utils.generate_var_conversion_map(a, b)
    assert conversion_map == {
        "chr1_10_30": "chr1_20_40",
        "chr1_110_130": "chr1_100_120",
        "chr1_210_230": "chr1_220_240",
    }


@pytest.mark.skip
def test_convert_to_universe(projector, adata):
    # convert to universe
    adata_converted = projector.convert_to_universe(adata)

    # ensure all chr_start_end in the new adata are in the universe
    def check_region_in_universe(r):
        assert f"{r['chr']}_{r['start'] }_{r['end']}" in projector.universe.regions

    adata_converted.var.apply(
        lambda r: check_region_in_universe(r),
        axis=1,
    )


def test_anndata_to_regionsets(adata):
    region_sets = scembed.utils.anndata_to_regionsets(adata)

    # assert the sum of the clipped row in the matrix equals
    # the number of regions in the region set
    x_clipped = adata.X.clip(max=1)
    for i in range(adata.shape[0]):
        assert len(region_sets[i]) == int(x_clipped[i, :].sum())


@pytest.mark.skip  # projector is too large to test in CI
def test_projection(projector, adata):
    adata_projected = projector.project(adata)

    # assert the embeddings are there and the shape is correct
    assert "embedding" in adata_projected.obs

    embeddings = np.array(adata_projected.obs["embedding"].to_numpy().tolist())
    assert embeddings.shape == (
        adata.shape[0],
        projector.model_config.model_parameters[0].embedding_dim,
    )
