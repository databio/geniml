import numpy as np
import pytest
import torch
from geniml.eval.ctt import get_ctt_score
from geniml.eval.gdst import get_gdst_score
from geniml.eval.npt import get_npt_score
from geniml.eval.utils import load_genomic_embeddings
from geniml.region2vec.main import Region2VecExModel


@pytest.mark.skipif(
    "not config.getoption('--huggingface')",
    reason="Only run when --huggingface is given",
)
def test_loading():
    repo = "databio/r2v-scatlas-hg38-v2"
    embeddings, regions = load_genomic_embeddings(repo, "exmodel")
    assert len(embeddings) == len(regions)

    exmodel = Region2VecExModel(repo)
    for i in range(len(embeddings)):
        vec = embeddings[i]
        token = regions[i]
        assert exmodel.tokenizer.decode(i)[0] == token
        assert np.equal(exmodel._model.projection(torch.tensor(i)).detach().numpy(), vec).all()


@pytest.mark.skipif(
    "not config.getoption('--huggingface')",
    reason="Only run when --huggingface is given",
)
def test_hf():
    repo = "databio/r2v-scatlas-hg38-v2"

    cts = get_ctt_score(repo, "exmodel")
    # print(f"CTT score: {cts}")
    nps = get_npt_score(repo, "exmodel", K=10)
    # pprint.pprint(nps)
    gds = get_gdst_score(repo, "exmodel")
    # print(f"GDS score: {gds}")


@pytest.mark.skip(reason="This is for my own testing")
def test_local_exmodel():
    path = "/home/claudehu/Desktop/sandbox_data/local_r2v"
    cts = get_ctt_score(path, "exmodel")
    # print(f"CTT score: {cts}")
    nps = get_npt_score(path, "exmodel", K=10)
    # print(f"NPT score: {nps}")
    gds = get_gdst_score(path, "exmodel")
    # print(f"GDS score: {gds}")
