import pytest
from geniml.eval.ctt import get_ctt_score
from geniml.eval.gdst import get_gdst_score
from geniml.eval.npt import get_npt_score


@pytest.mark.skipif(
    "not config.getoption('--huggingface')",
    reason="Only run when --huggingface is given",
)
def test_hf():
    repo = "databio/r2v-encode-hg38"

    cts = get_ctt_score(repo, "exmodel")
    # print(f"CTT score: {cts}")
    nps = get_npt_score(repo, "exmodel", K=10)
    # print(f"NPT score: {nps}")
    gds = get_gdst_score(repo, "exmodel")
    # print(f"GDS score: {gds}")


@pytest.mark.skip()
def test_local_exmodel():
    path = "/home/claudehu/Desktop/trials/local_r2v"
    cts = get_ctt_score(path, "exmodel")
    # print(f"CTT score: {cts}")
    nps = get_npt_score(path, "exmodel", K=10)
    # print(f"NPT score: {nps}")
    gds = get_gdst_score(path, "exmodel")
    # print(f"GDS score: {gds}")
