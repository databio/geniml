import os
import pytest
from geniml.bedshift import bedshift

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def bs():
    return bedshift.Bedshift(
        os.path.join(SCRIPT_PATH, "test.bed"),
        chrom_sizes=os.path.join(SCRIPT_PATH, "hg38.chrom.sizes"),
    )
