import numpy as np
from time import time
import os
# from models import PoissonModel
import pyBigWig
from scipy.stats import nbinom

from logging import getLogger
from ..const import PKG_NAME

_LOGGER = getLogger(PKG_NAME)

def run_hmm_save_bed(start, end, cove, out_file, normalize, save_max_cove):
    if os.path.isfile(out_file):
        raise Exception(f"File : {out_file} exists")
    bw_start = pyBigWig.open(start + ".bw")
    chroms = bw_start.chroms()
    bw_start.close()
    chroms = {"chr21": chroms["chr21"]}
    for C in chroms:
        if chroms[C] > 0:
            pred, m = run_hmm(start, end, cove, C, out_file, normalize=normalize)
            hmm_pred_to_bed(
                pred, C, out_file, save_max_cove=save_max_cove, cove_file=cove + ".bw"
            )



def test_hmm(message):
    """ Just prints a test message """
    _LOGGER.info(message)
