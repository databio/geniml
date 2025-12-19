import os
from functools import cmp_to_key
from logging import getLogger
from typing import List, Optional, Tuple

import numpy as np
import pyBigWig
from scipy.stats import nbinom

from ..const import PKG_NAME
from ..utils import natural_chr_sort
from .const import LAMBDAS, TRANSMAT
from .models import PoissonModel
from .utils import find_full, predictions_to_bed

_LOGGER = getLogger(PKG_NAME)

""" States legend
0 -> start
1 -> core
2 -> end
3 -> background"""


def norm(track: np.ndarray, mode: str) -> None:
    """Normalize the coverage track depending on track type.

    For each unique value in the track, calculates the corresponding quantile taking
    into account that values occur different number of times.

    Args:
        track (ndarray): Coverage track that will be modified in-place.
        mode (str): Type of track ('ends' or 'core').

    Returns:
        None: The function modifies the track array in-place.
    """
    important_val = track[track != 0]
    important_val_unique, counts = np.unique(important_val, return_counts=True)
    uniq_dict = {i: j for i, j in zip(important_val_unique, counts)}
    # how many times each value is present in the track
    important_val_unique_sort = np.sort(important_val_unique)
    if mode == "ends":
        n = 0.1
    if mode == "core":
        n = 0.2
    bs = 0  # what fraction of the distribution was used for normalization
    val = {}  # for each unique value in track holds the corresponding quantile
    for i in important_val_unique_sort:
        move_val = (uniq_dict[i] / len(important_val)) / 2
        # how far from last quantile is te next one
        val[i] = nbinom.ppf(bs + move_val, 1, n)
        bs = bs + move_val * 2
    track[track != 0] = [val[i] for i in important_val]


def process_bigwig(
    file: pyBigWig.pyBigWig,
    seq: np.ndarray,
    p: int,
    chrom: str,
    chrom_size: int,
    normalize: bool = True,
    mode: Optional[str] = None,
) -> None:
    """Preprocess bigWig file.

    Args:
        file: BigWig file object.
        seq (ndarray): Sequence array to store results, modified in-place.
        p (int): Position in sequence array.
        chrom (str): Chromosome name.
        chrom_size (int): Chromosome size.
        normalize (bool): Whether to normalize the track.
        mode (str): Normalization mode ('ends' or 'core').

    Returns:
        None: The function modifies the seq array in-place.
    """
    if pyBigWig.numpy:
        track = file.values(chrom, 0, chrom_size, numpy=True)
    else:
        track = file.values(chrom, 0, chrom_size)
        track = np.array(track)
    track[np.isnan(track)] = 0
    track = track.astype(np.uint16)
    if normalize:
        norm(track, mode)
    seq[:, p] = track


def read_data(
    start: str, core: str, end: str, chrom: str, normalize: bool = True
) -> Tuple[int, np.ndarray]:
    """Read in and preprocess data.

    Args:
        start (str): Path to file with start coverage.
        core (str): Path to file with core coverage.
        end (str): Path to file with end coverage.
        chrom (str): Chromosome to analyse.
        normalize (bool): Whether to normalize the coverage.

    Returns:
        tuple: Chromosome size and coverage matrix.
    """
    start = pyBigWig.open(start + ".bw")
    chroms = start.chroms()
    chrom_size = chroms[chrom]
    seq = np.zeros((chrom_size, 3), dtype=np.uint16)
    process_bigwig(start, seq, 0, chrom, chrom_size, normalize, mode="ends")
    start.close()
    core = pyBigWig.open(core + ".bw")
    process_bigwig(core, seq, 1, chrom, chrom_size, normalize, mode="core")
    core.close()
    end = pyBigWig.open(end + ".bw")
    process_bigwig(end, seq, 2, chrom, chrom_size, normalize, mode="ends")
    end.close()
    return chrom_size, seq


def split_predict(
    seq: np.ndarray, empty_starts: List[int], empty_ends: List[int], model
) -> np.ndarray:
    """Make model prediction only for regions containing nonzero positions.

    Args:
        seq (ndarray): Coverage sequence.
        empty_starts (list): List of start positions.
        empty_ends (list): List of end positions.
        model: HMM model.

    Returns:
        ndarray: HMM predictions.
    """
    hmm_predictions = np.full(len(seq), 3, dtype=np.uint8)
    for s, e in zip(empty_starts, empty_ends):
        res = model.predict(seq[s:e])
        hmm_predictions[s:e] = res
    return hmm_predictions


def run_hmm(
    start: str, core: str, end: str, chrom: str, normalize: bool = True
) -> Tuple[np.ndarray, any]:
    """Make HMM prediction for given chromosome.

    Args:
        start (str): Path to start coverage file.
        core (str): Path to core coverage file.
        end (str): Path to end coverage file.
        chrom (str): Chromosome to analyse.
        normalize (bool): Whether to normalize the coverage.

    Returns:
        tuple: HMM predictions and model.
    """
    chrom_size, seq = read_data(start, core, end, chrom, normalize=normalize)
    empty_starts, empty_ends = find_full(seq)
    model = PoissonModel(TRANSMAT, LAMBDAS, save_matrix=False).model
    hmm_predictions = split_predict(seq, empty_starts, empty_ends, model)
    return hmm_predictions, model


def hmm_universe(
    coverage_folder: str,
    out_file: str,
    prefix: str = "all",
    normalize: bool = True,
    save_max_cove: bool = False,
) -> None:
    """Create HMM based universe from coverage.

    Args:
        coverage_folder (str): Path to folder with coverage files.
        out_file (str): Path to the output file with universe.
        prefix (str): Prefix for coverage file names.
        normalize (bool): Whether to normalize file.
        save_max_cove (bool): Whether to save the maximum peak coverage.

    Returns:
        None: The function writes predictions to the output file.
    """
    if os.path.isfile(out_file):
        raise Exception(f"File : {out_file} exists")
    start = os.path.join(coverage_folder, f"{prefix}_start")
    core = os.path.join(coverage_folder, f"{prefix}_core")
    end = os.path.join(coverage_folder, f"{prefix}_end")
    bw_start = pyBigWig.open(start + ".bw")
    chroms = bw_start.chroms()
    bw_start.close()
    chroms_key = list(chroms.keys())
    chroms_key = sorted(chroms_key, key=cmp_to_key(natural_chr_sort))
    chroms = {i: chroms[i] for i in chroms_key}
    for C in chroms:
        if chroms[C] > 0:
            pred, m = run_hmm(start, core, end, C, normalize=normalize)
            predictions_to_bed(
                pred,
                C,
                out_file,
                save_max_cove=save_max_cove,
                cove_file=core + ".bw",
            )


def test_hmm(message: str) -> None:
    """Print a test message.

    Args:
        message (str): Message to print.

    Returns:
        None: The function logs the message.
    """
    _LOGGER.info(message)
