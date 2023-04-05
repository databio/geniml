#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from functools import cmp_to_key
from ..utils import natural_chr_sort, timer_func, read_chromosome_from_bw
from ..hmm.hmm import predictions_to_bed, find_full_full_pos, find_full
from .build_model import ModelLH
from numba import njit


@njit
def process_part(
    cove,
    model_start=np.array([[]]),
    model_core=np.array([[]]),
    model_end=np.array([[]]),
):
    """
    Finding ML path through matrix using dynamic programing
    :param array mat: fragment of likelihood model to be processed
    :return array: ML path through matrix
    """
    mat = np.zeros((len(cove), 4))
    (N, M) = mat.shape
    for i in range(N):
        sb = model_start[cove[i, 0], 0]
        cb = model_core[cove[i, 1], 0]
        eb = model_end[cove[i, 2], 0]
        s = model_start[cove[i, 0], 1]
        c = model_core[cove[i, 1], 1]
        e = model_end[cove[i, 2], 1]
        mat[i, 0] = s + cb + eb
        mat[i, 1] = sb + c + eb
        mat[i, 2] = sb + cb + e
        mat[i, 3] = sb + cb + eb
    for i in range(1, N):
        for j in range(M):
            mat[i, j] += max(mat[i - 1, j], mat[i - 1, j - 1])
    path = np.zeros(len(mat), dtype=np.int8)
    path[-1] = np.argmax(mat[-1])
    for i in range(len(mat) - 2, -1, -1):
        prev_index = path[i + 1]
        new_index = prev_index - (mat[i, prev_index - 1] > mat[i, prev_index])
        if new_index == -1:
            new_index = 3
        path[i] = new_index
    return path


def make_ml_flexible_universe(model_lh, cove_folder, cove_prefix, chrom, fout):
    """
    Make ML flexible universe per chromosome
    :param ModelLH model_lh: lh model
    :param str chrom: chromosome to be processed
    :param str fout: output file with the universe
    """
    model_lh.read_chrom(chrom)
    chrom_model = model_lh.chromosomes_models[chrom]
    start = read_chromosome_from_bw(
        os.path.join(cove_folder, f"{cove_prefix}_start.bw"), chrom
    )
    core = read_chromosome_from_bw(
        os.path.join(cove_folder, f"{cove_prefix}_core.bw"), chrom
    )
    end = read_chromosome_from_bw(
        os.path.join(cove_folder, f"{cove_prefix}_end.bw"), chrom
    )
    cove = np.zeros((len(start), 3), dtype=np.uint16)
    cove[:, 0] = start
    cove[:, 1] = core
    cove[:, 2] = end
    full_start, full_end = find_full(cove)
    path = np.full(len(cove), 3, dtype=np.uint8)
    for s, e in zip(full_start, full_end):
        res = process_part(
            cove[s:e],
            chrom_model.models["start"],
            chrom_model.models["core"],
            chrom_model.models["end"],
        )
        path[s:e] = res
    predictions_to_bed(path, chrom, fout)


def main(folder_in, cove_folder, cove_prefix, file_out):
    """
    Make ML flexible universe
    :param str folder_in: input name with likelihood models
    :param str file_out: output file with the universe
    """
    if os.path.isfile(file_out):
        raise Exception(f"File : {file_out} exists")
    lh_model = ModelLH(folder_in)
    chroms = sorted(lh_model.chromosomes_list, key=cmp_to_key(natural_chr_sort))
    for C in chroms:
        make_ml_flexible_universe(lh_model, cove_folder, cove_prefix, C, file_out)
