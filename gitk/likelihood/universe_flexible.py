#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from functools import cmp_to_key
from ..utils import natural_chr_sort, timer_func
from ..hmm.hmm import predictions_to_bed, find_full_full_pos, find_full_empty_pos
from .build_model import ModelLH
from numba import njit


@njit
def process_part(model):
    """
    Finding ML path through matrix using dynamic programing
    :param array mat: fragment of likelihood model to be processed
    :return array: ML path through matrix
    """
    mat = np.zeros((len(model), 4))
    (N, M) = mat.shape
    background = [0, 2, 4]
    for i in range(N):
        for k in background:
            mat[i, 3] += model[i, k]
        for j in range(M - 1):
            back = background[:]
            back.remove(2 * j)
            mat[i, j] = model[i, 2 * j + 1]
            for k in back:
                mat[i, j] += model[i, k]

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


def make_ml_flexible_universe(model_lh, chrom, fout):
    """
    Make ML flexible universe per chromosome
    :param str folderin: input folder with likelihood models
    :param str chrom: chromosome to be processed
    :param str fout: output file with the universe
    """
    model_lh.read_chrom(chrom)
    chrom_model = model_lh.chromosomes_models[chrom]
    model = np.hstack((chrom_model.models["start"], chrom_model.models["core"], chrom_model.models["end"]))
    model_lh.clear_chrom(chrom)
    seq = np.where(np.sum(model[:, [1, 3, 5]], axis=1) > -30, 1, 0).astype(np.uint8)
    full_pos_no = np.sum(seq)
    if full_pos_no < len(seq) - full_pos_no:
        full_start, full_end = find_full_full_pos(seq)
    else:
        full_start, full_end = find_full_empty_pos(seq)
    path = np.full(len(model), 3, dtype=np.uint8)
    for s, e in zip(full_start, full_end):
        res = process_part(model[s:e])
        path[s:e] = res
    predictions_to_bed(path, chrom, fout)


@timer_func
def main(folderin, fout):
    """
    Make ML flexible universe
    :param str folderin: input folder with likelihood models
    :param str fout: output file with the universe
    """
    if os.path.isfile(fout):
        raise Exception(f"File : {fout} exists")
    lh_model = ModelLH(folderin)
    chroms = sorted(lh_model.chromosomes_list, key=cmp_to_key(natural_chr_sort))
    for C in chroms:
        make_ml_flexible_universe(lh_model, C, fout)
