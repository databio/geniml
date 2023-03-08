#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from functools import cmp_to_key
from ..utils import natural_chr_sort, timer_func
from ..hmm.hmm import hmm_pred_to_bed, find_full_full_pos, find_full_empty_pos


def dynamic_path_finding(mat):
    (N, M) = mat.shape
    for i in range(1, N):
        for j in range(M):
            mat[i, j] += max(mat[i - 1, j], mat[i - 1, j - 1])
    return mat


def get_path(mat):
    path = np.zeros(len(mat), dtype=np.int8)
    path[-1] = np.argmax(mat[-1])
    for i in range(len(mat)-2, -1, -1):
        prev_index = path[i+1]
        new_index = prev_index - (mat[i, prev_index - 1] > mat[i, prev_index])
        if new_index == -1:
            new_index = 3
        path[i] = new_index
    return path


def process_part(mat):
    dynamic_path_finding(mat)
    path = get_path(mat)
    return path


def make_ml_flexible_universe(folderin, chrom, fout):
    model = np.load(os.path.join(folderin, chrom + ".npz"))
    model = model[model.files[0]]
    seq = np.where(np.sum(model[:, :3], axis=1) > -30, 1, 0).astype(np.uint8)
    full_pos_no = np.sum(seq)
    if full_pos_no < len(seq) - full_pos_no:
        full_start, full_end = find_full_full_pos(seq)
    else:
        full_start, full_end = find_full_empty_pos(seq)
    path = np.full(len(model), 3, dtype=np.uint8)
    for s, e in zip(full_start, full_end):
        res = process_part(model[s:e])
        path[s:e] = res
    hmm_pred_to_bed(path, chrom, fout)


def main(folderin, fout):
    if os.path.isfile(fout):
        raise Exception(f"File : {fout} exists")
    chroms = os.listdir(folderin)
    chroms = [i.split(".")[0] for i in chroms]
    chroms = sorted(chroms, key=cmp_to_key(natural_chr_sort))
    for C in chroms:
        make_ml_flexible_universe(folderin, C, fout)
