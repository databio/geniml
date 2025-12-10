#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import os
from functools import cmp_to_key

import numpy as np

from geniml.likelihood.build_model import ModelLH

from ..utils import natural_chr_sort, read_chromosome_from_bw, timer_func
from .utils import find_full, predictions_to_bed

package_name = "numba"

if importlib.util.find_spec(package_name) is None:

    def process_part(
        cove,
        model_start=np.array([[]]),
        model_core=np.array([[]]),
        model_end=np.array([[]]),
    ):
        """Find ML path through matrix using dynamic programming without numba.

        Args:
            cove (ndarray): Coverage tracks.
            model_start (ndarray): Likelihood model for starts.
            model_core (ndarray): Likelihood model for core.
            model_end (ndarray): Likelihood model for ends.

        Returns:
            ndarray: ML path through matrix.
        """
        mat = np.zeros((len(cove), 4))
        (N, M) = mat.shape
        start_b = model_start[cove[:, 0], 0]
        core_b = model_core[cove[:, 1], 0]
        end_b = model_end[cove[:, 2], 0]
        start = model_start[cove[:, 0], 1]
        core = model_core[cove[:, 1], 1]
        end = model_end[cove[:, 2], 1]
        mat[:, 0] = start + core_b + end_b
        mat[:, 1] = start_b + core + end_b
        mat[:, 2] = start_b + core_b + end
        mat[:, 3] = start_b + core_b + end_b

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

else:
    from numba import njit

    @njit
    def process_part(
        cove,
        model_start=np.array([[]]),
        model_core=np.array([[]]),
        model_end=np.array([[]]),
    ):
        """Find ML path through matrix using dynamic programming with numba.

        Args:
            cove (ndarray): Coverage tracks.
            model_start (ndarray): Likelihood model for starts.
            model_core (ndarray): Likelihood model for core.
            model_end (ndarray): Likelihood model for ends.

        Returns:
            ndarray: ML path through matrix.
        """
        mat = np.zeros((len(cove), 4))
        (N, M) = mat.shape
        for i in range(N):
            start_b = model_start[cove[i, 0], 0]
            core_b = model_core[cove[i, 1], 0]
            end_b = model_end[cove[i, 2], 0]
            start = model_start[cove[i, 0], 1]
            core = model_core[cove[i, 1], 1]
            end = model_end[cove[i, 2], 1]
            mat[i, 0] = start + core_b + end_b
            mat[i, 1] = start_b + core + end_b
            mat[i, 2] = start_b + core_b + end
            mat[i, 3] = start_b + core_b + end_b
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


def make_ml_flexible_universe(model_lh, cove_folder, cove_prefix, chrom, file_out):
    """Make ML flexible universe per chromosome.

    Args:
        model_lh (ModelLH): Likelihood model.
        cove_folder (str): Path to a folder with genome coverage by tracks.
        cove_prefix (str): Prefix used in uniwig for creating coverage.
        chrom (str): Chromosome to be processed.
        file_out (str): Output file with the universe.
    """
    model_lh.read_chrom(chrom)
    chrom_model = model_lh[chrom]
    start = read_chromosome_from_bw(os.path.join(cove_folder, f"{cove_prefix}_start.bw"), chrom)
    core = read_chromosome_from_bw(os.path.join(cove_folder, f"{cove_prefix}_core.bw"), chrom)
    end = read_chromosome_from_bw(os.path.join(cove_folder, f"{cove_prefix}_end.bw"), chrom)
    cove = np.zeros((len(start), 3), dtype=np.uint16)
    cove[:, 0] = start
    cove[:, 1] = core
    cove[:, 2] = end
    full_start, full_end = find_full(cove)
    path = np.full(len(cove), 3, dtype=np.uint8)
    for s, e in zip(full_start, full_end):
        res = process_part(
            cove[s:e],
            chrom_model["start"],
            chrom_model["core"],
            chrom_model["end"],
        )
        path[s:e] = res

    predictions_to_bed(path, chrom, file_out)


def ml_universe(model_file, cove_folder, cove_prefix, file_out):
    """Make ML flexible universe.

    Args:
        model_file (str): Input name with likelihood models.
        cove_folder (str): Path to a folder with genome coverage by tracks.
        cove_prefix (str): Prefix used in uniwig for creating coverage.
        file_out (str): Output file with the universe.
    """
    if os.path.isfile(file_out):
        raise Exception(f"File : {file_out} exists")
    lh_model = ModelLH(model_file)
    chroms = sorted(lh_model.chromosomes_list, key=cmp_to_key(natural_chr_sort))
    for C in chroms:
        make_ml_flexible_universe(lh_model, cove_folder, cove_prefix, C, file_out)
