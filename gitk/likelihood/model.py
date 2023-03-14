#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from ..utils import  timer_func
import pyBigWig


WINDOW_SIZE = 25
WRONG_UNIWIG = False


def model_binomial(folderin, in_file, chrom, folderout, fout,
                   file_no=None, start=0):
    """"Create binomial likelihood model
    First column likelihood of background
    Second column likelihood of coverage """
    in_file = os.path.join(folderin, in_file + ".bw")
    bw = pyBigWig.open(in_file)
    chrom_size = bw.chroms(chrom)
    if pyBigWig.numpy:
        distr_cov = bw.values(chrom, start, chrom_size, numpy=True)
    else:
        distr_cov = bw.values(chrom, start, chrom_size)
        distr_cov = np.array(distr_cov)
    distr_cov[np.isnan(distr_cov)] = 0
    if WRONG_UNIWIG and ("cove" not in in_file):
        distr_cov = np.pad(distr_cov[WINDOW_SIZE:], (0, WINDOW_SIZE))
    no_possible = file_no * len(distr_cov)  # number of possible spots covered
    no_cov = np.sum(distr_cov)  # number of spots covered
    no_ncov = np.subtract(no_possible, no_cov)  # number of spots uncovered
    distr_ncov = np.subtract(file_no, distr_cov)  # for each position in how many files is empty
    cov = distr_cov / no_cov
    ncov = distr_ncov / no_ncov
    p_cov = np.log10(cov + 1e-10)
    p_ncov = np.log10(ncov + 1e-10)
    prob_array = np.vstack((p_ncov, p_cov)).T
    header = f"{chrom}_{chrom_size}"
    r = {header: prob_array}
    np.savez_compressed(os.path.join(folderout, chrom + "_" + fout), **r)


def make_models_binomial(chrom, folder_out, folder_in,
                         in_file_start, in_file_end, in_file_core,
                         file_no=None):
    """"Make binomial model for each track
    :param str chrom: chromosome to process
    :param str folder_out: output folder
    :param str folder_in: folder with coverage files
    :param str in_file_start: file with coverage of start without extension
    :param str in_file_end: file with coverage of end without extension
    :param str in_file_core: file with coverage of core without extension
    :param int file_no: number of files used for making coverage tracks
    """

    model_binomial(folder_in, in_file_start, chrom, folder_out, "start", file_no=file_no)
    model_binomial(folder_in, in_file_end, chrom, folder_out, "end", file_no=file_no)
    model_binomial(folder_in, in_file_core, chrom, folder_out, "core", file_no=file_no)


def make_models_multinomial(chrom, folder_out, folder_in,
                            in_file_start, in_file_end, in_file_core,
                            file_no=None):
    """"Make multinomial model from tracks
    :param str chrom: chromosome to process
    :param str folder_out: output folder
    :param str folder_in: folder with coverage files
    :param str in_file_start: file with coverage of start without extension
    :param str in_file_end: file with coverage of end without extension
    :param str in_file_core: file with coverage of core without extension
    :param int file_no: number of files used for making coverage tracks
    """

    core = pyBigWig.open((os.path.join(folder_in, in_file_core + ".bw")))
    start = pyBigWig.open((os.path.join(folder_in, in_file_start + ".bw")))
    end = pyBigWig.open((os.path.join(folder_in, in_file_end + ".bw")))
    chrom_size = core.chroms(chrom)
    model = np.zeros((chrom_size, 4), dtype=float)
    if pyBigWig.numpy:
        model[:, 1] = core.values(chrom, 0, chrom_size, numpy=True)
        model[:, 0] = start.values(chrom, 0, chrom_size, numpy=True)
        model[:, 2] = end.values(chrom, 0, chrom_size, numpy=True)
    else:
        model[:, 1] = core.values(chrom, 0, chrom_size)
        model[:, 0] = start.values(chrom, 0, chrom_size)
        model[:, 2] = end.values(chrom, 0, chrom_size)
    core.close()
    start.close()
    end.close()
    model[np.isnan(model)] = 0
    cov_nonzero = np.argwhere(model[:, 1] > 0).flatten()
    model[cov_nonzero, :] = model[cov_nonzero, :] / np.sum(model[cov_nonzero, :],
                                                           axis=1, keepdims=True) * model[cov_nonzero, 1].reshape(
        (-1, 1))
    no_covered = np.sum(model)
    model[:, 3] = file_no - np.sum(model[:, :3], axis=1)
    no_background = np.sum(model[:, 3])
    model[:, :3] = model[:, :3] / no_covered  # probability of being covered
    model[:, 3] = model[:, 3] / no_background  # probability of being background
    model = np.log10(model + 1e-10)
    header = f"{chrom}_{chrom_size}"
    r = {header: model}
    np.savez_compressed(os.path.join(folder_out, chrom), **r)


@timer_func
def main(model_folder, coverage_folder,
         coverage_starts, coverage_ends, coverage_core,
         file_list=None, file_no=None,
         binomial=False, multinomial=False):
    """
    Crate likelihood models for all chromosomes
    :param bool multinomial: whether to use multinomial model
    :param bool binomial: whether to use binomial model
    :param str model_folder: output folder
    :param str coverage_folder: folder with coverage files
    :param str coverage_starts: file with coverage of start without extension
    :param str coverage_ends: file with coverage of end without extension
    :param str coverage_core: file with coverage of core without extension
    :param str file_list: file with list of files used for making coverage tracks
    :param int file_no: number of files used for making coverage tracks
    """
    os.makedirs(model_folder)
    bw_start = pyBigWig.open(os.path.join(coverage_folder,
                                          coverage_starts + ".bw"))
    chroms = bw_start.chroms()
    bw_start.close()
    if file_no is None:
        file_list = open(file_list).read().split("\n")[:-1]
        file_no = len(file_list)
    for c in chroms:
        if chroms[c] != 0:
            if binomial:
                make_models_binomial(c, model_folder, coverage_folder,
                                     coverage_starts, coverage_ends, coverage_core,
                                     file_no=file_no)
            if multinomial:
                make_models_multinomial(c, model_folder, coverage_folder,
                                        coverage_starts, coverage_ends, coverage_core,
                                        file_no=file_no)
