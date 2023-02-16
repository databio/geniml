#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os
from time import time
import pyBigWig


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1)/60:.4f}min')
        return result
    return wrap_func


WINDOW_SIZE = 25
WRONG_UNIWIG = True


def model(folderin, in_file, chrom, folderout, fout,
          file_no=None, start=0):
    """"Creates likelihood model"""
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
    no_posible = file_no * len(distr_cov)  # number of possible spots covered
    no_cov = np.sum(distr_cov)  # number of spots covered
    no_ncov = np.subtract(no_posible, no_cov)  # number of spots uncovered
    distr_ncov = np.subtract(file_no, distr_cov)  # for each position in how many files is empty
    cov = distr_cov / no_cov
    ncov = distr_ncov / no_ncov
    p_cov = np.log10(cov + 1e-10)
    p_ncov = np.log10(ncov + 1e-10)
    prob_array = np.vstack((p_ncov, p_cov)).T
    header = f"{chrom}_{chrom_size}"
    r = {header: prob_array}
    np.savez_compressed(os.path.join(folderout, chrom + "_" + fout), **r)


def make_models(chrom, folder_out, folder_in,
                in_file_start, in_file_end, in_file_core,
                fout_start, fout_end, fout_core,
                file_no=None):
    """"Makes model for each track
    :param str chrom: chromosome to process
    :param str folder_out: output folder
    :param str folder_in: folder with coverage files
    :param str in_file_start: file with coverage of start without extension
    :param str in_file_end: file with coverage of end without extension
    :param str in_file_core: file with coverage of core without extension
    :param str fout_start: output file suffix with likelihood model for starts
    :param str fout_end: output file suffix with likelihood model for ends
    :param str fout_core: output file suffix with likelihood model for core
    :param int file_no: number of files used for making coverage tracks
    """

    model(folder_in, in_file_start, chrom, folder_out,
          fout_start, file_no=file_no)
    model(folder_in, in_file_end, chrom, folder_out,
          fout_end, file_no=file_no)
    model(folder_in, in_file_core, chrom, folder_out,
          fout_core, file_no=file_no)


@timer_func
def main(model_folder, coverage_folder,
         coverage_starts, coverage_ends, coverage_core,
         model_starts, model_ends, model_body,
         file_list=None, file_no=None):
    """
    function for crating likelihood models for all chromosomes
    :param model_folder: output folder
    :param coverage_folder: folder with coverage files
    :param coverage_starts: file with coverage of start without extension
    :param coverage_ends: file with coverage of end without extension
    :param coverage_core: file with coverage of core without extension
    :param model_starts: output file suffix with likelihood model for starts
    :param model_ends: output file suffix with likelihood model for ends
    :param model_body: output file suffix with likelihood model for core
    :param file_list: file with list of files used for making coverage tracks
    :param file_no: number of files used for making coverage tracks
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
            make_models(c, model_folder, coverage_folder, coverage_starts, coverage_ends, coverage_core, model_starts,
                        model_ends, model_body, file_no=file_no)
