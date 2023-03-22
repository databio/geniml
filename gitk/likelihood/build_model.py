#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from ..utils import timer_func
import pyBigWig

WINDOW_SIZE = 25
WRONG_UNIWIG = False


def model_binomial(folder_in, in_file, chrom, file_out, file_no=None, start=0):
    """ "Create binomial likelihood model
    First column likelihood of background
    Second column likelihood of coverage"""
    in_file = os.path.join(folder_in, in_file + ".bw")
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
    distr_ncov = np.subtract(
        file_no, distr_cov
    )  # for each position in how many files is empty
    cov = distr_cov / no_cov
    ncov = distr_ncov / no_ncov
    p_cov = np.log10(cov + 1e-10)
    p_ncov = np.log10(ncov + 1e-10)
    prob_array = np.vstack((p_ncov, p_cov)).T
    header = f"{chrom}_{chrom_size}"
    r = {header: prob_array}
    np.savez_compressed(file_out, **r)


class ChromosomeModel:
    def __init__(self, folder, chrom):
        self.folder = folder
        self.chromosome = chrom
        self.start_file = os.path.join(folder, f"{self.chromosome}_start")
        self.core_file = os.path.join(folder, f"{self.chromosome}_core")
        self.end_file = os.path.join(folder, f"{self.chromosome}_end")
        self.files = {
            "start": self.start_file+ ".npz",
            "core": self.core_file+ ".npz",
            "end": self.end_file+ ".npz",
        }
        self.models = {}

    def make_model(
        self, coverage_folder, coverage_start, coverage_end, coverage_core, file_no
    ):
        model_binomial(
            coverage_folder, coverage_start, self.chromosome, self.start_file, file_no
        )
        model_binomial(
            coverage_folder, coverage_core, self.chromosome, self.core_file, file_no
        )
        model_binomial(
            coverage_folder, coverage_end, self.chromosome, self.end_file, file_no
        )

    def read(self):
        values = np.load(self.files["start"])
        self.models["start"] = values[values.files[0]]
        values = np.load(self.files["core"])
        self.models["core"] = values[values.files[0]]
        values = np.load(self.files["end"])
        self.models["end"] = values[values.files[0]]

    def read_track(self, track):
        values = np.load(self.files[track])
        self.models[track] = values[values.files[0]]


@timer_func
def main(
    model_folder,
    coverage_folder,
    coverage_start,
    coverage_core,
    coverage_end,
    file_no=None,
):
    """
    Crate likelihood models for all chromosomes
    :param str model_folder: output folder
    :param str coverage_folder: folder with coverage files
    :param str coverage_start: file with coverage of start without extension
    :param str coverage_end: file with coverage of end without extension
    :param str coverage_core: file with coverage of core without extension
    :param int file_no: number of files used for making coverage tracks
    """
    os.makedirs(model_folder)
    bw_start = pyBigWig.open(os.path.join(coverage_folder, coverage_start + ".bw"))
    chroms = bw_start.chroms()
    bw_start.close()
    for c in chroms:
        if chroms[c] != 0:
            chr_model = ChromosomeModel(model_folder, c)
            chr_model.make_model(
                coverage_folder, coverage_start, coverage_end, coverage_core, file_no
            )
