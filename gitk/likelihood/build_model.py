#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from ..utils import timer_func
import pyBigWig
import tarfile
import tempfile


def model_binomial(folder_in, in_file, chrom, file_out, file_no=None, start=0):
    """ "Create binomial likelihood model
    First column likelihood of background
    Second column likelihood of coverage"""
    in_file = os.path.join(folder_in, in_file)
    bw = pyBigWig.open(in_file)
    chrom_size = bw.chroms(chrom)
    if pyBigWig.numpy:
        distr_cov = bw.values(chrom, start, chrom_size, numpy=True)
    else:
        distr_cov = bw.values(chrom, start, chrom_size)
        distr_cov = np.array(distr_cov)
    distr_cov[np.isnan(distr_cov)] = 0
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
        self.start_file = f"{self.chromosome}_start"
        self.core_file = f"{self.chromosome}_core"
        self.end_file = f"{self.chromosome}_end"
        self.files = {
            "start": self.start_file + ".npz",
            "core": self.core_file + ".npz",
            "end": self.end_file + ".npz",
        }
        self.models = {}

    def make_model(
        self, coverage_folder, coverage_start, coverage_end, coverage_core, file_no
    ):
        model_binomial(
            coverage_folder,
            coverage_start,
            self.chromosome,
            os.path.join(self.folder, self.start_file),
            file_no,
        )
        model_binomial(
            coverage_folder,
            coverage_core,
            self.chromosome,
            os.path.join(self.folder, self.core_file),
            file_no,
        )
        model_binomial(
            coverage_folder,
            coverage_end,
            self.chromosome,
            os.path.join(self.folder, self.end_file),
            file_no,
        )

    def read(self):
        model_folder = tarfile.open(self.folder, "r")
        for f in self.files:
            file = model_folder.extractfile(self.files[f])
            values = np.load(file)
            self.models[f] = values[values.files[0]]
        model_folder.close()

    def read_track(self, track):
        model_folder = tarfile.open(self.folder, "r")
        file = model_folder.extractfile(self.files[track])
        values = np.load(file)
        self.models[track] = values[values.files[0]]
        model_folder.close()


class ModelLH:
    def __init__(self, folder):
        self.folder = folder
        self.chromosomes_list = []
        self.chromosomes_models = {}
        if os.path.exists(self.folder):
            if tarfile.is_tarfile(self.folder):
                files = tarfile.open(self.folder, "r")
                chroms = files.getnames()
                self.chromosomes_list = list(set([i.split("_")[0] for i in chroms]))

    def make(self, coverage_folder, coverage_prefix, file_no):
        tar_arch = tarfile.open(self.folder, "w")
        temp_dir = tempfile.TemporaryDirectory()
        bw_start = pyBigWig.open(
            os.path.join(coverage_folder, f"{coverage_prefix}_start.bw")
        )
        chroms = bw_start.chroms()
        bw_start.close()
        self.chromosomes_list = [i for i in chroms if chroms[i] != 0]
        for c in self.chromosomes_list:
            chrom_model = ChromosomeModel(temp_dir.name, c)
            chrom_model.make_model(
                coverage_folder,
                f"{coverage_prefix}_start.bw",
                f"{coverage_prefix}_core.bw",
                f"{coverage_prefix}_end.bw",
                file_no,
            )
            for f in chrom_model.files:
                tar_arch.add(
                    os.path.join(temp_dir.name, chrom_model.files[f]),
                    arcname=chrom_model.files[f],
                )
                os.remove(os.path.join(temp_dir.name, chrom_model.files[f]))
        temp_dir.cleanup()
        tar_arch.close()

    def read_chrom(self, chrom):
        self.chromosomes_models[chrom] = ChromosomeModel(self.folder, chrom)
        self.chromosomes_models[chrom].read()

    def read_chrom_track(self, chrom, track):
        self.chromosomes_models[chrom] = ChromosomeModel(self.folder, chrom)
        self.chromosomes_models[chrom].read_track(track)

    def clear_chrom(self, chrom):
        self.chromosomes_models[chrom] = None


@timer_func
def main(
    model_folder,
    coverage_folder,
    coverage_prefix,
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
    model = ModelLH(model_folder)
    model.make(coverage_folder, coverage_prefix, file_no)
