#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from ..utils import timer_func, read_chromosome_from_bw
import pyBigWig
import tarfile
import tempfile

WINDOW_SIZE = 25
WRONG_UNIWIG = False


def model_binomial(folder_in, in_file, chrom, file_out, file_no=None):
    """Create binomial likelihood model
    first column - likelihood of background
    second column - likelihood of coverage"""
    in_file = os.path.join(folder_in, in_file)
    distr_cov = read_chromosome_from_bw(in_file, chrom)
    chrom_size = len(distr_cov)
    no_possible = file_no * len(distr_cov)  # number of possible spots covered
    no_cov = np.sum(distr_cov)  # number of spots covered
    cov_uniq = np.unique(distr_cov).astype(np.uint16)
    prob_array = np.zeros((int(np.max(cov_uniq)) + 1, 2))
    for i in cov_uniq:
        prob_array[i] = [
            np.log10((file_no - i) / (no_possible - no_cov) + 1e-10),
            np.log10((i / no_cov) + 1e-10),
        ]
    header = f"{chrom}_{chrom_size}"
    r = {header: prob_array}
    np.savez_compressed(file_out, **r)


class ChromosomeModel:
    def __init__(self, folder, chrom):
        """
        :param str folder: file with the model
        :param str chrom: of which chromosome is the model
        """
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

    def __getitem__(self, item):
        return self.models[item]

    def make_model(self, coverage_folder, coverage_prefix, file_no):
        """
        Make a lh model of given chromosome from coverage files
        :param str coverage_folder: path to name with coverage files
        :param str coverage_prefix: prefixed used for making coverage files
        :param int file_no: number of files from which model is being created
        """
        model_binomial(
            coverage_folder,
            f"{coverage_prefix}_start.bw",
            self.chromosome,
            os.path.join(self.folder, self.start_file),
            file_no,
        )
        model_binomial(
            coverage_folder,
            f"{coverage_prefix}_core.bw",
            self.chromosome,
            os.path.join(self.folder, self.core_file),
            file_no,
        )
        model_binomial(
            coverage_folder,
            f"{coverage_prefix}_end.bw",
            self.chromosome,
            os.path.join(self.folder, self.end_file),
            file_no,
        )

    def read(self):
        """
        Read model
        """
        model_folder = tarfile.open(self.folder, "r")
        for f in self.files:
            file = model_folder.extractfile(self.files[f])
            values = np.load(file)
            self.models[f] = values[values.files[0]]
        model_folder.close()

    def read_track(self, track):
        """
        Read specific track from model
        """
        model_folder = tarfile.open(self.folder, "r")
        file = model_folder.extractfile(self.files[track])
        values = np.load(file)
        self.models[track] = values[values.files[0]]
        model_folder.close()


class ModelLH:
    def __init__(self, file):
        """
        Likelihood model class
        :param str file: file containing the model
        """
        self.name = file
        self.chromosomes_list = []
        self.chromosomes_models = {}
        if os.path.exists(self.name):
            if tarfile.is_tarfile(self.name):
                files = tarfile.open(self.name, "r")
                chroms = files.getnames()
                self.chromosomes_list = list(set([i.split("_")[0] for i in chroms]))

    def __getitem__(self, item):
        return self.chromosomes_models[item]

    def make(self, coverage_folder, coverage_prefix, file_no, force=False):
        """
        Make lh model for all chromosomes
        :param str coverage_folder: folder with coverage files
        :param str coverage_prefix: prefixed used for making coverage files
        :param int file_no: number of file from which model is being made
        :param bool force: if overwrite an existing model
        """
        if os.path.exists(self.name):
            if not force:
                print(
                    "Model already exists. If you want to overwrite it use force argument"
                )
                return
            else:
                print("Overwriting existing model")
        tar_arch = tarfile.open(self.name, "w")
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
                coverage_prefix,
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
        """
        Read into model specific chromosome
        """
        self.chromosomes_models[chrom] = ChromosomeModel(self.name, chrom)
        self.chromosomes_models[chrom].read()

    def read_chrom_track(self, chrom, track):
        """
        Read into model specific track for chromosome
        """
        self.chromosomes_models[chrom] = ChromosomeModel(self.name, chrom)
        self.chromosomes_models[chrom].read_track(track)

    def clear_chrom(self, chrom):
        """
        Clear model for given chromosome
        """
        self.chromosomes_models[chrom] = None


@timer_func
def main(model_file, coverage_folder, coverage_prefix, file_no=None, force=False):
    """
    Crate likelihood models for all chromosomes
    :param str model_file: output name
    :param str coverage_folder: folder with coverage files
    :param str coverage_prefix: prefix used for making coverage files
    :param int file_no: number of files used for making coverage tracks
    """
    model = ModelLH(model_file)
    model.make(coverage_folder, coverage_prefix, file_no, force)
