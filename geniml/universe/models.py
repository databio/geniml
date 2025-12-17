#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from abc import ABC

import numpy as np
from hmmlearn import hmm

from .custom_distribution import NBHMM, BetaHMM


class Model(ABC):
    """Abstract class of HMM models."""

    def __init__(self, trans_matrix, init_para, para, save_matrix):
        """Initialize HMM model.

        Args:
            trans_matrix (ndarray): Transition matrix.
            init_para (str): Parameters that need to be initialized before training.
            para (str): Parameters that are updated during training.
            save_matrix (bool): Whether to save transition matrix to a file.
        """
        self.init_para = init_para
        self.para = para
        self.state_no = 4
        self.save_matrix = save_matrix
        self.trans_matrix = trans_matrix
        self.start_matrix = np.array([0.01, 0.01, 0.97, 0.01])

    def save_tras(self, out_folder):
        np.savetxt(os.path.join(out_folder, "trans_matrix.csv"), self.trans_matrix)


class PoissonModel(Model):
    """HMM model with Poisson emissions."""

    def __init__(
        self,
        trans_matrix,
        lambdas_matrix,
        init_para="",
        para="",
        save_matrix=True,
        out_folder="",
    ):
        """Initialize Poisson HMM model.

        Args:
            trans_matrix (ndarray): Transition matrix.
            lambdas_matrix (ndarray): Lambda values for emission probabilities.
            init_para (str): Parameters to initialize.
            para (str): Parameters to update during training.
            save_matrix (bool): Whether to save parameter matrix.
            out_folder (str): Folder to which save parameter matrix.
        """
        Model.__init__(
            self,
            trans_matrix=trans_matrix,
            init_para=init_para,
            para=para,
            save_matrix=save_matrix,
        )
        self.lambdas_matrix = lambdas_matrix
        if save_matrix:
            super().save_tras(out_folder)
            np.savetxt(
                os.path.join(out_folder, "lambdas_matrix.csv"),
                self.lambdas_matrix,
            )

        self.model = hmm.PoissonHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        if "t" not in self.init_para:
            self.model.transmat_ = self.trans_matrix
        if "l" not in self.init_para:
            self.model.lambdas_ = self.lambdas_matrix
        self.model.startprob_ = self.start_matrix


class GaussianModel(Model):
    """HMM model with Gaussian emissions."""

    def __init__(
        self,
        trans_matrix,
        means_matrix,
        covars_matrix,
        init_para="",
        para="",
        save_matrix=True,
        out_folder="",
    ):
        """Initialize Gaussian HMM model.

        Args:
            trans_matrix (ndarray): Transition matrix.
            means_matrix (ndarray): Mean values for emission probabilities.
            covars_matrix (ndarray): Covariance values for emission probabilities.
            init_para (str): Parameters to initialize.
            para (str): Parameters to update during training.
            save_matrix (bool): Whether to save parameter matrix.
            out_folder (str): Folder to which save parameter matrix.
        """
        Model.__init__(
            self,
            trans_matrix=trans_matrix,
            init_para=init_para,
            para=para,
            save_matrix=save_matrix,
        )
        self.means_matrix = means_matrix
        self.covars_matrix = covars_matrix
        if save_matrix:
            super().save_tras(out_folder)
            np.savetxt(
                os.path.join(out_folder, "covars_matrix.csv"),
                self.covars_matrix,
            )
            np.savetxt(os.path.join(out_folder, "means_matrix.csv"), self.means_matrix)

        self.model = hmm.GaussianHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        if "t" not in self.init_para:
            self.model.transmat_ = self.trans_matrix
        if "m" not in self.init_para:
            self.model.means_ = self.means_matrix
        if "c" not in self.init_para:
            self.model.covars_ = self.covars_matrix
        self.model.startprob_ = self.start_matrix


class NBModel(Model):
    """HMM model with Negative Binomial emissions."""

    def __init__(
        self,
        trans_matrix,
        failures_matrix,
        prob_matrix,
        init_para="",
        para="",
        save_matrix=True,
        out_folder="",
    ):
        """Initialize Negative Binomial HMM model.

        Args:
            trans_matrix (ndarray): Transition matrix.
            failures_matrix (ndarray): Number of failures for emission probabilities.
            prob_matrix (ndarray): Success probability for emission probabilities.
            init_para (str): Parameters to initialize.
            para (str): Parameters to update during training.
            save_matrix (bool): Whether to save parameter matrix.
            out_folder (str): Folder to which save parameter matrix.
        """
        Model.__init__(
            self,
            trans_matrix=trans_matrix,
            init_para=init_para,
            para=para,
            save_matrix=save_matrix,
        )
        self.failures_matrix = failures_matrix
        self.prob_matrix = prob_matrix
        if save_matrix:
            super().save_tras(out_folder)
            np.savetxt(
                os.path.join(failures_matrix, "failures_matrix.csv"),
                self.covars_matrix,
            )
            np.savetxt(os.path.join(out_folder, "prob_matrix.csv"), self.prob_matrix)

        self.model = NBHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        self.model.transmat_ = self.trans_matrix
        self.model.failures_ = self.failures_matrix
        self.model.prob_ = self.prob_matrix
        self.model.startprob_ = self.start_matrix


class BetaModel(Model):
    """HMM model with Beta emissions."""

    def __init__(
        self,
        trans_matrix,
        alpha_matrix,
        beta_matrix,
        init_para="",
        para="",
        save_matrix=True,
        out_folder="",
    ):
        """Initialize Beta HMM model.

        Args:
            trans_matrix (ndarray): Transition matrix.
            alpha_matrix (ndarray): Alpha values for emission probabilities.
            beta_matrix (ndarray): Beta values for emission probabilities.
            init_para (str): Parameters to initialize.
            para (str): Parameters to update during training.
            save_matrix (bool): Whether to save parameter matrix.
            out_folder (str): Folder to which save parameter matrix.
        """
        Model.__init__(
            self,
            trans_matrix=trans_matrix,
            init_para=init_para,
            para=para,
            save_matrix=save_matrix,
        )
        self.alpha_matrix = alpha_matrix
        self.beta_matrix = beta_matrix
        if save_matrix:
            super().save_tras(out_folder)
            np.savetxt(os.path.join(out_folder, "alpha_matrix.csv"), self.alpha_matrix)
            np.savetxt(os.path.join(out_folder, "beta_matrix.csv"), self.beta_matrix)

        self.model = BetaHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        self.model.transmat_ = self.trans_matrix
        self.model.alfa_ = self.alpha_matrix
        self.model.beta_ = self.beta_matrix
        self.model.startprob_ = self.start_matrix
