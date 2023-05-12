#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hmmlearn import hmm
import numpy as np
import os
from .custom_distribution import NBHMM, BetaHMM
from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract class of HMM models"""

    def __init__(self, trans_matrix, init_para, para, save_matrix):
        """

        :param array trans_matrix: transition matrix
        :param str init_para: parameters that need to be initialized before training
        :param str para: parameters that are update during training
        :param bool save_matrix: whether to save to a file transition matrix
        """
        self.init_para = init_para
        self.para = para
        self.state_no = 4
        self.save_matrix = save_matrix
        self.trans_matrix = trans_matrix
        self.start_matrix = np.array([0.01, 0.01, 0.97, 0.01])

    @abstractmethod
    def make(self):
        """Abstract method for defining the model"""
        pass

    def save_tras(self, out_folder):
        np.savetxt(os.path.join(out_folder, "trans_matrix.csv"), self.trans_matrix)


class PoissonModel(Model):
    """HMM model with Poisson emissions"""

    def __init__(
        self,
        trans_matrix,
        lambdas_matrix,
        init_para="",
        para="",
        save_matrix=True,
        out_folder="",
    ):
        """

        :param lambdas_matrix: Lambda values for emission probabilities
        :param out_folder: folder to which save parameter matrix
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
                os.path.join(out_folder, "lambdas_matrix.csv"), self.lambdas_matrix
            )

    def make(self):
        """Initialize HMM model"""
        model = hmm.PoissonHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        if "t" not in self.init_para:
            model.transmat_ = self.trans_matrix
        if "l" not in self.init_para:
            model.lambdas_ = self.lambdas_matrix
        model.startprob_ = self.start_matrix
        return model


class GaussianModel(Model):
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
        """

        :param means_matrix: mean values for emission probabilities
        :param covars_matrix: covariance values for emission probabilities
        :param out_folder: folder to which save parameter matrix
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
                os.path.join(out_folder, "covars_matrix.csv"), self.covars_matrix
            )
            np.savetxt(os.path.join(out_folder, "means_matrix.csv"), self.means_matrix)

    def make(self):
        """Initialize HMM model"""
        model = hmm.GaussianHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        if "t" not in self.init_para:
            model.transmat_ = self.trans_matrix
        if "m" not in self.init_para:
            model.means_ = self.means_matrix
        if "c" not in self.init_para:
            model.covars_ = self.covars_matrix
        model.startprob_ = self.start_matrix
        return model


class NBModel(Model):
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
        """

        :param failures_matrix: number of failures for emission probabilities
        :param prob_matrix: success probability for emission probabilities
        :param out_folder: folder to which save parameter matrix
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
                os.path.join(failures_matrix, "failures_matrix.csv"), self.covars_matrix
            )
            np.savetxt(os.path.join(out_folder, "prob_matrix.csv"), self.prob_matrix)

    def make(self):
        """Initialize HMM model"""
        model = NBHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        model.transmat_ = self.trans_matrix
        model.failures_ = self.failures_matrix
        model.prob_ = self.prob_matrix
        model.startprob_ = self.start_matrix
        return model


class BetaModel(Model):
    def __init__(
        self,
        trans_matrix,
        alfa_matrix,
        beta_matrix,
        init_para="",
        para="",
        save_matrix=True,
        out_folder="",
    ):
        """

        :param alfa_matrix: alfa values for emission probabilities
        :param beta_matrix: beta values for emission probabilities
        :param out_folder: folder to which save parameter matrix
        """
        Model.__init__(
            self,
            trans_matrix=trans_matrix,
            init_para=init_para,
            para=para,
            save_matrix=save_matrix,
        )
        self.alfa_matrix = alfa_matrix
        self.beta_matrix = beta_matrix
        if save_matrix:
            super().save_tras(out_folder)
            np.savetxt(os.path.join(out_folder, "alfa_matrix.csv"), self.alfa_matrix)
            np.savetxt(os.path.join(out_folder, "beta_matrix.csv"), self.beta_matrix)

    def make(self):
        """Initialize HMM model"""
        model = BetaHMM(
            n_components=self.state_no,
            verbose=True,
            init_params=self.init_para,
            params=self.para,
        )

        model.transmat_ = self.trans_matrix
        model.alfa_ = self.alfa_matrix
        model.beta_ = self.beta_matrix
        model.startprob_ = self.start_matrix
        return model
