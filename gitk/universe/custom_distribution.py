#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from hmmlearn.base import BaseHMM
from scipy.stats import beta, nbinom
from sklearn.utils import check_random_state


def _check_and_set_n_features(model, X):
    _, n_features = X.shape
    if hasattr(model, "n_features"):
        if model.n_features != n_features:
            raise ValueError(
                f"Unexpected number of dimensions, got {n_features} but "
                f"expected {model.n_features}"
            )
    else:
        model.n_features = n_features


class NBHMM(BaseHMM):
    def __init__(
        self,
        n_components=1,
        startprob_prior=1.0,
        transmat_prior=1.0,
        failures_prior=0.0,
        prob_prior=0.0,
        failures_weight=0.0,
        prob_weight=0.0,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="strp",
        init_params="strp",
        implementation="log",
    ):
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )
        self.failures_prior = failures_prior
        self.failures_weight = failures_weight
        self.prob_prior = prob_prior
        self.prob_weight = prob_weight

    def _init(self, X):
        _check_and_set_n_features(self, X)
        super()._init(X)
        self.random_state = check_random_state(self.random_state)

        mean_X = X.mean()
        var_X = X.var()

        if self._needs_init("p", "prob_"):
            # initialize with method of moments based on X
            raise NotImplementedError

        if self._needs_init("r", "failures_"):
            # initialize with method of moments based on X
            raise NotImplementedError

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "p": nc * nf,
            "r": nc * nf,
        }

    def _check(self):
        super()._check()

        self.prob_ = np.atleast_2d(self.prob_)
        self.failures_ = np.atleast_2d(self.failures_)
        n_features = getattr(self, "n_features", self.prob_.shape[1])
        if self.prob_.shape != (self.n_components, n_features):
            raise ValueError("proba_ must have shape (n_components, n_features)")
        if self.failures_.shape != (self.n_components, n_features):
            raise ValueError("failures_ must have shape (n_components, n_features)")
        if self.failures_.shape != self.prob_.shape:
            raise ValueError(
                "failures_ and proba_ must have the same shape (n_components, n_features)"
            )
        self.n_features = n_features

    def _generate_sample_from_state(self, state, random_state):
        return random_state.negative_binomial(self.failures_[state], self.prob_[state])

    def _compute_log_likelihood(self, X):
        return np.array(
            [
                np.sum(nbinom.logpmf(X, failures, proba), axis=1)
                for failures, proba in zip(self.failures_, self.prob_)
            ]
        ).T

    def _compute_likelihood(self, X):
        return np.array(
            [
                np.prod(nbinom.pmf(X, failures, proba), axis=1)
                for failures, proba in zip(self.failures_, self.prob_)
            ]
        ).T

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats["post"] = np.zeros(self.n_components)
        stats["obs"] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, obs, lattice, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, obs, lattice, posteriors, fwdlattice, bwdlattice
        )
        raise NotImplementedError

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        raise NotImplementedError


class BetaHMM(BaseHMM):
    def __init__(
        self,
        n_components=1,
        startprob_prior=1.0,
        transmat_prior=1.0,
        alfa_prior=0.0,
        beta_prior=0.0,
        alfa_weight=0.0,
        beta_weight=0.0,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="strp",
        init_params="strp",
        implementation="log",
    ):
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )
        self.alfa_prior = alfa_prior
        self.alfa_weight = alfa_weight
        self.beta_prior = beta_prior
        self.beta_weight = beta_weight

    def _init(self, X):
        _check_and_set_n_features(self, X)
        super()._init(X)
        self.random_state = check_random_state(self.random_state)

        if self._needs_init("a", "alfa_"):
            raise NotImplementedError

        if self._needs_init("b", "beta_"):
            raise NotImplementedError

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "a": nc * nf,
            "b": nc * nf,
        }

    def _check(self):
        super()._check()

        self.alfa_ = np.atleast_2d(self.alfa_)
        self.beta_ = np.atleast_2d(self.beta_)
        n_features = getattr(self, "n_features", self.alfa_.shape[1])
        if self.alfa_.shape != (self.n_components, n_features):
            raise ValueError("alfa_ must have shape (n_components, n_features)")
        if self.beta_.shape != (self.n_components, n_features):
            raise ValueError("beta_ must have shape (n_components, n_features)")
        if self.alfa_.shape != self.beta_.shape:
            raise ValueError(
                "alfa_ and beta_ must have the same shape (n_components, n_features)"
            )
        self.n_features = n_features

    def _generate_sample_from_state(self, state, random_state):
        return random_state.beta(self.alfa_[state], self.beta_[state])

    def _compute_log_likelihood(self, X):
        return np.array(
            [
                np.sum(beta.logpdf(X, a, b), axis=1)
                for a, b in zip(self.alfa_, self.beta_)
            ]
        ).T

    def _compute_likelihood(self, X):
        return np.array(
            [np.prod(beta.pdf(X, a, b), axis=1) for a, b in zip(self.alfa_, self.beta_)]
        ).T

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        raise NotImplementedError

    def _accumulate_sufficient_statistics(
        self, stats, obs, lattice, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, obs, lattice, posteriors, fwdlattice, bwdlattice
        )
        raise NotImplementedError

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        raise NotImplementedError
