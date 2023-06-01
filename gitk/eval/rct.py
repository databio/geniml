import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
import time

import numpy as np
import sklearn.neural_network as nn
from gensim.models import Word2Vec
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..utils import timer_func
from .utils import Timer, cosine_distance, genome_distance, load_genomic_embeddings


def get_rct_score(
    path, embed_type, bin_path, out_dim=-1, cv_num=5, seed=42, num_workers=10
):
    embed_rep, vocab = load_genomic_embeddings(path, embed_type)
    embed_bin, vocab_bin = load_genomic_embeddings(bin_path, "base")
    region2idx = {r: i for i, r in enumerate(vocab)}
    region2idx_bin = {r: i for i, r in enumerate(vocab_bin)}
    # align embed_bin with embed_rep
    if out_dim <= 0:
        embed_bin = np.array([embed_bin[region2idx_bin[v]] for v in vocab])
    else:
        bin_dim = embed_bin.shape[1]
        out_dim = min(bin_dim, out_dim)
        sel_dims = np.random.choice(bin_dim, out_dim)
        embed_bin = np.array([embed_bin[region2idx_bin[v]][sel_dims] for v in vocab])

    regressor = nn.MLPRegressor(
        hidden_layer_sizes=(200),
        activation="relu",
        solver="adam",
        alpha=0.0001,  # regularizer strength
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=seed,
        tol=0.0001,
        verbose=False,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    model_in = make_pipeline(StandardScaler(), regressor)
    model = TransformedTargetRegressor(regressor=model_in, transformer=StandardScaler())

    kf = KFold(n_splits=cv_num, shuffle=True, random_state=seed)
    if num_workers > cv_num:
        num_workers = cv_num
    score = cross_val_score(
        model, embed_rep, embed_bin, cv=kf, n_jobs=num_workers, verbose=0
    )
    return score.mean()


def reconstruction_batch(
    batch, cv_num, out_dim=-1, seed=42, save_path=None, num_workers=10
):
    rct_arr = []
    for path, embed_type, bin_path in batch:
        score = get_rct_score(
            path, embed_type, bin_path, out_dim, cv_num, seed, num_workers
        )
        rct_arr.append((path, score))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(rct_arr, f)
    return rct_arr


def rct_eval(
    batch,
    num_runs=5,
    cv_num=5,
    out_dim=-1,
    save_folder=None,
    num_workers=10,
):
    results_seeds = []
    for seed in range(num_runs):
        print(f"----------------Run {seed}----------------")
        save_path = (
            os.path.join(save_folder, f"rct_eval_seed{seed}") if save_folder else None
        )
        result_list = reconstruction_batch(
            batch, cv_num, out_dim, seed, save_path, num_workers
        )
        results_seeds.append(result_list)

    rct_res = [[] for i in range(len(batch))]
    for results in results_seeds:
        for i, res in enumerate(results):
            rct_res[i].append(res[1])
            assert res[0] == batch[i][0], "key == batch[i][0]"
    mean_rct = [np.array(r).mean() for r in rct_res]
    std_rct = [np.array(r).std() for r in rct_res]
    models = [t[0] for t in batch]
    for i in range(len(mean_rct)):
        print(f"{batch[i][0]}\n RCT (std): {mean_rct[i]:.4f} ({std_rct[i]:.4f}) \n")
    rct_arr = [(batch[i][0], rct_res[i]) for i in range(len(batch))]
    return rct_arr
