import multiprocessing
import os
from typing import List

import numpy as np

from . import utils
from .region2vec_train import main as region2_train
from .region_shuffling import main as sent_gen


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def region2vec(
    token_folder: str,
    save_dir: str,
    file_list: List[str] = None,
    data_type: str = "files",
    mat_path: str = None,
    num_shufflings: int = 1000,
    num_processes: int = 10,
    tokenization_mode: str = "hard",
    embedding_dim: int = 100,
    context_win_size: int = 5,
    save_freq: int = -1,
    resume_path: str = "",
    train_alg: str = "cbow",
    min_count: int = 5,
    neg_samples: int = 5,
    init_lr: float = 0.025,
    min_lr: float = 1e-4,
    lr_scheduler: str = "linear",
    milestones: List[int] = [],
    hier_softmax: bool = False,
    seed: int = 0,
    update_vocab: str = "once",
):
    """Trains a Region2Vec model.

    Starts two subprocesses: one that generates shuffled datasets, and the
    other consumes the shuffled datasets to train a Region2Vec model.

    Args:
        token_folder (str): The path to the folder of tokenized files.
        save_dir (str): The folder that stores the training results.
        file_list (list[str], optional): Specifies which files from
            token_folder are used for training. When None, uses all the files
            in token_folder. Defaults to None.
        data_type (str, optional): "files" or "matrix". Defaults to "files".
        mat_path (str, optional): Used only when data_type = "matrix". Defaults
            to None.
        num_shufflings (int, optional): Number of shuffled datasets to
            generate. Defaults to 1000.
        num_processes (int, optional): Number of processes used. Defaults to 10.
        tokenization_mode (str, optional): Tokenization mode. Defaults to
            "hard", i.e., concatenating all regions in a BED files in a random order.
        embedding_dim (int, optional): Dimension of embedding vectors. Defaults
            to 100.
        context_win_size (int, optional): Context window size. Defaults to 5.
        save_freq (int, optional): Save frequency. Defaults to -1.
        resume_path (str, optional): Starts with a previously trained model.
            Defaults to "".
        train_alg (str, optional): Training algorithm. Defaults to "cbow".
        min_count (int, optional): Minimum frequency required to keep a region.
            Defaults to 5.
        neg_samples (int, optional): Number of negative samples used in
            training. Defaults to 5.
        init_lr (float, optional): Initial learning rate. Defaults to 0.025.
        min_lr (float, optional): Minimum learning rate. Defaults to 1e-4.
        lr_scheduler (str, optional): Type of the learning rate scheduler.
            Defaults to "linear".
        milestones (list[int], optional): Used only when
            lr_scheduler="milestones". Defaults to [].
        hier_softmax (bool, optional): Whether to use hierarchical softmax
            during training. Defaults to False.
        seed (int, optional): Random seed. Defaults to 0.
        update_vocab (str, optional): If "every", then updates the vocabulary
            for each shuffled dataset. Defaults to "once" assuming no new
            regions occur in shuffled datasets.
    """
    timer = utils.Timer()
    start_time = timer.t()
    if file_list is None:
        files = os.listdir(token_folder)
    else:
        files = file_list
    os.makedirs(save_dir, exist_ok=True)
    file_list_path = os.path.join(save_dir, "file_list.txt")
    utils.set_log_path(save_dir)
    with open(file_list_path, "w") as f:
        for file in files:
            f.write(file)
            f.write("\n")

    training_processes = []
    num_sent_processes = min(int(np.ceil(num_processes / 2)), 4)
    nworkers = min(num_shufflings, num_sent_processes)
    utils.log(f"num_sent_processes: {nworkers}")
    if nworkers <= 1:
        sent_gen_args = Namespace(
            tokenization_folder=token_folder,
            save_dir=save_dir,
            file_list=file_list_path,
            tokenization_mode=tokenization_mode,
            pool=1,  # maximum number of unused shuffled datasets generated at a time
            worker_id=0,
            number=num_shufflings,
        )
        p = multiprocessing.Process(target=sent_gen, args=(sent_gen_args,))
        p.start()
        training_processes.append(p)
    else:
        num_arrs = [num_shufflings // nworkers] * (nworkers - 1)

        num_arrs.append(num_shufflings - np.array(num_arrs).sum())
        sent_gen_args_arr = []
        for n in range(nworkers):
            sent_gen_args = Namespace(
                tokenization_folder=token_folder,
                data_type=data_type,
                mat_path=mat_path,
                save_dir=save_dir,
                file_list=file_list_path,
                tokenization_mode=tokenization_mode,
                pool=1,  # maximum number of unused shuffled datasets generated at a time
                worker_id=n,
                number=num_arrs[n],
            )
            sent_gen_args_arr.append(sent_gen_args)
        for n in range(nworkers):
            p = multiprocessing.Process(target=sent_gen, args=(sent_gen_args_arr[n],))
            p.start()
            training_processes.append(p)

    num_region2vec_processes = max(num_processes - nworkers, 1)
    region2vec_args = Namespace(
        num_shuffle=num_shufflings,
        embed_dim=embedding_dim,
        context_len=context_win_size,
        nworkers=num_region2vec_processes,
        save_freq=save_freq,
        save_dir=save_dir,
        resume=resume_path,
        train_alg=train_alg,
        min_count=min_count,
        neg_samples=neg_samples,
        init_lr=init_lr,
        min_lr=min_lr,
        lr_mode=lr_scheduler,
        milestones=milestones,
        hier_softmax=hier_softmax,
        update_vocab=update_vocab,
        seed=seed,
    )
    p = multiprocessing.Process(target=region2_train, args=(region2vec_args,))
    p.start()
    training_processes.append(p)
    for p in training_processes:
        p.join()
    os.remove(file_list_path)
    elapsed_time = timer.t() - start_time
    print(f"[Training] {utils.time_str(elapsed_time)}/{utils.time_str(timer.t())}")
