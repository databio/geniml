import multiprocessing
import os

import numpy as np
from gitk.region2vec import utils
from gitk.region2vec.region2vec_train import main as region2_train
from gitk.region2vec.region_shuffling import main as sent_gen


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def region2vec(
    token_folder,  # path to the folder of tokenized files
    save_dir,  # folder to save the training results
    file_list=None,  # specifies which files from token_folder are used for training
    data_type="files",
    mat_path=None,
    num_shufflings=1000,  # Number of shuffled datasets or number of training epochs
    num_processes=10,  # Maximum number of parallel processes
    tokenization_mode="hard",  # tokenization mode
    embedding_dim=100,  # Dimension of region2vec embeddings
    context_win_size=5,  # Context window size (half)
    save_freq=-1,  # Save a model after the given number of training epochs. If -1, then only save the best and latest models
    resume_path="",  # path to a trained model. If specified, the model will be used to initialize the region2vec embeddings
    train_alg="cbow",  # select training algorithms ['cbow','skip-gram']
    min_count=5,  # Threshold for filtering out regions with low frequency
    neg_samples=5,  # Number of negative samples
    init_lr=0.025,  # Initial learning rate
    min_lr=1e-4,  # Minimum learning rate
    lr_scheduler="linear",  # How to decay the learning rate. Select from linear and milestone
    milestones=[],  # Specify only when lr_scheduler=milestone. At each given epoch, the learning rate will be multiplied by 0.1
    hier_softmax=False,  # Whether to hierarchical softmax
    seed=0,  # random seed
):
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
        update_vocab="once",
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
