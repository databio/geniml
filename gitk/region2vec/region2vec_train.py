import argparse
import datetime
import glob
import logging
import os
import pickle
import random
import time

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from . import utils
from .const import *

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR)


def find_dataset(data_folder):
    train_pattern = os.path.join(data_folder, "pool*[0-9]")
    count = 0
    while True:
        dsets = glob.glob(train_pattern)
        if len(dsets) == 0:
            print("No available dataset, waiting for generation...", end="\r")
            time.sleep(1)
            count += 1
            if count == MAX_WAIT_TIME:
                print("Wait time exceeds MAX_WAIT_TIME, exit")
                return -1
        else:
            return dsets[random.randint(0, len(dsets) - 1)]


def main(args):
    save_dir = args.save_dir
    data_folder = os.path.join(
        save_dir, "shuffled_datasets"
    )  # shuffled datasets are stored in the shuffled_datasets folder
    model_dir = os.path.join(save_dir, "models")  # model snapshots are stored in the models folder
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    utils.set_log_path(save_dir)  # specify the path to the log file
    utils.log(str(vars(args)))
    # ----------- configure and prepare the model -----------
    if args.train_alg == "skip-gram":
        train_alg = 1
        msg_model = "\033[94mUsing skip-gram, "
    else:
        train_alg = 0
        msg_model = "\033[94mUsing cbow, "
    if args.hier_softmax == True or args.neg_samples == 0:
        hs = 1
        msg_model += "hierarchical softmax\033[00m"
    else:
        hs = 0
        msg_model += f"negative sampling with {args.neg_samples} negative samples\033[00m"
    if not os.path.exists(args.resume):
        vocab_update = False
        model = Word2Vec(
            vector_size=args.embed_dim,
            alpha=args.init_lr,
            window=args.context_len,
            min_count=args.min_count,
            seed=args.seed,
            workers=args.nworkers,
            sg=train_alg,
            negative=args.neg_samples,
            hs=hs,
        )
        utils.log(msg_model)
    else:
        utils.log(
            f"\033[91mResuming {args.resume}, make sure the model configurations are consistent\033[00m"
        )
        model = Word2Vec.load(args.resume)
        vocab_update = True

    # ----------- train the model -----------
    loss_all = []
    if args.lr_mode == "milestone":
        lr_info = {"milestones": args.milestones, "ratio": 0.1}
    elif args.lr_mode == "linear":
        lr_info = {"freq": 1}

    lr_scheduler = utils.lr_scheduler(
        args.init_lr,
        args.min_lr,
        args.num_shuffle,
        lr_info=lr_info,
        mode=args.lr_mode,
    )

    run_timer = utils.Timer()
    cur_time = datetime.datetime.now().strftime("%x-%X")
    utils.log(f"[{cur_time}] Start training")
    utils.log(f"[{cur_time}] Building vocabulary")
    dset = find_dataset(data_folder)
    if dset == -1:
        return
    sentences = LineSentence(dset)  # create sentence iterator
    model.build_vocab(sentences, update=vocab_update)  # prepare the model vocabulary
    cur_time = datetime.datetime.now().strftime("%x-%X")
    utils.log(f"[{cur_time}]\033[93m Vocabulary size is {len(model.wv.index_to_key)}\033[00m")
    build_vocab_time = run_timer.t()
    min_loss = 1.0e100

    # start training
    for sidx in range(args.num_shuffle):
        epoch_timer = utils.Timer()
        msg = f"[Shuffling {sidx + 1:>4d}] "
        dset = find_dataset(data_folder)
        if dset == -1:
            return
        dname = dset.split("/")[-1]
        dst_name = os.path.join(data_folder, dname + "using")
        os.rename(dset, dst_name)  # change to file name to pool%dusing
        sentences = LineSentence(dst_name)  # create sentence iterator
        if args.update_vocab == "every":
            model.build_vocab(sentences, update=True)  # prepare the model vocabulary
        model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=1,
            compute_loss=True,
            start_alpha=lr_scheduler.lr,
            end_alpha=lr_scheduler.lr,
        )

        loss = model.get_latest_training_loss()
        loss_all.append(loss)
        used_name = os.path.join(data_folder, dname + "used")
        os.rename(dst_name, used_name)

        if loss < min_loss:
            min_loss = loss
            model.save(os.path.join(model_dir, "region2vec_best.pt"))
        model.save(os.path.join(model_dir, "region2vec_latest.pt"))
        if args.save_freq > 0 and (sidx + 1) % args.save_freq == 0:
            model.save(os.path.join(model_dir, f"region2vec_{sidx + 1}.pt"))
        est_time = (run_timer.t() - build_vocab_time) / (
            sidx + 1
        ) * args.num_shuffle + build_vocab_time
        msg += f"loss {loss:>12.4f} lr {lr_scheduler.lr:>5.4f} vocab_size {len(model.wv.index_to_key):>12d} ({utils.time_str(epoch_timer.t())}/{utils.time_str(est_time)})"
        utils.log(msg)
        lr_scheduler.step()

    with open(os.path.join(model_dir, "loss_all.pickle"), "wb") as f:
        pickle.dump(loss_all, f)

    elasped_time = run_timer.t()
    cur_time = datetime.datetime.now().strftime("%x-%X")
    utils.log(f"[{cur_time}] Training finished, training Time {utils.time_str(elasped_time)}")
    # remove intermediate datasets
    os.system(f"rm -rf {data_folder}")  # remove the generated shuffled datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene Embedding")
    parser.add_argument("--num-shuffle", type=int, help="number of shuffled datasets")
    parser.add_argument("--embed-dim", type=int, help="embedding dimension")
    parser.add_argument("--context-len", type=int, help="window size")
    parser.add_argument("--nworkers", type=int, help="number of workers")
    parser.add_argument("--save-freq", type=int, default=0, help="save frequency")
    parser.add_argument("--save-dir", help="path to the folder that saves the training result")
    parser.add_argument("--resume", help="path to a saved model")
    parser.add_argument("--train-alg", help="training algorithm, select from [cbow, skip-gram]")
    parser.add_argument(
        "--min-count",
        type=int,
        help="threshold of pruning the internal vocabulary",
    )
    parser.add_argument(
        "--neg-samples",
        type=int,
        help="number of noise words in negative sampling, usually between 5-20",
    )
    parser.add_argument(
        "--hier-softmax",
        default=False,
        action="store_true",
        help="if given, hierarchical softmax will be used",
    )
    parser.add_argument("--init-lr", type=float, help="initial learning rate")
    parser.add_argument("--milestones", nargs="+", type=int, default=[100, 200])
    parser.add_argument(
        "--lr-mode",
        type=str,
        choices=["milestone", "linear"],
        help="type of learning rate scheduler, milestone or linear",
    )
    parser.add_argument(
        "--update-vocab",
        type=str,
        default="once",
        help="[every] update at every epoch; [once] Update once since the vocabulary does not change",
    )
    parser.add_argument("--min-lr", type=float, help="minimum learning rate")
    parser.add_argument("--seed", type=int, help="random seed")
    args = parser.parse_args()
    main(args)
