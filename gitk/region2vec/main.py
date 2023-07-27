import multiprocessing
import os
from logging import getLogger
from typing import List, Union

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from numba import config

from ..io import Region, RegionSet
from . import utils
from .const import *
from .region2vec_train import main as region2_train
from .region_shuffling import main as sent_gen

_GENSIM_LOGGER = getLogger("gensim")
_LOGGER = getLogger(MODULE_NAME)

# demote gensim logger to warning
_GENSIM_LOGGER.setLevel("WARNING")

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"


class ReportLossCallback(CallbackAny2Vec):
    """
    Callback to report loss after each epoch.
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model: Word2Vec):
        loss = model.get_latest_training_loss()
        _LOGGER.info(f"Epoch {self.epoch} complete. Loss: {loss}")
        self.epoch += 1


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


class Region2Vec(Word2Vec):
    # not needed since self.load() seems to work fine. I am not sure why, though.
    @staticmethod
    def from_word2vec_model(model: Word2Vec, **kwargs):
        r2v = Region2Vec(
            callbacks=model.callbacks or kwargs.get("callbacks"),
            window_size=model.window,
            vector_size=model.vector_size,
            min_count=model.min_count,
            threads=model.workers,  # assuming it stores thread info
            seed=model.random.seed,  # assuming it stores seed
        )
        # attach the word2vec model to the region2vec model
        r2v.wv = model.wv
        return r2v

    def __init__(self, **kwargs):
        """
        A class that implements Region2Vec. Region2Vec is a model that learns
        embedding vectors for genomic regions. It works by by considering each
        region set as a text document, with each region as a word inside that
        document.

        :param window_size: The size of the window to use for the skip-gram
            model. Defaults to 5.
        :param vector_size: The size of the embedding vectors. Defaults to 100.
        :param min_count: The minimum number of times a region must occur in
            the dataset to be included in the vocabulary. Defaults to 5.
        :param threads: The number of threads to use for training. Defaults to
            the number of CPUs - 2.
        :param seed: The seed to use for training. Defaults to 42.
        :param callbacks: A list of callbacks to use for training. Defaults to
            an empty list.
        """
        self.trained = False
        self.callbacks = kwargs.get("callbacks") or []

        # instantiate the Word2Vec model
        super().__init__(
            window=kwargs.get("window_size") or DEFAULT_WINDOW_SIZE,
            vector_size=kwargs.get("vector_size") or DEFAULT_EMBEDDING_SIZE,
            min_count=kwargs.get("min_count") or DEFAULT_MIN_COUNT,
            workers=kwargs.get("threads") or multiprocessing.cpu_count() - 2,
            seed=kwargs.get("seed") or 42,  #
            callbacks=kwargs.get("callbacks") or [],
        )

    def train(
        self,
        data: Union[List[str], List[RegionSet], List[List[Region]]],
        epochs: int = DEFAULT_EPOCHS,  # training cycles
        n_shuffles: int = DEFAULT_N_SHUFFLES,  # not the number of traiing cycles, actual shufle num
        report_loss: bool = True,
        lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        lr_schedule: Union[str, utils.ScheduleType] = "linear",
    ):
        """
        Train the model. This is done in two steps: First, we shuffle the documents.
        Second, we train the model.

        :param int epochs: The number of epochs to train for (note: this is the number of times regions are shuffled, then fed to the model for training).
        :param int n_shuffles: The number of times to shuffle the regions within each document.
        :param int gensim_epochs: The number of epochs to train for within each shuffle (or main epoch).
        :param bool report_loss: Whether or not to report the loss after each epoch.
        :param float lr: The initial learning rate.
        :param float min_lr: The minimum learning rate.
        :param Union[str, ScheduleType] lr_schedule: The learning rate schedule to use.
        """
        # force to 1 for now (see: https://github.com/databio/gitk/pull/20#discussion_r1205683978)
        n_shuffles = 1

        # verify data is a list of strings, region sets or a list of list of regions
        if not isinstance(data, list):
            raise TypeError(
                f"Data must be a list of strings, RegionSets, or a list of list of Regions. Got {type(data)}."
            )
        else:
            # list of strings? files?
            if all([isinstance(d, str) for d in data]):
                data = [RegionSet(d) for d in data]
            # list of RegionSets? Great.
            elif all([isinstance(d, RegionSet) for d in data]):
                pass
            # list of list of Regions? Great.
            elif all([isinstance(d, list) for d in data]) and all(
                [isinstance(d[0], Region) for d in data]
            ):
                data = [RegionSet(d) for d in data]
            # something else? error.
            else:
                raise TypeError(
                    f"Data must be a list of strings, RegionSets, or a list of list of Regions. Got {type(data)}."
                )

        if report_loss:
            self.callbacks.append(ReportLossCallback())

        # create a learning rate scheduler
        lr_scheduler = utils.LearningRateScheduler(
            init_lr=lr, min_lr=min_lr, type=lr_schedule, n_epochs=epochs
        )

        # "wordify" the regions
        region_sets = [utils.wordify_regions(rs) for rs in data]

        # train the model using these shuffled documents
        _LOGGER.info("Training starting.")

        # build up the vocab
        super().build_vocab(
            region_sets,
            update=False if not self.trained else True,
            min_count=self.min_count,
        )

        for shuffle_num in range(epochs):
            # update current values
            current_lr = lr_scheduler.get_lr()
            current_loss = self.get_latest_training_loss()

            # update user
            _LOGGER.info(f"SHUFFLE {shuffle_num} - lr: {current_lr}, loss: {current_loss}")
            _LOGGER.info("Shuffling documents.")

            # shuffle regions
            region_sets = utils.shuffle_documents(region_sets, n_shuffles=n_shuffles)

            # train the model on one iteration
            super().train(
                region_sets,
                total_examples=len(region_sets),
                epochs=1,  # for to 1 for now (see: https://github.com/databio/gitk/pull/20#discussion_r1205692089)
                callbacks=self.callbacks,
                compute_loss=report_loss,
                start_alpha=current_lr,
            )

            # update learning rates
            lr_scheduler.update()

            self.trained = True

    def save(self, filepath: str):
        """
        Save the current model to disk

        :param str filepath: The path to save the model to.
        """
        super().save(filepath)

    @classmethod
    def load(cls, filepath: str) -> "Region2Vec":
        """
        Load a model from disk. This should return a
        Region2Vec object.

        :param str filepath: The path to load the model from.
        """
        # I feel like this shouldnt work, but it does?
        return super().load(filepath)
        # return cls.from_word2vec_model(model)

    def forward(self, regions: Union[Region, RegionSet, List[Region]]) -> np.ndarray:
        """
        Get the embedding vector(s) for a given region or region set.

        :param Union[Region, RegionSet, List[Region]] regions: The region(s) to get the embedding vector(s) for.
        """
        if isinstance(regions, Region):
            region_word = utils.wordify_region(regions)
            return self.wv[region_word]
        elif isinstance(regions, RegionSet):
            region_words = utils.wordify_regions(regions)
            return np.mean([self.wv[r] for r in region_words], axis=0)
        elif isinstance(regions, list):
            region_words = [utils.wordify_region(r) for r in regions]
            return np.mean([self.wv[r] for r in region_words], axis=0)
        else:
            raise TypeError(
                f"Regions must be of type Region, RegionSet, or list, not {type(regions).__name__}"
            )

    def __call__(self, regions: Union[Region, RegionSet, List[Region]]) -> np.ndarray:
        """
        Get the embedding vector(s) for a given region or region set.

        :param Union[Region, RegionSet, List[Region]] regions: The region(s) to get the embedding vector(s) for.
        """
        return self.forward(regions)

    def __repr__(self) -> str:
        return f"Region2Vec(window={self.window}, vector_size={self.vector_size})"
