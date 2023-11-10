import multiprocessing
import os
from logging import getLogger
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track
from huggingface_hub import hf_hub_download
from yaml import safe_dump, safe_load

from ..io import Region, RegionSet
from ..tokenization.main import ITTokenizer, Tokenizer
from . import utils
from .utils import LearningRateScheduler, shuffle_documents
from .const import (
    MODULE_NAME,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_EMBEDDING_SIZE,
    DEFAULT_MIN_COUNT,
    DEFAULT_EPOCHS,
    UNIVERSE_FILE_NAME,
    MODEL_FILE_NAME,
    CONFIG_FILE_NAME,
    POOLING_TYPES,
)
from .region2vec_train import main as region2_train
from .region_shuffling import main as sent_gen

_GENSIM_LOGGER = getLogger("gensim")
_LOGGER = getLogger(MODULE_NAME)

# demote gensim logger to warning
_GENSIM_LOGGER.setLevel("WARNING")


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


class Word2Vec(nn.Module):
    """
    Word2Vec model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_SIZE,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.projection = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


class Region2Vec(Word2Vec):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_SIZE,
    ):
        super().__init__(vocab_size, embedding_dim)


class Region2VecExModel:
    def __init__(
        self,
        model_path: str = None,
        tokenizer: ITTokenizer = None,
        device: str = None,
        **kwargs,
    ):
        """
        Initialize Region2VecExModel.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param embedding_dim: Dimension of the embedding.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        super().__init__()
        self.model_path: str = model_path
        self.tokenizer: ITTokenizer
        self.trained: bool = False
        self._model: Region2Vec = None

        if model_path is not None:
            self._init_from_huggingface(model_path)
            self.trained = True

        elif tokenizer is not None:
            self._init_model(tokenizer, **kwargs)

        # set the device
        self._target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _init_tokenizer(self, tokenizer: Union[ITTokenizer, str]):
        """
        Initialize the tokenizer.

        :param tokenizer: Tokenizer to initialize.
        """
        if isinstance(tokenizer, str):
            if os.path.exists(tokenizer):
                self.tokenizer = ITTokenizer(tokenizer)
            else:
                self.tokenizer = ITTokenizer.from_pretrained(
                    tokenizer
                )  # download from huggingface (or at least try to)
        elif isinstance(tokenizer, ITTokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError("tokenizer must be a path to a bed file or an ITTokenizer object.")

    def _init_model(self, tokenizer, **kwargs):
        """
        Initialize the core model. This will initialize the model from scratch.

        :param kwargs: Additional keyword arguments to pass to the model.
        """
        self._init_tokenizer(tokenizer)
        self._vocab_length = len(self.tokenizer)
        self._model = Region2Vec(
            len(self.tokenizer),
            embedding_dim=kwargs.get("embedding_dim", DEFAULT_EMBEDDING_SIZE),
        )

    @property
    def model(self):
        """
        Get the core Region2Vec model.
        """
        return self._model

    def add_tokenizer(self, tokenizer: Tokenizer, **kwargs):
        """
        Add a tokenizer to the model. This should be use when the model
        is not initialized with a tokenizer.

        :param tokenizer: Tokenizer to add to the model.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        if self._model is not None:
            raise RuntimeError("Cannot add a tokenizer to a model that is already initialized.")

        self.tokenizer = tokenizer
        if not self.trained:
            self._init_model(**kwargs)

    def _load_local_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load the model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """
        self._model_path = model_path
        self._universe_path = vocab_path

        # init the tokenizer - only one option for now
        self.tokenizer = ITTokenizer(vocab_path)

        # load the model state dict (weights)
        params = torch.load(model_path)

        # get the model config (vocab size, embedding size)
        with open(config_path, "r") as f:
            config = safe_load(f)

        self._model = Region2Vec(
            config["vocab_size"],
            embedding_dim=config["embedding_size"],
        )
        self._model.load_state_dict(params)

    def _init_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the model from a huggingface model. This uses the model path
        to download the necessary files and then "build itself up" from those. This
        includes both the actual model and the tokenizer.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        :param kwargs: Additional keyword arguments to pass to the hf download function.
        """
        model_file_path = hf_hub_download(model_path, model_file_name, **kwargs)
        universe_path = hf_hub_download(model_path, universe_file_name, **kwargs)
        config_path = hf_hub_download(model_path, config_file_name, **kwargs)

        self._load_local_model(model_file_path, universe_path, config_path)

    @classmethod
    def from_pretrained(
        cls,
        path_to_files: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
    ) -> "Region2VecExModel":
        """
        Load the model from a set of files that were exported using the export function.

        :param str path_to_files: Path to the directory containing the files.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        """
        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_file_path = os.path.join(path_to_files, universe_file_name)
        config_file_path = os.path.join(path_to_files, config_file_name)

        instance = cls()
        instance._load_local_model(model_file_path, universe_file_path, config_file_path)
        instance.trained = True

        return instance

    def _validate_data_for_training(
        self, data: Union[List[RegionSet], List[str], List[List[Region]]]
    ) -> List[RegionSet]:
        """
        Validate the data for training. This will return a list of RegionSets if the data is valid.

        :param Union[List[RegionSet], List[str]] data: List of data to train on. This is either
                                                       a list of RegionSets or a list of paths to bed files.
        :return: List of RegionSets.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list or RegionSets or a list of paths to bed files.")
        if len(data) == 0:
            raise ValueError("data must not be empty.")

        # check if the data is a list of RegionSets
        if isinstance(data[0], RegionSet):
            return data
        elif isinstance(data[0], str):
            return [RegionSet(f) for f in data]
        elif isinstance(data[0], list) and isinstance(data[0][0], Region):
            return [RegionSet([r for r in region_list]) for region_list in data]

    def train(
        self,
        data: Union[List[RegionSet], List[str]],
        window_size: int = DEFAULT_WINDOW_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        min_count: int = DEFAULT_MIN_COUNT,
        num_cpus: int = 1,
        seed: int = 42,
        checkpoint_path: str = MODEL_FILE_NAME,
        save_model: bool = False,
        gensim_params: dict = {},
    ) -> np.ndarray:
        """
        Train the model.

        :param Union[List[RegionSet], List[str]] data: List of data to train on. This is either
                                                        a list of RegionSets or a list of paths to bed files.
        :param int window_size: Window size for the model.
        :param int epochs: Number of epochs to train for.
        :param int min_count: Minimum count for a region to be included in the vocabulary.
        :param int n_shuffles: Number of shuffles to perform on the data.
        :param int batch_size: Batch size for training.
        :param str checkpoint_path: Path to save the model checkpoint to.
        :param torch.optim.Optimizer optimizer: Optimizer to use for training.
        :param float learning_rate: Learning rate to use for training.
        :param int ns_k: Number of negative samples to use.
        :param torch.device device: Device to use for training.
        :param dict optimizer_params: Additional parameters to pass to the optimizer.
        :param bool save_model: Whether or not to save the model.

        :return np.ndarray: Loss values for each epoch.
        """
        # we only need gensim if we are training
        from gensim.models import Word2Vec as GensimWord2Vec

        # validate a model exists
        if self._model is None:
            raise RuntimeError(
                "Cannot train a model that has not been initialized. Please initialize the model first using a tokenizer or from a huggingface model."
            )

        # validate the data - convert all to RegionSets
        _LOGGER.info("Validating data for training.")
        data = self._validate_data_for_training(data)

        # create gensim model that will be used to train
        _LOGGER.info("Creating gensim model.")
        gensim_model = GensimWord2Vec(
            vector_size=self._model.embedding_dim,
            window=window_size,
            min_count=min_count,
            workers=num_cpus,
            seed=seed,
            **gensim_params,
        )

        # tokenize the data
        # convert to strings for gensim
        _LOGGER.info("Tokenizing data.")
        tokenized_data = [
            self.tokenizer.tokenize(list(region_set))
            for region_set in track(data, description="Tokenizing data", total=len(data))
            if len(region_set) > 0
        ]

        _LOGGER.info("Building vocabulary.")
        tokenized_data = [
            [str(t.id) for t in region_set]
            for region_set in track(
                tokenized_data,
                total=len(tokenized_data),
                description="Converting to strings.",
            )
        ]
        gensim_model.build_vocab(tokenized_data)

        # create our own learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            n_epochs=epochs,
        )

        # train the model
        losses = []

        for epoch in track(range(epochs), description="Training model", total=epochs):
            # shuffle the data
            _LOGGER.info(f"Starting epoch {epoch+1}.")
            _LOGGER.info("Shuffling data.")
            shuffled_data = shuffle_documents(
                tokenized_data, n_shuffles=1
            )  # shuffle once per epoch, no need to shuffle more
            gensim_model.train(
                shuffled_data,
                epochs=1,  # train for 1 epoch at a time, shuffle data each time
                compute_loss=True,
                total_words=gensim_model.corpus_total_words,
            )

            # log out and store loss
            _LOGGER.info(f"Loss: {gensim_model.get_latest_training_loss()}")
            losses.append(gensim_model.get_latest_training_loss())

            # update the learning rate
            lr_scheduler.update()

        # once done training, set the weights of the pytorch model in self._model
        for id in track(
            gensim_model.wv.key_to_index,
            total=len(gensim_model.wv.key_to_index),
            description="Setting weights.",
        ):
            self._model.projection.weight.data[int(id)] = torch.tensor(gensim_model.wv[id])

        # set the model as trained
        self.trained = True

        # export
        if save_model:
            self.export(checkpoint_path)

        return np.array(losses)

    def export(
        self,
        path: str,
        checkpoint_file: str = MODEL_FILE_NAME,
        universe_file: str = UNIVERSE_FILE_NAME,
        config_file: str = CONFIG_FILE_NAME,
    ):
        """
        Function to facilitate exporting the model in a way that can
        be directly uploaded to huggingface. This exports the model
        weights and the vocabulary.

        :param str path: Path to export the model to.
        """
        # make sure the model is trained
        if not self.trained:
            raise RuntimeError("Cannot export an untrained model.")

        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # export the model weights
        torch.save(self._model.state_dict(), os.path.join(path, checkpoint_file))

        # export the vocabulary
        with open(os.path.join(path, universe_file), "a") as f:
            for region in self.tokenizer.universe.regions:
                f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

        # export the config (vocab size, embedding size)
        config = {
            "vocab_size": len(self.tokenizer),
            "embedding_size": self._model.embedding_dim,
        }

        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(config, f)

    def encode(
        self, regions: Union[Region, List[Region]], pooling: POOLING_TYPES = "mean"
    ) -> np.ndarray:
        """
        Get the vector for a region.

        :param Region region: Region to get the vector for.
        :param str pooling: Pooling type to use.

        :return np.ndarray: Vector for the region.
        """
        # data validation
        if isinstance(regions, Region):
            regions = [regions]
        if isinstance(regions, str):
            regions = list(RegionSet(regions))
        if isinstance(regions, RegionSet):
            regions = list(regions)
        if not isinstance(regions, list):
            regions = [regions]
        if not isinstance(regions[0], Region):
            raise TypeError("regions must be a list of Region objects.")

        if pooling not in ["mean", "max"]:
            raise ValueError(f"pooling must be one of {POOLING_TYPES}")

        # tokenize the region
        tokens = [self.tokenizer.tokenize(r) for r in regions]
        tokens = [t.id for sublist in tokens for t in sublist]

        # get the vector
        region_embeddings = self._model.projection(torch.tensor(tokens))

        if pooling == "mean":
            return torch.mean(region_embeddings, axis=0).detach().numpy()
        elif pooling == "max":
            return torch.max(region_embeddings, axis=0).values.detach().numpy()
        else:
            # this should be unreachable
            raise ValueError(f"pooling must be one of {POOLING_TYPES}")
