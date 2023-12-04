import os
import logging
from typing import Union, List

import torch
import torch.nn as nn
import scanpy as sc

from rich.progress import Progress
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from .const import (
    MODULE_NAME,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
    CONFIG_FILE_NAME,
    DEFAULT_LABEL_KEY,
    DEFAULT_FINE_TUNE_LOSS_FN,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_OPTIMIZER,
    DEFAULT_TEST_TRAIN_SPLIT,
)
from .utils import (
    generate_fine_tuning_dataset,
    collate_finetuning_batch,
    FineTuneTrainingResult,
    FineTuningDataset,
)

from ..tokenization.main import ITTokenizer
from ..region2vec.main import Region2Vec
from ..region2vec.const import (
    DEFAULT_EMBEDDING_DIM,
    POOLING_TYPES,
    POOLING_METHOD_KEY,
)
from ..region2vec.utils import export_region2vec_model, load_local_region2vec_model

_LOGGER = logging.getLogger(MODULE_NAME)


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=1)


class RegionSet2Vec(nn.Module):
    def __init__(self, region2vec: Region2Vec, pooling: POOLING_TYPES = "mean"):
        """
        Initialize the RegionSet2Vec. RegionSet2Vec is a wrapper around the Region2Vec model that allows
        pooling over a set of regions. This is useful for classification tasks where the input is a set of
        regions, such as classifying a cell type based on the set of regions that are accessible.

        :param Union[Region2Vec, str] region2vec: Either a Region2Vec instance or a path to a huggingface model.
        :param POOLING_TYPES pooling: The pooling type to use. Either "mean" or "max".
        """
        super().__init__()

        self.region2vec: Region2Vec = region2vec
        # assign the pooling layer based on mean or max
        if pooling == "mean":
            self.pooling = MeanPooling()
        elif pooling == "max":
            self.pooling = MaxPooling()
        else:
            raise ValueError(f"Invalid pooling type {pooling} passed.")

    def forward(self, x) -> torch.Tensor:
        x = self.region2vec(x)
        x = self.pooling(x)
        return x


class Region2VecFineTuner:
    def __init__(
        self,
        model_path: str = None,
        tokenizer: ITTokenizer = None,
        region2vec: Union[Region2Vec, str] = None,
        device: str = None,
        pooling: POOLING_TYPES = "mean",
        **kwargs,
    ):
        """
        Initialize the Region2Vec fine tuning model from a huggingface model or from scratch.
        """

        # there really are two models here, the Region2Vec model and the FineTuning model
        # which holds the Region2Vec model. The Region2Vec model ideally is pre-trained, but can be trained
        # from scratch if a tokenizer is passed. The Finetuning model is always trained from scratch.
        self._model: RegionSet2Vec
        self.region2vec: Region2Vec
        self.device = device
        self.trained = False
        self.pooling_method = pooling

        # check for completely blank initialization
        if all(
            [
                model_path is None,
                tokenizer is None,
                region2vec is None,
            ]
        ):
            pass
        # try to load the model from huggingface
        elif model_path is not None:
            self._init_from_huggingface(model_path)
            self.trained = True
        else:
            self._init_model(region2vec, tokenizer, **kwargs)

    def _load_local_region2vec_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load a Region2Vec model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        :param str config_path: Path to the config file.
        """
        _model, tokenizer, config = load_local_region2vec_model(
            model_path, vocab_path, config_path
        )

        self.region2vec = _model
        self.tokenizer = tokenizer
        if POOLING_METHOD_KEY in config:
            self.pooling_method = config[POOLING_METHOD_KEY]

    def _init_region2vec_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the Region2Vec model from a huggingface model. This uses the model path
        to download the necessary files and then "build itself up" from those. This
        includes both the actual model and the tokenizer.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        :param str config_file_name: Name of the config file.
        :param kwargs: Additional keyword arguments to pass to the hf download function.
        """
        model_file_path = hf_hub_download(model_path, model_file_name, **kwargs)
        universe_path = hf_hub_download(model_path, universe_file_name, **kwargs)
        config_path = hf_hub_download(model_path, config_file_name, **kwargs)

        self._load_local_region2vec_model(model_file_path, universe_path, config_path)

    def _init_tokenizer(self, tokenizer: Union[ITTokenizer, str]):
        """
        Initialize the tokenizer.

        :param Union[ITTokenizer, str] tokenizer: Either a tokenizer instance, a huggingface repo, or a path to a tokenizer.
        """
        # they didn't pass anything valid, so we try to build
        # a region2vec model from scratch, please pass a tokenizer :(
        if tokenizer is None:
            raise ValueError(
                "Can't build a Region2Vec model from scratch without a tokenizer. A vobab size is needed!"
            )
        elif isinstance(tokenizer, str):
            # is path to a local vocab?
            if os.path.exists(tokenizer):
                self.tokenizer = ITTokenizer(tokenizer)
            else:
                # assume its a huggingface tokenizer, try to load it
                self.tokenizer = ITTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, ITTokenizer):
            # ideal case - they passed a tokenizer instance
            self.tokenizer = tokenizer
        else:
            raise ValueError(
                "Invalid tokenizer passed. Must be either a path to a tokenizer, a huggingface tokenizer, or a tokenizer instance."
            )

    def _init_model(self, region2vec, tokenizer, **kwargs):
        """
        Initialize a new model from scratch. Ideally, someone is passing in a Region2Vec instance,
        but in theory we can build one from scratch if we have a tokenizer.
        In the "build from scratch" scenario, someone is probably training a new model from scratch on some new data.

        The most important aspect is initializing the inner Region2Vec model. For this, the order of operations
        is as follows:

        1. If a Region2Vec instance is passed, use that.
        2. If a path to a local Region2Vec model is passed, use that.
        3. If a path to a huggingface Region2Vec model is passed, use that.
        4. Nothing was passed? Try from scratch... if a tokenizer is passed, build a Region2Vec model from scratch.
        5. If no tokenizer was passed, raise an error, because we need a vocab size to build a Region2Vec model.

        :param Union[Region2Vec, str] region2vec: Either a Region2Vec instance or a path to a huggingface model.
        :param ITTokenizer tokenizer: The tokenizer to use. This is **required** if building a Region2Vec model from scratch.
        :param kwargs: Additional keyword arguments to pass to the Region2Vec model.
        """
        # first init the tokenizer they passed in
        self._init_tokenizer(tokenizer)

        if isinstance(region2vec, Region2Vec):
            # ideal case - they passed a Region2Vec instance
            self.region2vec = region2vec

        elif isinstance(region2vec, str):
            # is it a local model?
            if os.path.exists(region2vec):
                self._load_local_region2vec_model(
                    os.path.join(region2vec, MODEL_FILE_NAME),
                    os.path.join(region2vec, UNIVERSE_FILE_NAME),
                    os.path.join(region2vec, CONFIG_FILE_NAME),
                )
            else:
                # assume its a huggingface model, try to load it
                self._init_region2vec_from_huggingface(region2vec)

        else:
            self.region2vec = Region2Vec(
                len(self.tokenizer),
                embedding_dim=kwargs.get("embedding_dim") or DEFAULT_EMBEDDING_DIM,
            )

        assert (
            len(self.tokenizer) == self.region2vec.vocab_size
        ), "Tokenizer and Region2Vec vocab size mismatch. Are you sure they are compatible?"

        # build the model, finally using the region2vec model
        self._model = RegionSet2Vec(self.region2vec, self.pooling_method)

    def _load_local_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load the model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """
        r2v, tokenizer, config = load_local_region2vec_model(model_path, vocab_path, config_path)

        self._model = RegionSet2Vec(r2v, config[POOLING_METHOD_KEY])
        self.region2vec = r2v
        self.tokenizer = tokenizer
        self.pooling_method = config[POOLING_METHOD_KEY]

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
    ) -> "Region2VecFineTuner":
        """
        Load the model from a set of files that were exported using the export function.

        :param str path_to_files: Path to the directory containing the files.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        :param str config_file_name: Name of the config file.

        :return: The loaded model.
        """
        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_file_path = os.path.join(path_to_files, universe_file_name)
        config_file_path = os.path.join(path_to_files, config_file_name)

        instance = cls()
        instance._load_local_model(model_file_path, universe_file_path, config_file_path)
        instance.trained = True

        return instance

    def _validate_data(self, data: Union[sc.AnnData, str], label_key: str) -> sc.AnnData:
        """
        Validate the data passed to the training function.

        :param Union[sc.AnnData, str] data: Either a scanpy AnnData object or a path to a h5ad file.
        :param str label_key: The key in the `.obs` attribute that contains the cell type labels.

        :return: The scanpy AnnData object.
        """
        if isinstance(data, str):
            data = sc.read_h5ad(data)
        elif not isinstance(data, sc.AnnData):
            raise ValueError(
                "Data must be either a scanpy AnnData object or a path to a h5ad file."
            )
        assert label_key in data.obs.columns, f"Label key {label_key} not found in data."
        return data

    def train(
        self,
        data: Union[sc.AnnData, str],
        label_key: str = DEFAULT_LABEL_KEY,
        epochs: int = DEFAULT_EPOCHS,
        loss_fn: nn.Module = DEFAULT_FINE_TUNE_LOSS_FN,
        optimizer: torch.optim.Optimizer = DEFAULT_OPTIMIZER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        optimizer_kwargs: dict = {},
        test_train_split: float = DEFAULT_TEST_TRAIN_SPLIT,
        seed: any = 42,
        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: Union[List, str] = None,
        early_stopping: bool = False,
    ) -> FineTuneTrainingResult:
        """
        Train the model. The training loop assumes your data is a scanpy AnnData object,
        with the cell type labels in the `.obs` attribute. The cell type labels are
        expected to be in the column specified by `label_key`.

        :param Union[sc.AnnData, str] data: Either a scanpy AnnData object or a path to a h5ad file.
        :param str label_key: The key in the `.obs` attribute that contains the cell type labels.
        :param int epochs: Number of epochs to train for.
        :param nn.Module loss_fn: Loss function to use.
        :param torch.optim.Optimizer optimizer: Optimizer to use.
        :param int batch_size: Batch size to use.
        :param dict optimizer_kwargs: Additional keyword arguments to pass to the optimizer.
        :param float test_train_split: Fraction of data to use for training (defaults to 0.8).
        :param any seed: Random seed to use.
        :param torch.optim.lr_scheduler._LRScheduler learning_rate_scheduler: Learning rate scheduler to use.
        :param Union[List, str] device: Device to use for training.
        :param bool early_stopping: Whether or not to use early stopping. The training loop will stop if the validation loss
                                    does not improve for 5 epochs. This is an indicator of overfitting.
        :param List[float] r2v_gradient_ramp: List of floats between 0 and 1. If passed, the Region2Vec model will be
        """
        # validate the data
        data = self._validate_data(data, label_key)

        pos_pairs, neg_pairs, pos_labels, neg_labels = generate_fine_tuning_dataset(
            data, self.tokenizer, seed=seed
        )

        # combine the positive and negative pairs
        pairs = pos_pairs + neg_pairs
        labels = pos_labels + neg_labels

        # get the pad token id
        pad_token_id = self.tokenizer.padding_token_id()

        train_pairs, test_pairs, Y_train, Y_test = train_test_split(
            pairs,
            labels,
            train_size=test_train_split,
            random_state=seed,
        )

        # create the datasets
        train_dataloader = DataLoader(
            FineTuningDataset(train_pairs, Y_train),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_finetuning_batch(x, pad_token_id),
        )
        test_dataloader = DataLoader(
            FineTuningDataset(test_pairs, Y_test),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_finetuning_batch(x, pad_token_id),
        )

        # move the model to the device
        if isinstance(device, list):
            self._model = nn.DataParallel(self._model, device_ids=device)
            self._model.to(device[0])
        else:
            self._model.to(device)

        # tensor device
        if isinstance(device, list):
            tensor_device = device[0]
        else:
            tensor_device = "cuda" if torch.cuda.is_available() else "cpu"

        # loss_fn = loss_fn()
        loss_fn = DEFAULT_FINE_TUNE_LOSS_FN()

        # create the optimizer
        optimizer = optimizer(self._model.parameters(), **optimizer_kwargs)

        # init scheduler if passed
        if learning_rate_scheduler is not None:
            learning_rate_scheduler = learning_rate_scheduler(optimizer)

        all_loss = []
        epoch_loss = []
        validation_loss = []
        consecutive_no_improvement = 0

        with Progress() as progress_bar:
            epoch_tid = progress_bar.add_task("Epochs", total=epochs)
            batches_tid = progress_bar.add_task("Batches", total=len(train_dataloader))
            for epoch in range(epochs):
                _LOGGER.info(f"Epoch {epoch + 1}/{epochs}")
                # set the model to train mode
                self._model.train()
                this_epoch_loss = []
                for _, batch in enumerate(train_dataloader):
                    # zero the gradients
                    optimizer.zero_grad()

                    # move the batch to the device
                    pair, target = batch
                    t1, t2 = pair

                    t1 = t1.to(tensor_device)
                    t2 = t2.to(tensor_device)
                    target = target.to(tensor_device)

                    # forward pass for the batch
                    u = self._model(t1)
                    v = self._model(t2)

                    # compute the loss
                    loss = loss_fn(u, v, target.float())
                    loss.backward()
                    all_loss.append(loss.item())
                    this_epoch_loss.append(loss.item())

                    # clip gradients
                    optimizer.step()

                    # update the progress bar
                    progress_bar.update(batches_tid, advance=1)

                # compute the loss for the epoch
                epoch_loss.append(sum(this_epoch_loss) / len(this_epoch_loss))

                # compute the validation loss
                with torch.no_grad():
                    self._model.eval()
                    val_loss = []
                    for i, batch in enumerate(test_dataloader):
                        # move the batch to the device
                        pair, target = batch
                        t1, t2 = pair

                        t1 = t1.to(tensor_device)
                        t2 = t2.to(tensor_device)
                        target = target.to(tensor_device)

                        # forward pass for the batch
                        u = self._model(t1)
                        v = self._model(t2)

                        # compute the loss
                        loss = loss_fn(u, v, target.float())
                        val_loss.append(loss.item())

                # compute the loss for the epoch
                val_loss = sum(val_loss) / len(val_loss)
                validation_loss.append(val_loss)

                # update the progress bar
                progress_bar.update(epoch_tid, advance=1)

                # check for early stopping
                if early_stopping:
                    if len(validation_loss) > 1:
                        if validation_loss[-1] > validation_loss[-2]:
                            consecutive_no_improvement += 1
                        else:
                            consecutive_no_improvement = 0

                        if consecutive_no_improvement >= 5:
                            break

                # update the learning rate scheduler
                if learning_rate_scheduler is not None:
                    learning_rate_scheduler.step()

                self.trained = True

                # log out the losses
                _LOGGER.info(f"Epoch loss: {epoch_loss[-1]}")
                _LOGGER.info(f"Validation loss: {val_loss}")

        return FineTuneTrainingResult(
            validation_loss=validation_loss,
            epoch_loss=epoch_loss,
            all_loss=all_loss,
        )

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
        # export the model
        kwargs = {POOLING_METHOD_KEY: self.pooling_method}
        export_region2vec_model(
            self.region2vec,
            self.tokenizer,
            path,
            checkpoint_file,
            universe_file,
            config_file,
            **kwargs,
        )
