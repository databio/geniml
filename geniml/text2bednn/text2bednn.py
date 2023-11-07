import math
import os

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import hf_hub_download
from torch.nn import (CosineEmbeddingLoss, CosineSimilarity, Linear, MSELoss,
                      ReLU, Sequential)
from yaml import safe_dump, safe_load

from ..search.backends import HNSWBackend, QdrantBackend
from .const import *
from .utils import *

_LOGGER = logging.getLogger(MODULE_NAME)
# _TORCH_LOGGER = logging.getLogger("torch")


class Vec2VecFNNtorch:
    def __init__(self):
        """
        Initializate Vec2VecFNNtorch.
        """
        # initialize the feedforward neural network model, which is a torch.nn.Sequential
        self.model = Sequential()
        # whether the model is trained
        self.trained = False
        # optimizer
        self.optimizer = None
        # loss function
        self.loss_fn = None
        # model configure
        self.config = {}
        #
        self.most_recent_train = {}

    def reinit_model(
        self,
        input_dim: int,
        output_dim: int,
        num_units: Union[int, List[int]],
        num_extra_hidden_layers: int,
    ):
        """
        Re-initiate self.model(a torch.nn.Sequential model) with list of layers

        :param input_dim: dimension of input layer
        :param output_dim: dimension of output layer
        :param num_units: number of units in each hidden layer, if it is an integer, each hidden layer has same
        number of units
        :param num_extra_hidden_layers: number of extra hidden layers
        :return:
        """

        # convert the integer value of num_units to a list
        if not isinstance(num_units, list):
            num_units = [num_units] * (1 + num_extra_hidden_layers)

        # update model config
        self.config["input_dim"] = input_dim
        self.config["output_dim"] = output_dim
        self.config["num_units"] = num_units
        self.config["num_extra_hidden_layers"] = num_extra_hidden_layers

        # check if number of layers match length of num_units
        if len(num_units) != 1 + num_extra_hidden_layers:
            _LOGGER.error("ValueError: list of units number does not match number of layers")

        # input and first hiden layer
        current_layer_units_num = num_units[0]
        layers_list = [Linear(in_features=input_dim, out_features=current_layer_units_num), ReLU()]
        previous_layer_units_num = current_layer_units_num

        # extra hidden layer
        for i in range(num_extra_hidden_layers):
            current_layer_units_num = num_units[i + 1]
            layers_list.append(
                Linear(in_features=previous_layer_units_num, out_features=current_layer_units_num)
            )
            layers_list.append(ReLU())
            previous_layer_units_num = current_layer_units_num

        # output layer
        layers_list.append(Linear(in_features=previous_layer_units_num, out_features=output_dim))

        # reinitiate self.model
        self.model = Sequential(*layers_list)

    def load_from_disk(self, model_path: str, config_path: str):
        """
        Load model from local files

        :param model_path: path of saved model file (usually in format of .pt)
        :param config_path: path of saved config file (in format of yaml)
        """
        # get the model config (layer structure)
        with open(config_path, "r") as f:
            config = safe_load(f)

        # reinitiate the self.model
        self.reinit_model(
            config["input_dim"],
            config["output_dim"],
            config["num_units"],
            config["num_extra_hidden_layers"],
        )
        # load the Sequential model from saved files
        self.model.load_state_dict(torch.load(model_path))

    def export(
        self,
        path: str,
        checkpoint_file: str,
        config_file: str = CONFIG_FILE_NAME,
        must_trained: bool = DEFAULT_MUST_TRAINED,
    ):
        """
        Save model weights and config

        :param path: path to export the model to
        :param checkpoint_file: name of model checkpoint file
        :param config_file: name of model config file
        :param must_trained: whether the model needs training to be exported
        """
        # whether the model must be finished training to export
        if must_trained and not self.trained:
            raise RuntimeError("Cannot export an untrained model.")

        if not os.path.exists(path):
            os.makedirs(path)

        # export the model weights
        torch.save(self.model.state_dict(), os.path.join(path, checkpoint_file))

        # export the model config
        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(self.config, f)

    def embedding_to_embedding(self, input_vecs: np.ndarray) -> np.ndarray:
        """
        predict the region set embedding from embedding of natural language strings

        :param input_vecs: input embedding vectors
        :return:
        """
        # pytorch tensor's default dtype is float 32
        if not isinstance(input_vecs.dtype, type(np.dtype("float32"))):
            input_vecs = input_vecs.astype(np.float32)
        return self.model(torch.from_numpy(input_vecs)).detach().numpy()

    def compile(self, optimizer: str, loss: str, learning_rate: float):
        """
        Configures the model for training.

        :param optimizer: the name of optimizer
        :param loss: the name of loss function
        :param learning_rate: the learning rate of model backpropagation
        """

        # set optimizer
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        elif optimizer == "SGD":
            self.optimizer

        else:
            raise ValueError("Please give a valid name of optimizer")
            # _LOGGER.error("ValueError: Please give a valid name of optimizer")

        # set loss function
        if loss == "cosine_embedding_loss":
            self.loss_fn = CosineEmbeddingLoss()
        elif loss == "cosine_similarity":
            self.loss_fn = CosineSimilarity()
        elif loss == "mean_squared_error":
            self.loss_fn = MSELoss()
        else:
            raise ValueError("Please give a valid name of loss function")
            # _LOGGER.error("ValueError: Please give a valid name of loss function")

        self.config["optimizer"] = optimizer
        self.config["loss"] = loss

    def train(
        self,
        training_X: np.ndarray,
        training_Y: np.ndarray,
        validating_data: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        save_best: bool = False,
        folder_path: Union[str, None] = None,
        model_file_name: Union[str, None] = None,
        early_stop: bool = True,
        patience: float = DEFAULT_PATIENCE,
        opt_name: str = DEFAULT_OPTIMIZER_NAME,
        loss_func: str = DEFAULT_LOSS_NAME,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        **kwargs,
    ):
        """
        Based on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

        """
        if len(self.model) == 0:
            # dimensions of input and output
            input_dim = training_X.shape[1]
            output_dim = training_Y.shape[1]

            self.reinit_model(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=kwargs.get("num_units") or DEFAULT_NUM_UNITS,
                num_extra_hidden_layers=kwargs.get("num_extra_hidden_layers")
                or DEFAULT_NUM_EXTRA_HIDDEN_LAYERS,
            )

        if validating_data is not None:
            validating_X, validating_Y = validating_data
            validating_data = arrays_to_torch_dataloader(
                validating_X, validating_Y, batch_size=batch_size, shuffle=False
            )
            self.most_recent_train["val_loss"] = []
        elif save_best or early_stop:
            _LOGGER.error("ValueError: Validating data is not provided")
        if save_best and folder_path is None:
            _LOGGER.error(
                "ValueError: Path to folder where the best performance model will be saved is required"
            )

        self.compile(optimizer=opt_name, loss=loss_func, learning_rate=learning_rate)
        training_data = arrays_to_torch_dataloader(training_X, training_Y, batch_size)
        best_val_loss = 1_000_000.0

        patience_count = 0
        self.most_recent_train["loss"] = []

        for epoch in range(num_epochs):
            self.model.train(True)
            avg_loss = self.train_one_epoch(training_data)
            self.most_recent_train["loss"].append(avg_loss)
            self.model.eval()

            if validating_data is not None:
                running_val_loss = 0.0
                with torch.no_grad():
                    for i, (val_x, val_y) in enumerate(validating_data):
                        val_output = self.model(val_x)
                        val_loss = self.calc_loss(val_output, val_y)
                        running_val_loss += val_loss

                avg_val_loss = running_val_loss / (i + 1)
                self.most_recent_train["val_loss"].append(avg_val_loss)
                _LOGGER.info(f"EPOCH {epoch + 1}: loss: -{avg_loss} - val_loss: -{avg_val_loss}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if save_best:
                        self.export(
                            folder_path,
                            model_file_name
                            or TORCH_MODEL_FILE_NAME_PATTERN.format(
                                callback="best", checkpoint=str(epoch)
                            ),
                            must_trained=False,
                        )

                if early_stop:
                    if avg_val_loss > avg_loss:
                        patience_count += 1
                    if patience_count > int(math.ceil(patience * num_epochs)):
                        self.export(
                            folder_path,
                            model_file_name
                            or TORCH_MODEL_FILE_NAME_PATTERN.format(
                                callback="early_stop", checkpoint=str(epoch)
                            ),
                            must_trained=False,
                        )
                        break

            else:
                _LOGGER.info(f"EPOCH {epoch + 1}: loss: -{avg_loss}")
        self.trained = True

    def train_one_epoch(self, training_data):
        epoch_loss = 0.0

        for i, (x, y) in enumerate(training_data):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            batch_loss = self.calc_loss(outputs, y)
            batch_loss.backward()
            # adjust learning weights
            self.optimizer.step()

            # geather loss and report
            epoch_loss += batch_loss.item()
        return epoch_loss / (i + 1)

    def calc_loss(self, outputs: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.float:
        """
        Calculating loss when different loss funcion is given

        :param outputs: the output of model
        :param y: the correct label
        """

        if not self.config["loss"]:
            raise ValueError("Please compile the model first")
            # _LOGGER.error("ValueError: Please compile the model first")

        # when all targets are 1
        # loss = 1 - cos(output, y)
        # https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        elif self.config["loss"] == "cosine_embedding_loss":
            target = kwargs.get("target") or torch.tensor(1.0).repeat(len(y))
            return self.loss_fn(outputs, y, target)
        else:
            return self.loss_fn(outputs, y)

    def plot_training_hist(
        self,
        save_path: Union[str, None] = None,
        plot_file_name: Union[str, None] = DEFAULT_PLOT_FILE_NAME,
        title: Union[str, None] = DEFAULT_PLOT_TITLE,
    ):
        """
        Plot the training & validating loss of the most recent training
        :return:
        """

        epoch_range = range(1, len(self.most_recent_train["loss"]) + 1)
        train_loss = self.most_recent_train["loss"]
        plt.plot(epoch_range, train_loss, "r", label="Training loss")
        if self.most_recent_train["val_loss"]:
            valid_loss = self.most_recent_train["val_loss"]
            plt.plot(epoch_range, valid_loss, "b", label="Validation loss")
        plt.title(title)
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, plot_file_name))
        else:
            plt.show()


class Vec2VecFNN(tf.keras.models.Sequential):
    """
    A feedforward neural network that maps embedding vectors or region sets metadata
    to the embedding vectors of region sets
    """

    def __init__(self, model_path: str = None):
        """
        Initializate Vec2VecFNN.

        :param str model_path: Path to the pre-trained model on huggingface.
        """
        super().__init__()
        # initiate a Sequential model from keras
        if model_path is not None:
            if os.path.exists(model_path):
                # load from disk
                self.load_from_disk(model_path)
            else:
                # load from hugging face
                self.load_from_huggingface(model_path)

        # most recent training history
        self.most_recent_train = None

    def add_layers(
        self, input_dim: int, output_dim: int, num_units: int, num_extra_hidden_layers: int
    ):
        """
        Add layers to an empty Sequential model

        :param input_dim: dimension of input vector
        :param output_dim: dimension of output vector
        :param num_units: number of units in dense layer
        :param num_extra_hidden_layers: number of extra hidden layers
        :return:
        """
        # the dense layer that accept the input
        self.add(tf.keras.layers.Dense(num_units, input_shape=(input_dim,), activation="relu"))

        # extra dense layers
        for i in range(num_extra_hidden_layers):
            self.add(tf.keras.layers.Dense(num_units, input_shape=(num_units,), activation="relu"))

        # output
        self.add(tf.keras.layers.Dense(output_dim))

    def load_from_disk(self, model_path: str):
        """
        load pretrained model from disk

        :param model_path: path where the model is saved
        :return:
        """

        # empty the layers if current model is not empty
        while len(self.layers) > 0:
            self.layers.pop()

        # https://stackoverflow.com/questions/63068639/valueerror-unknown-layer-functional
        local_model = tf.keras.models.load_model(
            model_path, custom_objects={"Vec2VecFNN": tf.keras.models.Sequential()}
        )
        # add layers from pretrained model
        for layer in local_model.layers:
            self.add(layer)

    def load_from_huggingface(
        self,
        model_repo: str,
        model_file_name: str = DEFAULT_VEC2VEC_MODEL_FILE_NAME,
        **kwargs,
    ):
        """
        Download pretrained model from huggingface

        :param str model: The name of the model to download (this is the same as the repo name).
        :param str model_file_name: The name of the model file - this should almost never be changed.
        """
        model_path = hf_hub_download(model_repo, model_file_name, **kwargs)
        self.load_from_disk(model_path)

    def train(
        self,
        training_X: np.ndarray,
        training_Y: np.ndarray,
        validating_data: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        opt_name: str = DEFAULT_OPTIMIZER_NAME,
        loss_func: str = DEFAULT_LOSS_NAME,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        patience: float = DEFAULT_PATIENCE,
        **kwargs,
    ):
        """
        Fit the feedforward neural network

        :param training_X: embedding vectors of metadata, np.ndarray with shape of (n, <dim>)
        :param training_Y: embedding vectors of region set, np.ndarray with shape of (n, <dim>)
        :param validating_data: validating data, which contains validating X and validating Y
        :param opt_name: name of optimizer
        :param loss_func: name of loss function
        :param num_epochs: number of training epoches
        :param batch_size: size of batch for training
        :param learning_rate: learning rate of optimizer
        :param patience: the percentage of epoches in which if no validation loss improvement,
        the training will be stopped
        :param kwargs: see units and layers in add_layers()
        :return:
        """

        # if current model is empty, add layers
        if len(self.layers) == 0:
            # dimensions of input and output
            input_dim = training_X.shape[1]
            output_dim = training_Y.shape[1]

            self.add_layers(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=kwargs.get("num_units") or DEFAULT_NUM_UNITS,
                num_extra_hidden_layers=kwargs.get("num_extra_hidden_layers")
                or DEFAULT_NUM_EXTRA_HIDDEN_LAYERS,
            )

        # compile the model
        self.compile(optimizer=opt_name, loss=loss_func)
        # set the learning rate
        tf.keras.backend.set_value(self.optimizer.learning_rate, learning_rate)
        # if there is validating data, set early stoppage to prevent over-fitting
        callbacks = None
        if validating_data:
            early_stoppage = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(num_epochs * patience)
            )
            callbacks = [early_stoppage]

        # record training history
        train_hist = self.fit(
            training_X,
            training_Y,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=validating_data,
            callbacks=callbacks,
        )
        self.most_recent_train = train_hist

    def embedding_to_embedding(self, input_vec: np.ndarray):
        """
        predict the region set embedding from embedding of natural language strings
        :param input_vec:
        :return:
        """
        # the network only accept input vectors in shape of (n, <input dim>)
        # so if the input np.ndarray has shape (<input dim>,)
        # it needs reshaping
        if len(input_vec.shape) == 1:
            vec_dim = input_vec.shape[0]
            input_vec = input_vec.reshape((1, vec_dim))

        # reshape output vector if the input np.ndarray has shape (<input dim>,)
        output_vec = self.predict(input_vec)
        if output_vec.shape[0] == 1:
            output_vec_dim = output_vec.shape[1]
            output_vec = output_vec.reshape(
                output_vec_dim,
            )
        return output_vec

    def plot_training_hist(self, save_path: Union[str, None] = None):
        """
        plot the training & validating loss of the most recent training
        :return:
        """

        epoch_range = range(1, len(self.most_recent_train.history["loss"]) + 1)
        train_loss = self.most_recent_train.history["loss"]
        valid_loss = self.most_recent_train.history["val_loss"]
        plt.plot(epoch_range, train_loss, "r", label="Training loss")
        plt.plot(epoch_range, valid_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, "train_hist.png"))
        else:
            plt.show()


class Text2BEDSearchInterface(object):
    """
    search backend interface
    """

    def __init__(
        self,
        nl2vec_model: Union[SentenceTransformer, None],
        vec2vec_model: Union[Vec2VecFNN, str, None],
        search_backend: Union[QdrantBackend, HNSWBackend, None],
    ):
        """
        initiate the search interface

        :param nl2vec_model: model that embed natural language to vectors
        :param vec2vec_model: model that map natural language embedding vectors to region set embedding vectors
        :param search_backend: search backend that can store vectors and perform KNN search
        """
        # load the natural language encoder model
        if isinstance(nl2vec_model, type(None)):
            # default SentenceTransformer model
            self.set_sentence_transformer()
        else:
            self.nl2vec = nl2vec_model

        # load the vec2vec model
        if isinstance(vec2vec_model, Vec2VecFNN):
            self.vec2vec = vec2vec_model
        elif isinstance(vec2vec_model, str):
            self.set_vec2vec(vec2vec_model)

        # init search backend
        if isinstance(search_backend, type(None)):
            # init a default HNSWBackend if input is None
            self.search_backend = HNSWBackend()
        else:
            self.search_backend = search_backend

    def set_vec2vec(self, model_name: str):
        """
        With a given model_path or huggingface repo, set the vec2vec model

        :param model_name: the path where the model file is saved, or the hugging face repo
        """
        self.vec2vec = Vec2VecFNN(model_name)

    def set_sentence_transformer(self, hf_repo: str = DEFAULT_HF_ST_MODEL):
        """
        With a given huggingface repo, set the nl2vec model as a sentence transformer

        :param hf_repo: the hugging face repository of sentence transformer
        see https://huggingface.co/sentence-transformers
        :return:
        """
        _LOGGER.info(f"Setting sentence transformer model {hf_repo}")
        self.nl2vec = SentenceTransformer(hf_repo)

    def nl_vec_search(
        self, query: Union[str, np.ndarray], k: int = 10
    ) -> Tuple[Union[List[int], List[List[int]]], Union[List[float], List[List[float]]]]:
        """
        Given an input natural language, suggest region sets

        :param query: searching input string
        :param k: number of results (nearst neighbor in vectors)
        :return: a list of Qdrant Client search results
        """

        # first, get the embedding of the query string
        if isinstance(query, str):
            query = self.nl2vec.encode(query)
        search_vector = self.vec2vec.embedding_to_embedding(query)
        # perform the KNN search among vectors stored in backend
        return self.search_backend.search(search_vector, k)

    def __repr__(self):
        return f"Text2BEDSearchInterface(nl2vec_model={self.nl2vec}, vec2vec_model={self.vec2vec}, search_backend={self.search_backend})"
