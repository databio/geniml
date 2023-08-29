from typing import Union

import matplotlib as plt
import numpy as np
import tensorflow as tf
from gitk.search.backends import HNSWBackend, QdrantBackend

from .const import *
from .utils import *


class Embed2EmbedNN(tf.keras.models.Sequential):
    """
    A feedforward neural network that maps embedding vectors or region sets metadata
    to the embedding vectors of region sets
    """

    def __init__(self):
        # initiate a Sequential model from keras
        super().__init__()
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

    def load_local_pretrained(self, model_path: str):
        """
        load pretrained model from local file

        :param model_path: path where the model is saved
        :return:
        """

        # empty the layers if current model is not empty
        while len(self.layers) > 0:
            self.layers.pop()

        # https://stackoverflow.com/questions/63068639/valueerror-unknown-layer-functional
        local_model = tf.keras.models.load_model(
            model_path, custom_objects={"Embed2EmbedNN": tf.keras.models.Sequential()}
        )
        # add layers from pretrained model
        for layer in local_model.layers:
            self.add(layer)

    def train(
        self,
        training_X: np.ndarray,
        training_Y: np.ndarray,
        validating_X: np.ndarray,
        validating_Y: np.ndarray,
        opt_name: str = DEFAULT_OPTIMIZER_NAME,
        loss_func: str = DEFAULT_LOSS_NAME,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ):
        """
        Fit the feedforward neural network

        :param training_X: embedding vectors of metadata, np.ndarray with shape of (n, <dim>)
        :param training_Y: embedding vectors of region set, np.ndarray with shape of (n, <dim>)
        :param validating_X: validating vectors of metadata
        :param validating_Y: validating vectors of region set
        :param opt_name: name of optimizer
        :param loss_func: name of loss function
        :param num_epochs: number of training epoches
        :param batch_size: size of batch for training
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
        # early stoppage to prevent over-fitting
        early_stoppage = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(num_epochs * 0.25)
        )

        # record training history
        train_hist = self.fit(
            training_X,
            training_Y,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(validating_X, validating_Y),
            callbacks=[early_stoppage],
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

    def plot_training_hist(self):
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
        plt.show()


class TextToBedNNSearchInterface(object):
    """
    search backend interface
    """

    def __init__(
        self,
        nl2vec_model: Union[SentenceTransformer, None],
        vec2vec_model: Union[Embed2EmbedNN, None],
        search_backend: Union[QdrantBackend, HNSWBackend, None],
    ):
        """
        initiate the search interface

        :param nl2vec_model: model that embed natural language to vectors
        :param vec2vec_model: model that map natural language embedding vectors to region set embedding vectors
        :param search_backend: search backend that can store vectors and perform KNN search
        """

        if isinstance(nl2vec_model, type(None)):
            # default SentenceTransformer model
            self.set_sentence_transformer()
        else:
            self.nl2vec = nl2vec_model

        if isinstance(vec2vec_model, type(None)):
            # init an empty Embed2EmbedNN model if input is None
            self.vec2vec = Embed2EmbedNN()
        else:
            self.vec2vec = vec2vec_model

        if isinstance(search_backend, type(None)):
            # init a default HNSWBackend if input is None
            self.search_backend = HNSWBackend()
        else:
            self.search_backend = search_backend

    def set_sentence_transformer(self, st_repo: str = DEFAULT_HF_ST_MODEL):
        """
        With a given huggingface repo, set the nl2vec model as a sentence transformer

        :param st_repo: the hugging face repository of sentence transformer
        see https://huggingface.co/sentence-transformers
        :return:
        """
        self.nl2vec = SentenceTransformer(st_repo)

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


#
# # Example of how to use the TextToBedNN Search Interface
#
# betum = BEDEmbedTUM(RSC, universe, tokenizer)
# embeddings = betum.compute_embeddings()
# T2BNNSI = TextToBedNNSearchInterface(betum, embeddings)  # ???
