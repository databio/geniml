import numpy as np
from typing import List, Union, Tuple

from .const import *

from .utils import *
from ..models import Model
from ..search import QdrantBackend, HNSWBackend
import tensorflow as tf
import matplotlib as plt


class Embed2EmbedNN(tf.keras.models.Sequential):
    def __init__(self):
        super()
        # super.__init__()
        self.most_recent_train = None

    def add_layers(self,
                   input_dim: int,
                   output_dim: int,
                   units: int,
                   layers: int):

        self.add(tf.keras.layers.Dense(units, input_shape=(input_dim,), activation='relu'))

        # extra dense layers
        for i in range(layers):
            self.add(tf.keras.layers.Dense(units, input_shape=(units,), activation='relu'))

        # output
        self.add(tf.keras.layers.Dense(output_dim))

    def load_local_pretrained(self,
                              model_path: str):
        while len(self.layers) > 0:
            self.layers.pop()

        local_model = tf.keras.models.load_model(model_path)
        for layer in local_model.layers:
            self.add(layer)

    def train(self,
              training_X: np.ndarray,
              training_Y: np.ndarray,
              validating_X: np.ndarray,
              validating_Y: np.ndarray,
              opt_name: str = DEFAULT_OPTIMIZER,
              loss_func: str = DEFAULT_LOSS,
              epochs: int = DEFAULT_EPOCHES,
              batch_size: int = DEFAULT_BATCH_SIZE,
              **kwargs
              ):

        if len(self.layers) == 0:
            # dimensions of input and output
            input_dim = training_X.shape[1]
            output_dim = training_Y.shape[1]

            self.add_layers(
                input_dim=input_dim,
                output_dim=output_dim,
                units=kwargs.get("units") or DEFAULT_UNITS,
                layers=kwargs.get("layers") or DEFAULT_LAYERS
            )

        self.compile(optimizer=opt_name, loss=loss_func)
        # early stoppage to prevent over-fitting
        early_stoppage = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(epochs * 0.25))

        # record training history
        train_hist = self.fit(training_X, training_Y, epochs=epochs, batch_size=batch_size,
                              validation_data=(validating_X, validating_Y),
                              callbacks=[early_stoppage])
        self.most_recent_train = train_hist

    def embedding_to_embedding(self, input_vec: np.ndarray):
        if len(np.ndarray.shape) == 1:
            vec_dim = input_vec.shape[0]
            input_vec = input_vec.reshape((1, vec_dim))

        output_vec = self.predict(input_vec)
        if output_vec.shape[0] == 1:
            output_vec_dim = output_vec.shape[1]
            output_vec = output_vec.reshape(output_vec_dim, )
        return output_vec

    def plotting_training_hist(self):

        epoch_range = range(1, len(self.most_recent_train.history['loss']) + 1)
        train_loss = self.most_recent_train.history['loss']
        valid_loss = self.most_recent_train.history['val_loss']
        plt.plot(epoch_range, train_loss, 'r', label='Training loss')
        plt.plot(epoch_range, valid_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()


class TextToBedNNSearchInterface(object):
    def __init__(self,
                 nl2vec_model: Union[SentenceTransformer, None],
                 vec2vec_model: Union[Embed2EmbedNN, None],
                 region_set_backend: Union[QdrantBackend, HNSWBackend, None]):

        if isinstance(nl2vec_model, None):
            self.set_sentence_transformer()
        else:
            self.nl2vec = nl2vec_model

        if isinstance(vec2vec_model, None):
            self.vec2vec = Embed2EmbedNN()
        else:
            self.vec2vec = vec2vec_model

        if isinstance(region_set_backend, None):
            self.region_set_backend = QdrantBackend()
        else:
            self.region_set_backend = region_set_backend

    def set_sentence_transformer(self,
                                 st_repo: str = DEFAULT_HF_ST_MODEL):
        self.nl2vec = SentenceTransformer(st_repo)

    def nl_vec_search(self,
                      query: Union[str, np.ndarray],
                 k: int = 10) -> List:
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
        #
        return self.region_set_backend.search(search_vector, k)

    # def evaluate(self,
    #              ):

    # Todo
    # function that load a folder of bed files into the qdrant client db
    # then fit the embed2embed nn with metadata
#
# # Example of how to use the TextToBedNN Search Interface
#
# betum = BEDEmbedTUM(RSC, universe, tokenizer)
# embeddings = betum.compute_embeddings()
# T2BNNSI = TextToBedNNSearchInterface(betum, embeddings)  # ???
