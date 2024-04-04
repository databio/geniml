import logging
import math
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.nn import CosineEmbeddingLoss, CosineSimilarity, Linear, MSELoss, ReLU, Sequential
from yaml import safe_dump, safe_load

from .const import *
from .utils import arrays_to_torch_dataloader, dtype_check

_LOGGER = logging.getLogger(MODULE_NAME)


class Vec2Vec(Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_units: Union[int, List[int]],
        num_extra_hidden_layers: int,
    ):
        if not isinstance(num_units, list):
            num_units = [num_units] * (1 + num_extra_hidden_layers)
        # check if number of layers match length of num_units
        if len(num_units) != 1 + num_extra_hidden_layers:
            raise ValueError("list of units number does not match number of layers")
        # input and first hidden layer
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

        super().__init__(*layers_list)


class Vec2VecFNN:
    def __init__(self, model_path: Union[str, None] = None):
        """
        Initialize Vec2VecFNNtorch.

        :param model_path: path to the pretrained model on huggingface.
        """
        # initialize the feedforward neural network model, which is a torch.nn.Sequential
        # self.model =
        self.model = None
        # whether the model is trained
        self.trained = False
        # optimizer
        self.optimizer = None
        # loss function
        self.loss_fn = None
        # model configure
        self.config = {}
        # training history
        self.most_recent_train = {}

        if model_path is not None:
            # load from hugging face
            self._init_from_huggingface(model_path)
            self.trained = True

    def _init_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = DEFAULT_HUGGINGFACE_MODEL_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the model from a huggingface model. This uses the model path
        to download the necessary files and then "build itself up" from those.

        :param model_path: path to the pre-trained model on huggingface.
        :param model_file_name: name of the model file.
        :param config_file_name: name of the config file
        """
        model_file_path = hf_hub_download(model_path, model_file_name, **kwargs)
        config_path = hf_hub_download(model_path, config_file_name, **kwargs)

        self.load_from_disk(model_file_path, config_path)

    def load_from_disk(self, model_path: str, config_path: str):
        """
        Load model from local files

        :param model_path: path of saved model file (usually in format of .pt)
        :param config_path: path of saved config file (in format of yaml)
        """
        # get the model config (layer structure)
        with open(config_path, "r") as f:
            config = safe_load(f)

        self.config = config
        # reinitiate the self.model
        self.model = Vec2Vec(
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
        Predict the region set embedding from embedding of natural language strings

        :param input_vecs: input embedding vectors
        :return: the output of the neural network model
        """
        # pytorch tensor's default dtype is float 32
        return self.model(torch.from_numpy(dtype_check(input_vecs))).detach().numpy()

    def compile(
        self,
        optimizer: str,
        loss: str,
        learning_rate: float,
        margin: Union[float, None] = DEFAULT_MARGIN,
    ):
        """
        Configure the model for training. This includes setting the optimizer and loss function.

        :param optimizer: the name of optimizer
        :param loss: the name of loss function
        :param learning_rate: the learning rate of model backpropagation
        :param margin: should be a number from −1−1 to 1, 0 to 0.5 is suggested, only for CosineEmbeddingLoss
        """

        # set optimizer
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)

        else:
            raise ValueError("Please give a valid name of optimizer")

        # set loss function
        if loss == "cosine_embedding_loss":
            self.loss_fn = CosineEmbeddingLoss(margin=margin)
        elif loss == "cosine_similarity":
            self.loss_fn = CosineSimilarity()
        elif loss == "mean_squared_error":
            self.loss_fn = MSELoss()
        else:
            raise ValueError("Please give a valid name of loss function")

        # add information to model config
        self.config["optimizer"] = optimizer
        self.config["loss"] = loss

    def train(
        self,
        training_X: np.ndarray,
        training_Y: np.ndarray,
        validating_data: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        save_best: bool = False,
        folder_path: Union[str, None] = None,
        best_model_file_name: Union[str, None] = None,
        early_stop: bool = False,
        patience: float = DEFAULT_PATIENCE,
        opt_name: str = DEFAULT_OPTIMIZER_NAME,
        loss_func: str = DEFAULT_LOSS_NAME,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        training_target: Union[np.ndarray, None] = None,
        validating_target: Union[np.ndarray, None] = None,
        **kwargs,
    ):
        """
        Based on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        Fit the feedforward neural network

        :param training_X: embedding vectors of metadata, np.ndarray with shape of (n, <dim>)
        :param training_Y: embedding vectors of region set, np.ndarray with shape of (n, <dim>)
        :param validating_data: validating data, which contains validating X and validating Y
        :param save_best: whether the best performance model is saved after each epoch (based on validation loss)
        :param folder_path: the path to the folder to save the model and config
        :param best_model_file_name: the name of the file of saved best model
        :param early_stop: whether the training should be stopped early to prevent overfitting
        :param patience: the percentage of epoches to stop training if no validation loss improvement
        :param opt_name: name of optimizer
        :param loss_func: name of loss function
        :param num_epochs: number of training epoches
        :param batch_size: size of batch for training
        :param learning_rate: learning rate of optimizer
        :param training_target:
        :param validating_target:
        :param kwargs: see units and layers in reinit_model()
        """
        # if current model is empty, add layers
        if self.model is None:
            # dimensions of input and output
            input_dim = training_X.shape[1]
            output_dim = training_Y.shape[1]

            self.config["input_dim"] = input_dim
            self.config["output_dim"] = output_dim
            self.config["num_units"] = kwargs.get("num_units") or DEFAULT_NUM_UNITS
            self.config["num_extra_hidden_layers"] = (
                kwargs.get("num_extra_hidden_layers") or DEFAULT_NUM_EXTRA_HIDDEN_LAYERS
            )
            self.model = Vec2Vec(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=self.config["num_units"],
                num_extra_hidden_layers=self.config["num_extra_hidden_layers"],
            )

        if training_target is None:
            training_target = np.repeat(1, training_X.shape[0])
        # raise the error if validating data is needed but not provided
        if validating_data is not None:
            validating_X, validating_Y = validating_data
            if validating_target is None:
                validating_target = np.repeat(1, validating_X.shape[0])
            validating_data = arrays_to_torch_dataloader(
                validating_X, validating_Y, validating_target, batch_size=batch_size, shuffle=False
            )

            self.most_recent_train["val_loss"] = []
        elif save_best or early_stop:
            raise ValueError("Validating data is not provided")
        if save_best and folder_path is None:
            raise ValueError(
                "ValueError: Path to folder where the best performance model will be saved is required"
            )

        # compile the model
        self.compile(optimizer=opt_name, loss=loss_func, learning_rate=learning_rate)

        # convert training data from np.ndarray to DataLoader
        training_data = arrays_to_torch_dataloader(
            training_X, training_Y, training_target, batch_size
        )

        best_val_loss = 1_000_000.0
        patience_count = 0
        self.most_recent_train["loss"] = []

        for epoch in range(num_epochs):
            # gradient tracking is on
            self.model.train(True)
            avg_loss = self.train_one_epoch(training_data)
            self.most_recent_train["loss"].append(avg_loss)
            # set model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            if validating_data is not None:
                running_val_loss = 0.0
                # disable gradient computation
                with torch.no_grad():
                    for i, (val_x, val_y, val_target) in enumerate(validating_data):
                        val_output = self.model(val_x)
                        val_loss = self.calc_loss(val_output, val_y, val_target)
                        running_val_loss += val_loss

                avg_val_loss = running_val_loss / (i + 1)
                self.most_recent_train["val_loss"].append(avg_val_loss)
                # logging training and validating loss
                _LOGGER.info(f"EPOCH {epoch + 1}: loss: -{avg_loss} - val_loss: -{avg_val_loss}")

                # save the best-performing model
                if avg_val_loss < best_val_loss:
                    # reset the patience count
                    patience_count = 0
                    best_val_loss = avg_val_loss
                    if save_best:
                        self.export(
                            folder_path,
                            best_model_file_name
                            or TORCH_MODEL_FILE_NAME_PATTERN.format(
                                callback="best", checkpoint=str(epoch)
                            ),
                            must_trained=False,
                        )

                # early stop to prevent overfitting
                if early_stop:
                    if avg_val_loss > avg_loss:
                        patience_count += 1
                    if patience_count > int(math.ceil(patience * num_epochs)):
                        break

            else:
                _LOGGER.info(f"EPOCH {epoch + 1}: loss: -{avg_loss}")
        self.trained = True

    def train_one_epoch(self, training_data) -> torch.Tensor:
        """
        Based on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

        One epoch's training loop

        :return: the average training loss of one epoch
        """
        epoch_loss = 0.0

        # train on each batch
        for i, (x, y, target) in enumerate(training_data):
            # zero gradients for every batch
            self.optimizer.zero_grad()

            # batch prediction
            outputs = self.model(x)

            # compute loss and gradients
            batch_loss = self.calc_loss(outputs, y, target)
            batch_loss.backward()

            # adjust learning weights
            self.optimizer.step()

            # gather loss and report
            epoch_loss += batch_loss.item()
        return epoch_loss / (i + 1)

    def calc_loss(
        self, outputs: torch.Tensor, y: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculating loss when different loss function is given

        :param outputs: the output of model
        :param y: the correct label
        :param target:
        :return: the loss

        """

        if not self.config["loss"]:
            raise ValueError("Please compile the model first")

        # when all targets are 1
        # loss = 1 - cos(output, y)
        # https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        elif self.config["loss"] == "cosine_embedding_loss":
            return self.loss_fn(outputs, y, target)
        else:
            return self.loss_fn(outputs, y)

    def plot_training_hist(
        self,
        save_path: Union[str, None] = None,
        plot_file_name: Union[str, None] = DEFAULT_PLOT_FILE_NAME,
        title: Union[str, None] = DEFAULT_PLOT_TITLE,
    ) -> None:
        """
        Plot the training & validating loss of the most recent training

        :param save_path: the path of folder where image will be saved
        :param plot_file_name: the file name of the png file
        :param title: the title in the image
        :return: None
        """

        epoch_range = range(1, len(self.most_recent_train["loss"]) + 1)
        train_loss = self.most_recent_train["loss"]
        plt.figure()
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
        plt.close()

    def __repr__(self):
        return f"Vec2Vec(input_dimension={self.config['input_dim']}, output_dimension={self.config['output_dim']}, trained={self.trained})"
