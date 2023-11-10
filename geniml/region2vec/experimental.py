import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from rich.progress import track

from .utils import shuffle_documents
from .const import MODULE_NAME, DEFAULT_WINDOW_SIZE, DEFAULT_N_SHUFFLES, DEFAULT_NS_POWER

from torch.utils.data import Dataset

_LOGGER = logging.getLogger(MODULE_NAME)


class Region2VecDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[any], any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[List[any], any]:
        # we need to return things as a tensor for proper batching
        return self.samples[idx]


def generate_window_training_data(
    data: List[List[any]],
    window_size: int = DEFAULT_WINDOW_SIZE,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    threads: int = None,
    padding_value: any = 0,
    return_tensor: bool = True,
) -> List[Tuple[List[any], any]]:
    """
    Generates the windowed training data by sliding across the region sets. This is for the CBOW model.

    :param List[any] data: The data to generate the training data from.
    :param int window_size: The window size to use.
    :param int n_shuffles: The number of shuffles to perform.
    :param int threads: The number of threads to use.
    :param any padding_value: The padding value to use.
    :param bool return_tensor: Whether or not to return the data as a tensor.

    :return Tuple[List[List[any]], List[any]]: The contexts and targets.
    """
    _LOGGER.info("Generating windowed training data.")

    # shuffle the documents
    documents = shuffle_documents(
        [[t for t in tokens] for tokens in data], n_shuffles=n_shuffles, threads=threads
    )

    # compute the context length (inputs)
    context_len_req = 2 * window_size
    # contexts = []
    # targets = []
    samples = []
    for document in track(documents, total=len(documents), description="Generating training data"):
        for i, target in enumerate(document):
            context = document[max(0, i - window_size) : i] + document[i + 1 : i + window_size + 1]

            # pad the context if necessary
            if len(context) < context_len_req:
                context = context + [padding_value] * (context_len_req - len(context))

            # contexts.append(context)
            # targets.append(target)
            if return_tensor:
                samples.append(
                    (
                        torch.tensor(context, dtype=torch.long),
                        torch.tensor(target, dtype=torch.long),
                    )
                )
            else:
                samples.append((context, target))

    # return contexts, targets
    return samples


def generate_window_training_data_wrap(
    data: List[List[any]],
    window_size: int = DEFAULT_WINDOW_SIZE,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    threads: int = None,
    padding_value: any = 0,
    return_tensor: bool = True,
) -> List[Tuple[List[any], any]]:
    """
    Generates the windowed training data by sliding across the region sets. When the sliding window runs into the bounds of the list, it wraps around to the start or end of the array.

    :param List[any] data: The data to generate the training data from.
    :param int window_size: The window size to use.
    :param int n_shuffles: The number of shuffles to perform.
    :param int threads: The number of threads to use.
    :param any padding_value: The padding value to use.
    :param bool return_tensor: Whether or not to return the data as a tensor.

    :return Tuple[List[List[any]], List[any]]: The contexts and targets.
    """
    _LOGGER.info("Generating windowed training data.")

    # shuffle the documents
    documents = shuffle_documents(
        [[t for t in tokens] for tokens in data], n_shuffles=n_shuffles, threads=threads
    )

    samples = []

    for document in track(documents, total=len(documents), description="Generating training data"):
        for i in range(0, window_size):
            target = document[i]
            context = (
                document[i - window_size :] + document[0:i] + document[i + 1 : i + 1 + window_size]
            )
            if return_tensor:
                samples.append(
                    (
                        torch.tensor(context, dtype=torch.long),
                        torch.tensor(target, dtype=torch.long),
                    )
                )
            else:
                samples.append((context, target))
        for i in range(window_size, len(document) - window_size):
            target = document[i]
            context = document[i - window_size : i] + document[i + 1 : i + 1 + window_size]
            if return_tensor:
                samples.append(
                    (
                        torch.tensor(context, dtype=torch.long),
                        torch.tensor(target, dtype=torch.long),
                    )
                )
            else:
                samples.append((context, target))
        for i in range(len(document) - window_size, len(document)):
            target = document[i]
            context = (
                document[i - window_size : i]
                + document[i + 1 :]
                + document[0 : i - len(document) + window_size + 1]
            )
            if return_tensor:
                samples.append(
                    (
                        torch.tensor(context, dtype=torch.long),
                        torch.tensor(target, dtype=torch.long),
                    )
                )
            else:
                samples.append((context, target))

    return samples


def generate_frequency_distribution(tokens: List[List[int]], vocab_length: int) -> torch.Tensor:
    """
    Generate the frequency distribution of the tokens.

    :param List[List[int]] tokens: The tokens to generate the frequency distribution from.
    """
    tokens_flat = [t for tokens in tokens for t in tokens]

    # create a tensor of all zeros with the length of the vocabulary
    freq_dist = torch.zeros(vocab_length, dtype=torch.float)

    # count the number of times each token appears
    for token in track(
        tokens_flat, total=len(tokens_flat), description="Generating frequency distribution"
    ):
        freq_dist[token] += 1

    # normalize the frequency distribution
    freq_dist /= freq_dist.sum()

    return freq_dist


class NegativeSampler:
    def __init__(
        self, freq_dist: torch.Tensor, power: float = DEFAULT_NS_POWER, batch_size: int = None
    ):
        """
        Initialize the negative sampler.

        :param torch.Tensor freq_dist: List of frequencies for each token. Must be normalized.
        :param float power: The power to use for the negative sampling. It is not recommended to change this.
        """
        self.dist = freq_dist**power
        self.dist /= self.dist.sum()
        self.power = power
        self.batch_size = batch_size

    def sample(self, k: int = 5, batch_size: int = None) -> torch.Tensor:
        """
        Sample from the negative sampler.

        :param int k: The number of samples to draw.
        """
        batch_size = batch_size or self.batch_size
        if batch_size is None:
            raise ValueError(
                "Must provide batch_size to sample from negative sampler. This can be set in the constructor or in the sample method."
            )
        negative_samples = torch.multinomial(self.dist, batch_size * k, replacement=True)
        return negative_samples.view(batch_size, k)


class NegativeSampleDataset(Dataset):
    def __init__(self, samples: torch.Tensor):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.samples[idx]


# negative sampling loss
class NSLoss(nn.Module):
    def __init__(self):
        super(NSLoss, self).__init__()

    def forward(self, context: torch.Tensor, negative_samples: torch.Tensor, target: torch.Tensor):
        """
        :param torch.Tensor context: The context vectors.
        :param torch.Tensor negative_samples: The negative sample vectors.
        :param torch.Tensor target: The target vectors.
        """
        # there is one target that gets mapped to each context
        target_v_context = target.unsqueeze(1).expand(
            context.shape[0], context.shape[1], context.shape[2]
        )
        target_v_neg = target.unsqueeze(1).expand(
            negative_samples.shape[0], negative_samples.shape[1], negative_samples.shape[2]
        )

        # target is now of shape (batch_size, num_context_vectors, embedding_size)
        # negative_samples is of shape (batch_size, num_negative_samples, embedding_size)
        # context is of shape (batch_size, num_context_vectors, embedding_size)

        # compute the dot product between the context and target
        pos_loss = torch.sum(
            torch.nn.functional.logsigmoid(torch.bmm(context, target_v_context.transpose(1, 2)))
        )
        neg_loss = torch.sum(
            torch.nn.functional.logsigmoid(
                torch.bmm(-negative_samples, target_v_neg.transpose(1, 2))
            )
        )

        return -(pos_loss + neg_loss)
