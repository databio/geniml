import logging
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    from rich.progress import track
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError(
        "Please install Machine Learning dependencies by running 'pip install geniml[ml]'"
    )

from .const import DEFAULT_N_SHUFFLES, DEFAULT_NS_POWER, DEFAULT_WINDOW_SIZE, MODULE_NAME
from .utils import shuffle_documents

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
    """Generates the windowed training data by sliding across the region sets.

    This is for the CBOW model.

    Args:
        data (List[any]): The data to generate the training data from.
        window_size (int): The window size to use.
        n_shuffles (int): The number of shuffles to perform.
        threads (int): The number of threads to use.
        padding_value (any): The padding value to use.
        return_tensor (bool): Whether or not to return the data as a tensor.

    Returns:
        List[Tuple[List[any], any]]: The contexts and targets.
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
    """Generates the windowed training data by sliding across the region sets.

    When the sliding window runs into the bounds of the list, it wraps around to
    the start or end of the array.

    Args:
        data (List[any]): The data to generate the training data from.
        window_size (int): The window size to use.
        n_shuffles (int): The number of shuffles to perform.
        threads (int): The number of threads to use.
        padding_value (any): The padding value to use.
        return_tensor (bool): Whether or not to return the data as a tensor.

    Returns:
        List[Tuple[List[any], any]]: The contexts and targets.
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
    """Generate the frequency distribution of the tokens.

    Args:
        tokens (List[List[int]]): The tokens to generate the frequency distribution from.
        vocab_length (int): The length of the vocabulary.

    Returns:
        torch.Tensor: The frequency distribution tensor.
    """
    tokens_flat = [t for tokens in tokens for t in tokens]

    # create a tensor of all zeros with the length of the vocabulary
    freq_dist = torch.zeros(vocab_length, dtype=torch.float)

    # count the number of times each token appears
    for token in track(
        tokens_flat,
        total=len(tokens_flat),
        description="Generating frequency distribution",
    ):
        freq_dist[token] += 1

    # normalize the frequency distribution
    freq_dist /= freq_dist.sum()

    return freq_dist


class NegativeSampler:
    def __init__(
        self,
        freq_dist: torch.Tensor,
        power: float = DEFAULT_NS_POWER,
        batch_size: int = None,
    ):
        """Initialize the negative sampler.

        Args:
            freq_dist (torch.Tensor): List of frequencies for each token. Must be normalized.
            power (float): The power to use for the negative sampling. It is not recommended to change this.
            batch_size (int): Optional batch size for sampling.
        """
        self.dist = freq_dist**power
        self.dist /= self.dist.sum()
        self.power = power
        self.batch_size = batch_size

    def sample(self, k: int = 5, batch_size: int = None) -> torch.Tensor:
        """Sample from the negative sampler.

        Args:
            k (int): The number of samples to draw.
            batch_size (int): Optional batch size override.

        Returns:
            torch.Tensor: The negative samples.

        Raises:
            ValueError: If batch_size is not provided.
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

    def forward(
        self,
        context: torch.Tensor,
        negative_samples: torch.Tensor,
        target: torch.Tensor,
    ):
        """Compute the negative sampling loss.

        Args:
            context (torch.Tensor): The context vectors.
            negative_samples (torch.Tensor): The negative sample vectors.
            target (torch.Tensor): The target vectors.

        Returns:
            torch.Tensor: The computed loss.
        """
        # there is one target that gets mapped to each context
        target_v_context = target.unsqueeze(1).expand(
            context.shape[0], context.shape[1], context.shape[2]
        )
        target_v_neg = target.unsqueeze(1).expand(
            negative_samples.shape[0],
            negative_samples.shape[1],
            negative_samples.shape[2],
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
