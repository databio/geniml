import logging
import math
import subprocess
from functools import partial
from typing import List, Dict
from collections import defaultdict

import torch
import scanpy as sc

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from gtars.tokenizers import Tokenizer
from gtars.models import Region
from transformers import DataCollatorForLanguageModeling, TrainerCallback
from transformers.integrations.integration_utils import is_wandb_available

from .data_processing import TrainingTokenizer


def get_git_hash() -> str:
    """
    Get the current git hash of the repository.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        raise RuntimeError("Could not get git hash. Make sure you are in a git repository.")


def tokenize_anndata(adata: sc.AnnData, tokenizer: Tokenizer):
    """
    Tokenize an AnnData object. This is more involved, so it gets its own function.

    Args:
        adata (sc.AnnData): The AnnData object to tokenize.
        tokenizer (Tokenizer): The tokenizer to use.
    """
    # extract regions from AnnData
    # its weird because of how numpy handle Intervals, the parent class of Region,
    # see here:
    # https://stackoverflow.com/a/43722306/13175187
    adata_features = [
        Region(chr, int(start), int(end))
        for chr, start, end in tqdm(
            zip(adata.var["chr"], adata.var["start"], adata.var["end"]),
            total=adata.var.shape[0],
            desc="Extracting regions from AnnData",
        )
    ]
    features = np.ndarray(len(adata_features), dtype=object)
    for i, region in enumerate(adata_features):
        features[i] = region
    del adata_features
    # tokenize
    tokenized = []
    x = adata.X
    for row in tqdm(
        range(adata.shape[0]),
        total=adata.shape[0],
        desc="Tokenizing",
    ):
        _, non_zeros = x[row].nonzero()
        regions = features[non_zeros]
        tokenized.append(tokenizer(regions))
    return tokenized


class WandbMixin:
    def __init__(self, *args, **kwargs):
        if not is_wandb_available():
            raise RuntimeError(
                "WandbCallback requires wandb to be installed. Run `pip install wandb`."
            )
        import wandb

        self.wandb = wandb
        super().__init__(*args, **kwargs)


class DataCollatorForReplacedTokenDetection(WandbMixin, DataCollatorForLanguageModeling):
    """
    Like HF’s MLM collator but:
      • never uses [MASK]
      • picks replacement tokens from a user-supplied distribution
      • returns per-token 0/1 labels for ELECTRA-style discrimination
    """

    def __init__(
        self,
        tokenizer: TrainingTokenizer,
        mlm_probability: float = 0.15,
        seed: int | None = None,
    ):
        """
        Simple data collator for ELECTRA-style token replacement detection.
        Args:
            tokenizer (TrainingTokenizer): The tokenizer to use.
            vocab_counts (torch.Tensor | None): 1-D tensor, size == vocab, log-probs OR probs
            mlm_probability (float): Probability of masking a token.
            seed (int | None): Random seed for reproducibility.
        """
        super().__init__(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            mask_replace_prob=0.0,  # we’ll never emit [MASK]
            random_replace_prob=1.0,  # always replace
            seed=seed,
        )

        self.mlm_probability = mlm_probability

        # generate a uniform distribution
        # over the vocabulary (ignoring special tokens)
        vocab_counts = torch.ones(tokenizer.vocab_size, dtype=torch.float)
        vocab_counts[tokenizer.all_special_ids] = 0
        self.uniform_vocab_probs = vocab_counts / vocab_counts.sum()

    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        original = inputs.clone()

        # 1 pick positions to corrupt
        # for probability_matrix, each spot gets mlm_probability (probability of being replaced)
        probability_matrix = torch.full_like(inputs, self.mlm_probability, dtype=torch.float)
        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [self.tokenizer.get_special_tokens_mask(v, None, True) for v in inputs.tolist()],
                dtype=torch.bool,
                device=inputs.device,
            )
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        replace_mask = torch.bernoulli(probability_matrix, generator=self.generator).bool()

        # 2) sample replacements from your distribution
        num_to_replace = replace_mask.sum()
        if num_to_replace > 0:
            sampled = torch.multinomial(
                self.uniform_vocab_probs.to(inputs.device),
                num_samples=num_to_replace,
                replacement=True,
            )
            inputs[replace_mask] = sampled.long().to(inputs.device)

        # 3 discriminator labels: 1 ≙ token was replaced, 0 ≙ original
        labels = (inputs != original).long()
        labels[special_tokens_mask] = -100  # ignore loss on special / pad

        return inputs, labels


class DataCollatorForTripletLoss:
    """
    A simple data collator for triplet loss to fine-tune Atacformer for cell-type clustering
    """

    def __init__(self, tokenizer: TrainingTokenizer, max_position_embeddings: int = None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_position_embeddings = max_position_embeddings

    def _truncate(self, seq: List[int]) -> List[int]:
        if self.max_position_embeddings is not None and len(seq) > self.max_position_embeddings:
            idx = np.random.choice(len(seq), size=self.max_position_embeddings, replace=False)
            idx.sort()
            return [seq[i] for i in idx]
        return seq

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Unpack triplets and truncate if needed
        anchors = [
            torch.tensor(self._truncate(f["input_ids_anchor"]), dtype=torch.long) for f in features
        ]
        positives = [
            torch.tensor(self._truncate(f["input_ids_positive"]), dtype=torch.long)
            for f in features
        ]
        negatives = [
            torch.tensor(self._truncate(f["input_ids_negative"]), dtype=torch.long)
            for f in features
        ]

        # Pad all
        input_ids_anchor = pad_sequence(anchors, batch_first=True, padding_value=self.pad_token_id)
        input_ids_positive = pad_sequence(
            positives, batch_first=True, padding_value=self.pad_token_id
        )
        input_ids_negative = pad_sequence(
            negatives, batch_first=True, padding_value=self.pad_token_id
        )

        # Create attention masks
        attention_mask_anchor = input_ids_anchor != self.pad_token_id
        attention_mask_positive = input_ids_positive != self.pad_token_id
        attention_mask_negative = input_ids_negative != self.pad_token_id

        return {
            "input_ids_anchor": input_ids_anchor,
            "input_ids_positive": input_ids_positive,
            "input_ids_negative": input_ids_negative,
            "attention_mask_anchor": attention_mask_anchor,
            "attention_mask_positive": attention_mask_positive,
            "attention_mask_negative": attention_mask_negative,
        }


class DataCollatorForUnsupervisedBatchCorrection(DataCollatorForReplacedTokenDetection):
    def __init__(
        self,
        tokenizer: TrainingTokenizer,
        max_position_embeddings: int = None,
        mlm_probability: float = 0.15,
        seed: int | None = None,
    ):
        # call parent __init__ to properly initialize the ELECTRA token replacement
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability, seed=seed)
        self.max_position_embeddings = max_position_embeddings

    def _truncate(self, seq: List[int]) -> List[int]:
        if self.max_position_embeddings is not None and len(seq) > self.max_position_embeddings:
            idx = np.random.choice(len(seq), size=self.max_position_embeddings, replace=False)
            idx.sort()
            return [seq[i] for i in idx]
        return seq

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # unpack
        input_ids = [
            torch.tensor(self._truncate(f["input_ids"]), dtype=torch.long) for f in features
        ]
        batch_labels = torch.tensor([f["batch_id"] for f in features], dtype=torch.long)

        pad_token_id = self.tokenizer.pad_token_id

        # run the masking
        input_ids, labels = super().torch_mask_tokens(
            inputs=pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            special_tokens_mask=None,
        )

        # create attention mask
        attention_mask = input_ids != pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "batch_labels": batch_labels,  # batch labels for batch correction
        }


class ModelParameterChangeCallback(WandbMixin, TrainerCallback):
    """
    A callback to log the changes in model parameters after training.
    """

    def __init__(self, initial_params: dict[str, torch.Tensor]):
        super().__init__()

        self.initial_params = initial_params

    def _compute_param_changes(self, model: torch.nn.Module):
        """
        Compute the changes in model parameters after training.

        Args:
            model (torch.nn.Module): The model to check.
        """
        updates = defaultdict(float)
        counts = defaultdict(int)

        for (name, p) in model.named_parameters():
            delta = (p.detach().cpu() - self.initial_params[name]).norm().item()
            module = name.rsplit(".", 1)[0]  # e.g. 'encoder.layer1'
            updates[module] += delta
            counts[module] += 1

        for m in updates:
            updates[m] = math.sqrt(updates[m] / counts[m])

        return updates

    def on_log(self, args, state, control, **kwargs):
        """
        Log the changes in model parameters after training.
        """
        model = kwargs.get("model")
        step = state.global_step
        if model is not None:
            updates = self._compute_param_changes(model)
            table = self.wandb.Table(columns=["module", "delta", "step"])

            if self.wandb.run is not None:
                for module, delta in updates.items():
                    table.add_data(module, delta, step)
                self.wandb.log({"parameter_changes": table}, step=step)
            else:
                print("Parameter changes:")
                for module, delta in updates.items():
                    print(f"{module}: {delta:.4f}")
        else:
            raise ValueError(
                "Model is not available in the callback. Please check the Trainer configuration."
            )


class AdjustedRandIndexCallback(WandbMixin, TrainerCallback):
    """
    A callback to log the adjusted Rand index (ARI) during training.
    """

    def __init__(
        self,
        input_ids: List[List[int]],
        cell_type_labels: List[int],
        pad_token_id: int,
        batch_size: int = 128,
        log_every_n_steps: int = 500,
    ):
        super().__init__()
        try:
            from sklearn.metrics import adjusted_rand_score
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for AdjustedRandIndexCallback. Please install it with `pip install scikit-learn`."
            )

        assert len(input_ids) == len(
            cell_type_labels
        ), "Input IDs and cell type labels must have the same length."

        self.initial_labels = cell_type_labels
        self.num_classes = len(set(cell_type_labels))
        self.input_ids = input_ids
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.log_every_n_steps = log_every_n_steps

    def on_log(self, args, state, control, **kwargs):
        """
        Log the adjusted Rand index (ARI) during training.
        """
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans

        model = kwargs.get("model")
        step = state.global_step

        assert (
            model is not None
        ), "Model is not available in the callback. Please check the Trainer configuration."

        if model is None:
            raise ValueError(
                "Model is not available in the callback. Please check the Trainer configuration."
            )

        if step % self.log_every_n_steps != 0:
            # only compute ARI every n steps regardless of other logging
            return

        cell_embeddings = model.encode_tokenized_cells(self.input_ids, batch_size=self.batch_size)

        # detach, move to cpu, and convert to numpy
        cell_embeddings = cell_embeddings.detach().cpu().to(torch.float32).numpy()

        # perform KMeans clustering
        kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
        kmeans.fit(cell_embeddings)
        predicted_labels = kmeans.labels_

        # compute ARI
        ari = adjusted_rand_score(self.initial_labels, predicted_labels)
        if self.wandb.run is not None:
            self.wandb.log({"adjusted_rand_index": ari}, step=step)
        else:
            print(f"Adjusted Rand Index at step {step}: {ari:.4f}")


def _get_decaying_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    amp = (num_training_steps + num_warmup_steps - current_step) / (2 * num_training_steps)
    if progress >= 1.0:
        return 0.0
    return max(0.0, amp * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_decaying_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
):
    """
    Very similar to huggingfaces built-in cosine with restarts, however the amplitude slowly decreases so that
    the "kick ups" are less aggressive.

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_decaying_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_linear_schedule_with_floor_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    lam = max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )
    return max(lam, 0.5)


def get_linear_schedule_with_floor_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer, down to a floor value,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_floor_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
