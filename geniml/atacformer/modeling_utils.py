from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from atacformer.modeling_atacformer import AtacformerModel  # type: ignore[import-untyped]


def freeze_except_last_n(model: "AtacformerModel", n: int = 2):
    """
    Freeze all parameters except the last n layers of the encoder.
    Also keeps all layer norms trainable for stability.

    Args:
        model (AtacformerModel): The model to freeze.
        n (int): The number of last layers to keep trainable.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    if n > len(model.encoder.layers):
        raise ValueError(
            f"n must be less than or equal to the number of layers ({len(model.encoder.layers)})."
        )
    if n == 0:
        return
    for p in model.parameters():
        p.requires_grad = False  # freeze everything first
    for i in range(-n, 0):  # last n layers
        for p in model.encoder.layers[i].parameters():
            p.requires_grad = True
    # always keep layer norms trainable for stability
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True


def patch_atacformer_model_for_mps(model: nn.Module):
    """
    Look for any `TransformerEncoder` layers in the model and patch them
    by setting `enable_nested_tensor` to False and setting
    `use_nested_tensor` to False.

    Args:
        model (nn.Module): The model to patch.
    """
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoder):
            module.enable_nested_tensor = False
            module.use_nested_tensor = False
