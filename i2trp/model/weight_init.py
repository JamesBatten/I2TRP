# i2trp/model/weight_init.py

import torch
import torch.nn as nn
from typing import Callable


def make_weights_init(nonlinearity: str = 'relu', initialisation: str = 'xavier') -> Callable[[nn.Module], None]:
    """
    Factory function that returns a weight initialization function (a closure).

    This returned function can be passed to `model.apply()` to recursively
    initialize the weights of a model's layers. It handles common layer types
    like Linear, Conv2d, GroupNorm, and LayerNorm.

    Args:
        nonlinearity (str): The name of the non-linearity that follows the layers.
            This is used to apply specific initialization heuristics, e.g., for ReLU.
            Defaults to 'relu'.
        initialisation (str): The name of the weight initialization strategy.
            Currently, only 'xavier' (Xavier Uniform) is supported.
            Defaults to 'xavier'.

    Returns:
        Callable[[nn.Module], None]: A function that takes a PyTorch module
            and initializes its weights in-place.

    Raises:
        ValueError: If an unsupported `initialisation` string is provided.
    """
    init_method = initialisation.lower()
    if init_method != 'xavier':
        raise ValueError(
            f"Unknown initialisation method: '{initialisation}'. "
            f"Supported options are: ['xavier']"
        )

    def weights_init(m: nn.Module) -> None:
        """The actual initialization function to be applied to a module."""
        # --- Handle Convolutional and Linear Layers ---
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # Apply Xavier uniform initialization to the weight matrix.
            torch.nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                # A common heuristic for ReLU is to initialize biases to a small
                # positive value to ensure neurons are active in the beginning.
                if nonlinearity == 'relu':
                    torch.nn.init.constant_(m.bias, 0.1)
                else:
                    # Preserve original behavior for other activation functions.
                    torch.nn.init.uniform_(m.bias, -1.0, 1.0)

        # --- Handle Normalization Layers ---
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                if isinstance(m, nn.GroupNorm):
                    torch.nn.init.uniform_(m.bias, -1.0, 1.0)
                else:
                    nn.init.constant_(m.bias, 0.0)

    return weights_init