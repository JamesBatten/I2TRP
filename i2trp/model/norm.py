import torch.nn as nn
from typing import Dict, Type, Optional

# A central, case-insensitive mapping from string identifiers to normalization layer classes.
# This simplifies maintenance and extension.
_NORMALIZATION_LAYERS: Dict[str, Type[nn.Module]] = {
    'groupnorm': nn.GroupNorm,
    'layernorm': nn.LayerNorm,
}


def make_norm(
    out_dims: int,
    norm: str = 'groupnorm',
    numgroups: int = 8
) -> nn.Module:
    """
    Factory function to create and instantiate a normalization layer.

    Args:
        out_dims (int): The number of features or channels of the input tensor.
        norm (str): The name of the normalization layer to instantiate.
        numgroups (int): The number of groups for GroupNorm. This is ignored
            by other normalization types like LayerNorm.

    Returns:
        nn.Module: An instantiated PyTorch normalization layer.

    Raises:
        ValueError: If `norm` is None or an unrecognized string.
    """
    if norm is None:
        # Preserve original behavior of raising an error for None input.
        raise ValueError("The 'norm' argument cannot be None for `make_norm`.")

    norm_key = norm.lower()
    norm_class = _NORMALIZATION_LAYERS.get(norm_key)

    if norm_class is None:
        raise ValueError(
            f"Unknown normalization type: '{norm}'. "
            f"Available options are: {list(_NORMALIZATION_LAYERS.keys())}"
        )

    # Handle layers with different constructor arguments
    if norm_class == nn.GroupNorm:
        return nn.GroupNorm(numgroups, out_dims)
    
    if norm_class == nn.LayerNorm:
        return nn.LayerNorm(out_dims)

    # This fallback is for future-proofing, in case a new norm type with a
    # simple constructor is added to the map without specific handling.
    return norm_class(out_dims)


def make_norm_layer(norm: Optional[str] = 'groupnorm') -> Optional[Type[nn.Module]]:
    """
    Factory function that returns the class of a normalization layer, not an instance.

    Args:
        norm (Optional[str]): The name of the normalization layer. If None,
            the function returns None.

    Returns:
        Optional[Type[nn.Module]]: The PyTorch normalization layer class, or None.

    Raises:
        ValueError: If `norm` is an unrecognized string.
    """
    if norm is None:
        return None

    norm_key = norm.lower()
    norm_class = _NORMALIZATION_LAYERS.get(norm_key)

    if norm_class is None:
        raise ValueError(
            f"Unknown normalization type: '{norm}'. "
            f"Available options are: {list(_NORMALIZATION_LAYERS.keys())}"
        )

    return norm_class