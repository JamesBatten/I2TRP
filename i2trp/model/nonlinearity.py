import torch.nn as nn
from typing import Dict, Type

def make_nlrity(nonlinearity: str = 'relu') -> nn.Module:
    """
    Factory function for creating torch.nn non-linearity modules from a string identifier.

    This function provides a convenient way to instantiate activation functions,
    making model configurations cleaner and more extensible.

    Args:
        nonlinearity (str): The name of the non-linearity.
                            Case-insensitive. Defaults to 'relu'.

    Returns:
        nn.Module: An instance of the corresponding PyTorch activation module.

    Raises:
        ValueError: If the provided `nonlinearity` name is not recognized.
    """
    # A mapping from string names to their corresponding nn.Module classes.
    # This is easily extensible with new activation functions.
    nonlinearity_map: Dict[str, Type[nn.Module]] = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        # Example of how to add more:
        # 'silu': nn.SiLU,
        # 'leakyrelu': nn.LeakyReLU,
    }

    module_class = nonlinearity_map.get(nonlinearity.lower())
    
    if module_class is None:
        raise ValueError(
            f"Unknown non-linearity: '{nonlinearity}'. "
            f"Available options are: {list(nonlinearity_map.keys())}"
        )
        
    return module_class()