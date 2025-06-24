import torch
import torch.nn as nn
from typing import Dict, Any

from i2trp.model.mlp2 import CMLP2
from i2trp.model.fc import CFC


class Predictor(nn.Module):
    """
    A multi-headed prediction module that generates outputs for various tasks.

    This module contains a collection of "heads," each being a small network
    (CFC or CMLP2) responsible for a specific prediction. The heads are
    dynamically created based on the provided configuration.

    Heads are categorized into two types:
    1.  Grid Heads: Operate on 2D feature maps (e.g., for segmentation).
        Invoked via `forward_conv`.
    2.  Set Heads: Operate on feature vectors (e.g., for classification or
        property prediction of a set of items). Invoked via `forward_mlp`.
    """
    # Define categories for prediction heads as immutable sets for efficient lookup
    _SET_HEAD_KEYS = frozenset(['selection', 'topology'])
    _GRID_HEAD_KEYS = frozenset(['blob', 'root_blob', 'leaf_blob', 'segmentation'])

    # Define a static map for head output dimensions
    _HEAD_OUTPUT_MAP = {
        'blob': 1,
        'root_blob': 1,
        'leaf_blob': 1,
        'segmentation': 1,
        'selection': 1,
        'topology': 3  # Default topology size
    }

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initializes the Predictor module.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
                If None, uses `default_config()`. Expected keys include 'heads',
                'in_dims', 'hidden_dim', 'head_layers', etc.
        """
        super().__init__()
        self.config = config or self.default_config()

        self.heads = nn.ModuleDict()
        for head_str in self.config['heads']:
            self._add_head(head_str, self.config)

    def _add_head(self, head_str: str, config: Dict[str, Any]) -> None:
        """Factory method to create and register a single prediction head."""
        out_dims = self.head_output(config)[head_str]

        if config['head_layers'] == 2:
            # Use a two-layer MLP for more complex predictions
            self.heads[head_str] = CMLP2(
                in_dims=config['in_dims'],
                hidden_dims=config['hidden_dim'],
                out_dims=out_dims,
                nonlinearity=config['nonlinearity']
            )
        else:
            # Use a single linear layer for simpler predictions
            self.heads[head_str] = CFC(
                in_dims=config['in_dims'],
                out_dims=out_dims,
                nonlinearity=None  # Final activation is handled in forward pass
            )

    @classmethod
    def head_output(cls, config: Dict[str, Any]) -> Dict[str, int]:
        """
        Returns a dictionary mapping head names to their output dimensions,
        adjusted by the current configuration.
        """
        head_dims = cls._HEAD_OUTPUT_MAP.copy()
        # Allow overriding the default topology size from the config
        if 'topology_size' in config:
            head_dims['topology'] = config['topology_size']
        return head_dims

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides the default configuration for the Predictor."""
        return {
            'in_dims': 64,
            'hidden_dim': 256,
            'head_layers': 2,
            'heads': ['blob'],
            'nonlinearity': 'gelu',
            'topology_size': 3,
            'sel_out': 'logits',  # 'sigmoid' or 'logits'
            'grid_out': {
                'root_blob': 'sigmoid',
                'leaf_blob': 'sigmoid',
                'segmentation': 'sigmoid'
            }
        }

    def forward_conv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass for all registered grid heads.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of predictions from each
                                     grid head.
        """
        # Generate raw outputs (logits) for all grid heads
        outputs = {
            key: head.forward_conv(x)
            for key, head in self.heads.items()
            if key in self._GRID_HEAD_KEYS
        }

        # Apply final sigmoid activations to specific grid heads based on config
        grid_activations = self.config.get('grid_out', {})
        for key, activation_type in grid_activations.items():
            if key in outputs and activation_type == 'sigmoid':
                outputs[key] = torch.sigmoid(outputs[key])

        return outputs

    def forward_mlp(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass for all registered set heads.

        Args:
            x (torch.Tensor): Input feature tensor of shape (..., C).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of predictions from each
                                     set head.
        """
        # Generate raw outputs (logits) for all set heads
        outputs = {
            key: head.forward_mlp(x)
            for key, head in self.heads.items()
            if key in self._SET_HEAD_KEYS
        }

        # Apply final activation to the 'selection' head if configured
        if 'selection' in outputs and self.config['sel_out'] == 'sigmoid':
            outputs['selection'] = torch.sigmoid(outputs['selection'])

        return outputs