import torch
import torch.nn as nn
from typing import Dict, Any

from i2trp.model.backbone import Backbone


class PatchEncoder(nn.Module):
    """
    Encodes image patches into fixed-size feature vectors using a CNN backbone.

    This module is designed to handle both batches of individual images (4D tensor)
    and batches of sequences of images (5D tensor). It dynamically dispatches
    to the correct processing logic based on the input tensor's dimensionality.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initializes the PatchEncoder.

        Args:
            config (Dict[str, Any], optional): A configuration dictionary. If None,
                the default configuration will be used.
                Expected keys: 'in_dims', 'out_dims', 'backbone'.
        """
        super().__init__()
        self.config = config or self.default_config()

        # Robustly create the configuration for the backbone by updating defaults
        backbone_config = Backbone.default_config()
        backbone_config.update({
            'input_channels': self.config['in_dims'],
            'vdim': self.config['out_dims'],
            'model': self.config['backbone'],
        })
        self.backbone = Backbone(backbone_config)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides the default configuration for the PatchEncoder."""
        return {
            'in_dims': 5,        # Input channels for the backbone (e.g., RGB + 2 coord channels)
            'out_dims': 64,      # Dimension of the output feature vector
            'backbone': 'resnet18', # CNN backbone model to use
        }

    def _forward_5dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a 5D tensor of batched image sequences.

        Args:
            x (torch.Tensor): A float tensor of shape (B, P, C, H, W), where B is
                              batch size, P is number of patches, and V is feature dimension.

        Returns:
            torch.Tensor: A float tensor of shape (B, P, V).
        """
        b, p, c, h, w = x.shape

        # Reshape for batch processing: merge batch and patch dimensions
        x_reshaped = x.view(b * p, c, h, w)

        # Encode the flattened batch of patches
        features = self.backbone(x_reshaped)  # Shape: (B * P, V)

        # Reshape back to the original batch and sequence structure
        return features.view(b, p, -1)  # Shape: (B, P, V)

    def _forward_4dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a 4D tensor of batched images.

        Args:
            x (torch.Tensor): A float tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: A float tensor of shape (B, V).
        """
        # The backbone directly handles 4D input
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes image patches using the configured backbone.

        This method inspects the input tensor's shape and routes it to the
        appropriate handler for either 4D (batch of images) or 5D (batch of
        image sequences) input.

        Args:
            x (torch.Tensor): Input tensor. Can be either:
                              - 4D: (batch_size, channels, height, width)
                              - 5D: (batch_size, num_patches, channels, height, width)

        Returns:
            torch.Tensor: The encoded feature tensor. Will be either:
                          - 2D: (batch_size, out_dims) if input was 4D.
                          - 3D: (batch_size, num_patches, out_dims) if input was 5D.

        Raises:
            ValueError: If the input tensor is not 4D or 5D.
        """
        if x.ndim == 5:
            return self._forward_5dim(x)
        elif x.ndim == 4:
            return self._forward_4dim(x)
        else:
            raise ValueError(
                f"Unsupported input tensor dimensions. "
                f"Expected 4 or 5, but got {x.ndim}."
            )