import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from i2trp.model.conv import Conv


class DownBlock(nn.Module):
    """
    A sequence of convolutional layers that progressively downsamples the input feature map.

    This block consists of an initial downsampling convolution followed by a series of
    standard convolutions. It is highly configurable and supports an optional auxiliary
    input grid that can be added to the feature map after the first downsampling step.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the DownBlock module.

        Args:
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.
                If None, default_config() will be used.
                Expected keys:
                - n_layers (int): Total number of convolutional layers.
                - in_channels (int): Channels of the input tensor.
                - hidden_channels (int): Channels for intermediate layers.
                - out_channels (int): Channels of the output tensor.
                - k_1 (int): Kernel size and stride for the first downsampling layer.
                - k_n (int): Kernel size for subsequent layers.
                - padding_n (int): Padding for subsequent layers.
                - nonlinearity (str): Activation function for hidden layers.
                - num_groups (int): Number of groups for GroupNorm.
                - normalise_output (bool): Whether to apply normalization to the final layer.
                - activation_output (bool): Whether to apply activation to the final layer.
                - bias_output (bool): Whether to use bias in the final layer.
        """
        super().__init__()

        if config is None:
            config = self.default_config()
        self.config = config

        # Ensure the block has at least a downsampling and a processing layer.
        assert config['n_layers'] >= 2, "DownBlock must have at least 2 layers."

        layers = []
        num_layers = config['n_layers']

        for i in range(num_layers):
            is_first_layer = (i == 0)
            is_last_layer = (i == num_layers - 1)

            # Determine parameters for the current layer based on its position
            current_in_channels = config['in_channels'] if is_first_layer else config['hidden_channels']
            
            current_out_channels = config['out_channels'] if is_last_layer else config['hidden_channels']

            kernel_size = config['k_1'] if is_first_layer else config['k_n']
            stride = config['k_1'] if is_first_layer else 1
            padding = 0 if is_first_layer else config['padding_n']
            
            # Set default options for intermediate layers
            nonlinearity = config['nonlinearity']
            norm = 'groupnorm'
            bias = True

            # Modify options for the final layer based on config
            if is_last_layer:
                if not config['normalise_output']:
                    norm = None
                if not config['activation_output']:
                    nonlinearity = None
                if not config['bias_output']:
                    bias = False

            # Create and add the convolutional layer to the list
            layers.append(Conv(
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                nonlinearity=nonlinearity,
                norm=norm,
                num_groups=config['num_groups']
            ))

        self.layers = nn.ModuleList(layers)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides the default configuration for the DownBlock."""
        return {
            'n_layers': 3,
            'in_channels': 64,
            'hidden_channels': 64,
            'out_channels': 64,
            'k_1': 2,
            'k_n': 3,
            'padding_n': 1,
            'padding_mode': 'zeros',
            'normalise_output': True,
            'activation_output': True,
            'bias_output': True,
            'num_groups': 8,
            'nonlinearity': 'relu'
        }

    def forward(
        self,
        x: torch.Tensor,
        aux_input_grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass through the downsampling block.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C_in, H, W).
            aux_input_grid (Optional[torch.Tensor]): An optional auxiliary tensor to be
                added to the feature map after the first downsampling convolution.
                Its shape must be broadcastable to the feature map's shape.

        Returns:
            torch.Tensor: The output tensor of shape (B, C_out, H_out, W_out).
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add auxiliary input grid after the first (downsampling) layer
            if i == 0 and aux_input_grid is not None:
                x = x + aux_input_grid
        return x