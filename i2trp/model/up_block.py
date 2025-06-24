import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from i2trp.model.conv import Conv


class UpBlock(nn.Module):
    """
    A sequence of convolutional layers that progressively upsamples a feature map.

    This block is a core component of a U-Net decoder. It begins with an
    upsampling operation combined with a 1x1 convolution, followed by a series of
    standard convolutions. It also supports an optional auxiliary input grid, which
    is added to the feature map after the initial upsampling step.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the UpBlock module.

        Args:
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.
                If None, `default_config()` will be used. See `default_config`
                for expected keys.
        """
        super().__init__()

        self.config = config or self.default_config()
        assert self.config['n_layers'] >= 2, "UpBlock must have at least 2 layers."

        self.layers = nn.ModuleList()
        self._build_layers()

    def _build_layers(self) -> None:
        """Constructs the sequence of convolutional layers for the block."""
        config = self.config
        num_layers = config['n_layers']
        
        layers = []
        for i in range(num_layers):
            is_first_layer = (i == 0)
            is_last_layer = (i == num_layers - 1)

            # --- Determine layer parameters based on position ---
            
            # 1. Channel dimensions
            current_in_channels = config['in_channels'] if is_first_layer else config['hidden_channels']
            current_out_channels = config['out_channels'] if is_last_layer else config['hidden_channels']

            # 2. Convolution parameters
            # The first layer is a 1x1 conv combined with upsampling.
            kernel_size = 1 if is_first_layer else config['kernel_size']
            padding = 0 if is_first_layer else config['padding_n']
            upsample_factor = config['upsample_factor'] if is_first_layer else None

            # 3. Layer options (norm, activation, bias)
            # Set defaults for intermediate layers
            norm = 'groupnorm'
            nonlinearity = config['nonlinearity']
            bias = True

            # Override defaults for the final layer based on config
            if is_last_layer:
                if config['final_nonlinearity'] is not None:
                    nonlinearity = config['final_nonlinearity']
                elif not config['activation_output']:
                    nonlinearity = None
                
                if not config['normalise_output']:
                    norm = None
                
                if not config['bias_output']:
                    bias = False
            
            # --- Create and add the convolutional layer ---
            layers.append(Conv(
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
                nonlinearity=nonlinearity,
                norm=norm,
                num_groups=config['num_groups'],
                upsample_factor=upsample_factor
            ))

        self.layers = nn.ModuleList(layers)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides the default configuration for the UpBlock."""
        return {
            'n_layers': 3,
            'in_channels': 64,
            'hidden_channels': 64,
            'out_channels': 64,
            'upsample_factor': 2,
            'kernel_size': 3,
            'padding_n': 1,
            'padding_mode': 'zeros',
            'normalise_output': True,
            'activation_output': True,
            'num_groups': 8,
            'bias_output': True,
            'nonlinearity': 'gelu',
            'final_nonlinearity': None
        }

    def forward(
        self,
        x: torch.Tensor,
        aux_input_grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass through the upsampling block.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C_in, H, W).
            aux_input_grid (Optional[torch.Tensor]): An optional auxiliary tensor to be
                added to the feature map after the first upsampling convolution.
                Its shape must be broadcastable to the feature map's shape.

        Returns:
            torch.Tensor: The output tensor of shape (B, C_out, H_out, W_out).
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add auxiliary input grid after the first (upsampling) layer
            if i == 0 and aux_input_grid is not None:
                x = x + aux_input_grid
        return x