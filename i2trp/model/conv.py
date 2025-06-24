import torch
import torch.nn as nn
from typing import Optional

from i2trp.model.nonlinearity import make_nlrity
from i2trp.model.weight_init import make_weights_init
from i2trp.model.norm import make_norm


class Conv(nn.Module):
    """
    A composite convolutional block that bundles a sequence of operations:
    (optional) Upsampling -> Convolution -> (optional) Activation -> (optional) Normalization.

    This module uses nn.Sequential to create a clean and readable implementation,
    and is designed for convenience in building architectures like U-Nets.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool,
        nonlinearity: Optional[str],
        norm: Optional[str],
        num_groups: int,
        upsample_factor: Optional[int] = None,
        padding_mode: str = 'zeros',
    ) -> None:
        """
        Initializes the Conv module.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to all sides of the input.
            bias (bool): If True, adds a learnable bias to the output.
            nonlinearity (Optional[str]): The name of the activation function to use
                (e.g., 'relu', 'gelu'). If None, no activation is applied.
            norm (Optional[str]): The name of the normalization layer to use
                (e.g., 'groupnorm'). If None, no normalization is applied.
            num_groups (int): The number of groups for GroupNorm. Only used if
                norm is 'groupnorm'.
            upsample_factor (Optional[int]): The factor by which to bilinearly upsample
                the input before the convolution. If None, no upsampling is performed.
            padding_mode (str): The padding mode for the convolution.
                Defaults to 'zeros'.
        """
        super().__init__()

        layers = []

        # 1. Optional Upsampling Layer
        if upsample_factor is not None and upsample_factor > 1:
            layers.append(nn.UpsamplingBilinear2d(scale_factor=upsample_factor))

        # 2. Convolutional Layer
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        ))

        # 3. Optional Non-linearity (Activation)
        if nonlinearity is not None:
            layers.append(make_nlrity(nonlinearity))

        # 4. Optional Normalization Layer
        if norm is not None:
            layers.append(make_norm(out_channels, norm, num_groups))

        # Encapsulate the layers in an nn.Sequential container
        self.net = nn.Sequential(*layers)

        # Apply weight initialization to the entire sequential block.
        # The `make_weights_init` function will correctly find and initialize
        # the Conv2d and GroupNorm layers within it
        if nonlinearity is not None:
            self.net.apply(make_weights_init(nonlinearity))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the sequential convolutional block."""
        return self.net(x)