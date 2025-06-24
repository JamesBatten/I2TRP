import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Optional

from i2trp.model.nonlinearity import make_nlrity
from i2trp.model.norm import make_norm
from i2trp.model.weight_init import make_weights_init


class FC(nn.Module):
    """
    A single fully-connected (linear) layer followed by optional non-linearity and normalization.
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        nonlinearity: Optional[str] = 'relu',
        bias: bool = True,
        norm: Optional[str] = None,
        numgroups: int = 8
    ) -> None:
        """
        Args:
            in_dims (int): Number of input features.
            out_dims (int): Number of output features.
            nonlinearity (Optional[str]): Activation function name (e.g., 'relu'). If None, no activation is applied.
            bias (bool): If True, the linear layer includes a learnable bias.
            norm (Optional[str]): Normalization layer name (e.g., 'groupnorm'). If None, no normalization is applied.
            numgroups (int): Number of groups for GroupNorm, if used.
        """
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dims, out_dims, bias=bias))

        if nonlinearity is not None:
            layers.append(make_nlrity(nonlinearity))
        
        if norm is not None:
            # Use the helper to create the normalization layer
            layers.append(make_norm(out_dims, norm, numgroups))
            
        self.net = nn.Sequential(*layers)
        
        # Apply weight initialization to the sequential module
        self.net.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the sequential block."""
        return self.net(x)


class ConvFC(nn.Module):
    """
    A single 1x1 convolutional layer, effectively acting as a fully-connected layer
    on the channel dimension of a 2D feature map. Followed by optional
    non-linearity and normalization.
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        nonlinearity: Optional[str] = 'relu',
        bias: bool = True,
        stride: int = 1,
        norm: Optional[str] = None,
        numgroups: int = 8
    ) -> None:
        """
        Args:
            in_dims (int): Number of input channels.
            out_dims (int): Number of output channels.
            nonlinearity (Optional[str]): Activation function name. If None, no activation is applied.
            bias (bool): If True, the conv layer includes a learnable bias.
            stride (int): The stride of the convolution.
            norm (Optional[str]): Normalization layer name. If None, no normalization is applied.
            numgroups (int): Number of groups for GroupNorm, if used.
        """
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(
            in_dims, out_dims, kernel_size=1, stride=stride, bias=bias
        ))

        if nonlinearity is not None:
            layers.append(make_nlrity(nonlinearity))

        if norm is not None:
            layers.append(make_norm(out_dims, norm, numgroups))
            
        self.net = nn.Sequential(*layers)

        self.net.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the sequential block."""
        return self.net(x)


class CFC(nn.Module):
    """
    Composite Fully-Connected Layer.

    A versatile single layer that can be applied either as a standard linear
    transformation (MLP-style) or as a 1x1 convolution. This is achieved by sharing
    a weight parameter between `F.linear` and `F.conv2d` operations.
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        nonlinearity: Optional[str] = 'relu',
        bias: bool = True,
        stride: int = 1,
        norm: Optional[str] = None,
        numgroups: int = 8
    ) -> None:
        super().__init__()
        
        self.weight = Parameter(torch.empty((out_dims, in_dims), dtype=torch.float32))
        
        if bias:
            self.bias = Parameter(torch.zeros(out_dims, dtype=torch.float32))
        else:
            self.bias = None
        
        self.nlrity = make_nlrity(nonlinearity) if nonlinearity is not None else None
        
        self.norm = make_norm(out_dims, norm, numgroups) if norm is not None else None
        
        self.cshape = (out_dims, in_dims, 1, 1)
        self.stride = stride

        nn.init.xavier_normal_(self.weight)

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer as a 1x1 convolution.

        Args:
            x (torch.Tensor): A float tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: The output tensor.
        """
        x = nn.functional.conv2d(
            x, self.weight.view(self.cshape), self.bias, stride=self.stride
        )
        if self.nlrity is not None:
            x = self.nlrity(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer as a standard linear transformation.

        Args:
            x (torch.Tensor): A float tensor of shape (..., C_in).

        Returns:
            torch.Tensor: The output tensor.
        """
        x = nn.functional.linear(x, self.weight, self.bias)
        if self.nlrity is not None:
            x = self.nlrity(x)
        if self.norm is not None:
            x = self.norm(x)
        return x