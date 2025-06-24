import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Optional

from i2trp.model.nonlinearity import make_nlrity
from i2trp.model.norm import make_norm
from i2trp.model.weight_init import make_weights_init


class MLP2(nn.Module):
    """
    A simple two-layer Multi-Layer Perceptron (MLP) with a non-linearity
    and optional normalization in the hidden layer.
    
    Architecture: Linear -> Non-linearity -> [Normalization] -> Linear
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity: str = 'relu',
        bias_1: bool = True,
        bias_2: bool = True,
        norm_1: Optional[str] = None,
        numgroups: int = 8
    ) -> None:
        """
        Args:
            in_dims (int): Number of input features.
            hidden_dims (int): Number of features in the hidden layer.
            out_dims (int): Number of output features.
            nonlinearity (str): The activation function for the hidden layer.
            bias_1 (bool): Whether to use a bias in the first linear layer.
            bias_2 (bool): Whether to use a bias in the second linear layer.
            norm_1 (Optional[str]): The normalization type for the hidden layer (e.g., 'groupnorm').
            numgroups (int): The number of groups for GroupNorm, if used.
        """
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dims, hidden_dims, bias=bias_1))
        layers.append(make_nlrity(nonlinearity))
        
        if norm_1 is not None:
            layers.append(make_norm(hidden_dims, norm=norm_1, numgroups=numgroups))
            
        layers.append(nn.Linear(hidden_dims, out_dims, bias=bias_2))

        self.net = nn.Sequential(*layers)
        self.net.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the sequential MLP."""
        return self.net(x)


class ConvMLP2(nn.Module):
    """
    A two-layer MLP implemented with 1x1 convolutions. This is useful for
    applying MLP-like transformations to every spatial location in a feature map.

    Architecture: Conv2d(1x1) -> Non-linearity -> Conv2d(1x1)
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity: str = 'relu',
        bias_1: bool = True,
        bias_2: bool = True,
        stride_1: int = 1
    ) -> None:
        """
        Args:
            in_dims (int): Number of input channels.
            hidden_dims (int): Number of channels in the hidden layer.
            out_dims (int): Number of output channels.
            nonlinearity (str): The activation function for the hidden layer.
            bias_1 (bool): Whether to use a bias in the first convolutional layer.
            bias_2 (bool): Whether to use a bias in the second convolutional layer.
            stride_1 (int): Stride for the first convolutional layer.
        """
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(
            in_dims, hidden_dims, kernel_size=1, stride=stride_1, bias=bias_1
        ))
        layers.append(make_nlrity(nonlinearity))
        layers.append(nn.Conv2d(
            hidden_dims, out_dims, kernel_size=1, stride=1, bias=bias_2
        ))
        
        self.net = nn.Sequential(*layers)
        self.net.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input feature map through the 1x1 convolutions."""
        return self.net(x)


class CMLP2(nn.Module):
    """
    Composite Two-Layer MLP.

    A versatile two-layer network that shares weights to function as either a
    standard MLP (using F.linear) or a 1x1 convolutional network (using F.conv2d).
    This allows the same learned transformation to be applied to both vector-like
    and grid-like data.
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity_1: str = 'relu',
        nonlinearity_2: Optional[str] = None,
        bias_1: bool = True,
        bias_2: bool = True,
        stride_1: int = 1,
        norm_1: Optional[str] = None,
        norm_2: Optional[str] = None,
        numgroups: int = 8
    ) -> None:
        super().__init__()

        # --- Layer 1 Components ---
        self.weight_1 = Parameter(torch.empty(hidden_dims, in_dims, dtype=torch.float32))
        self.bias_1 = Parameter(torch.empty(hidden_dims).uniform_(-1.0, 1.0)) if bias_1 else None
        self.nlrity_1 = make_nlrity(nonlinearity_1)
        self.norm_1 = make_norm(hidden_dims, norm_1, numgroups) if norm_1 else None

        # --- Layer 2 Components ---
        self.weight_2 = Parameter(torch.empty(out_dims, hidden_dims, dtype=torch.float32))
        self.bias_2 = Parameter(torch.empty(out_dims).uniform_(-1.0, 1.0)) if bias_2 else None
        self.nlrity_2 = make_nlrity(nonlinearity_2) if nonlinearity_2 else None
        self.norm_2 = make_norm(out_dims, norm_2, numgroups) if norm_2 else None

        # --- Convolution & Initialization Details ---
        self.cshape1 = (hidden_dims, in_dims, 1, 1)
        self.cshape2 = (out_dims, hidden_dims, 1, 1)
        self.stride_1 = stride_1

        nn.init.xavier_normal_(self.weight_1)
        nn.init.xavier_normal_(self.weight_2)

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the network as a sequence of 1x1 convolutions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: The transformed tensor.
        """
        # First layer
        x = nn.functional.conv2d(
            x, self.weight_1.view(self.cshape1), self.bias_1, stride=self.stride_1
        )
        x = self.nlrity_1(x)
        if self.norm_1 is not None:
            x = self.norm_1(x)

        # Second layer
        x = nn.functional.conv2d(
            x, self.weight_2.view(self.cshape2), self.bias_2
        )
        if self.nlrity_2 is not None:
            x = self.nlrity_2(x)
        if self.norm_2 is not None:
            x = self.norm_2(x)
        return x

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the network as a sequence of linear transformations.

        Args:
            x (torch.Tensor): Input tensor of shape (..., C_in).

        Returns:
            torch.Tensor: The transformed tensor.
        """
        # First layer
        x = nn.functional.linear(x, self.weight_1, self.bias_1)
        x = self.nlrity_1(x)
        if self.norm_1 is not None:
            x = self.norm_1(x)

        # Second layer
        x = nn.functional.linear(x, self.weight_2, self.bias_2)
        if self.nlrity_2 is not None:
            x = self.nlrity_2(x)
        if self.norm_2 is not None:
            x = self.norm_2(x)
        return x