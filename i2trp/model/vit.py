# i2trp/model/vit.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class ViT(nn.Module):
    """
    A Vision Transformer (ViT) that processes an image grid into a sequence of embeddings.

    This implementation follows the standard ViT architecture:
    1.  The input image grid is divided into non-overlapping patches.
    2.  Each patch is linearly projected into an embedding vector using a 2D convolution.
    3.  Learnable positional encodings are added to the patch embeddings to retain
        spatial information.
    4.  The resulting sequence of vectors is fed through a standard Transformer Encoder.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ViT module.

        Args:
            config (Optional[Dict[str, Any]]): A configuration dictionary. If None,
                the default configuration will be used.
        """
        super().__init__()
        self.config = config or self.default_config()

        # --- Unpack Configuration ---
        in_channels: int = self.config['in_channels']
        input_height: int = self.config['input_height']
        input_width: int = self.config['input_width']
        patch_size: int = self.config['patch_size']
        vdim: int = self.config['vdim']
        n_heads: int = self.config['n_heads']
        n_transformer_layers: int = self.config['n_transformer_layers']
        dim_feedforward: int = self.config['dim_feedforward_transformer']
        activation: str = self.config['activation_transformer']
        
        # The main embedding dimension for the transformer.
        embedding_dim = vdim * n_heads

        # --- 1. Patch Embedding Layer ---
        # A single convolution with a kernel size and stride equal to the patch size
        # effectively projects each patch into a linear embedding.
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # --- 2. Positional Encoding ---
        # Calculate the number of patches and create learnable positional encodings.
        # These are crucial as the transformer itself has no notion of spatial order.
        num_patches_h = input_height // patch_size
        num_patches_w = input_width // patch_size
        num_patches = num_patches_h * num_patches_w
        
        self.pos_enc = nn.Parameter(torch.zeros((num_patches, 1, embedding_dim)))
        nn.init.kaiming_normal_(self.pos_enc)

        # --- 3. Transformer Encoder ---
        # A stack of standard transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation=activation,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides the default configuration for the ViT."""
        return {
            'patch_size': 16,
            'input_height': 160,
            'input_width': 240,
            'in_channels': 5,
            'vdim': 64,         # Base dimension per head
            'n_heads': 16,        # Number of attention heads
            'n_transformer_layers': 8,
            'dim_feedforward_transformer': 2048,
            'activation_transformer': 'relu'
            # Note: The total embedding dimension will be vdim * n_heads.
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input image grid of shape (B, C, H, W), where
                              B=batch_size, C=channels, H=height, W=width.

        Returns:
            torch.Tensor: A sequence of patch embeddings of shape (N, B, E), where
                          N=number_of_patches, B=batch_size, E=embedding_dim.
        """
        # --- Patch Embedding and Reshaping ---
        # x shape: (B, C, H, W)
        x = self.patch_embed(x)
        # x shape: (B, E, H_patch, W_patch)

        # Flatten the spatial dimensions (H_patch, W_patch) into a single sequence dimension.
        # Then, permute to match the transformer's expected input shape (N, B, E).
        x = torch.flatten(x, start_dim=2, end_dim=3).permute(2, 0, 1)
        # x shape: (N, B, E), where N = H_patch * W_patch

        # --- Add Positional Encoding and Normalize ---
        x = x + self.pos_enc
        x = self.norm(x)

        # --- Pass through Transformer Encoder ---
        x = self.transformer(x)
        # x final shape: (N, B, E)

        return x