# i2trp/model/i2trp.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from i2trp.model.unet import UNet
from i2trp.model.vit import ViT
from i2trp.model.mlp2 import MLP2
from i2trp.model.predictor import Predictor
from i2trp.model.patch_encoder import PatchEncoder


class I2TRP(nn.Module):
    """
    The main I2TRP model.

    This model orchestrates a pipeline to process a visual grid input (e.g., an image)
    and predict properties of a tree structure associated with it. The architecture
    consists of:
    1. A U-Net (optional) and Vision Transformer (ViT) to encode the input grid into a
       sequence of visual features.
    2. A Patch Encoder to generate feature vectors for specific locations (patches)
       of interest, which serve as queries.
    3. A Transformer Decoder that attends to the visual features (memory) using the
       patch features (queries) to produce refined representations.
    4. A final Predictor module with multiple heads to output predictions like
       node selection and topology.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the I2TRP model and its components.

        Args:
            config (Optional[Dict[str, Any]]): A configuration dictionary. If None,
                the default configuration will be used.
        """
        super().__init__()
        self.config = config or self.default_config()

        # The feature dimension for the transformer layers.
        self.ff_dim = self.config['vdim'] * self.config['n_heads']

        # Build the model's components using private helper methods
        self.unet: Optional[UNet] = self._build_unet()
        self.vit: ViT = self._build_vit()
        self.mlp_a: MLP2 = self._build_mlp_a()
        self.patch_encoder: PatchEncoder = self._build_patch_encoder()
        self.decoder: nn.TransformerDecoder = self._build_decoder()
        self.predictor: Predictor = self._build_predictor()

    def _build_unet(self) -> Optional[UNet]:
        """Builds the U-Net module based on the configuration."""
        if not self.config['include_unet']:
            # If UNet is excluded, the input channels for ViT must match the grid input.
            assert self.config['vit_in_channels'] == self.config['grid_input_channels']
            return None

        unet_config = UNet.default_config()
        unet_config.update({
            'in_channels': self.config['grid_input_channels'],
            'out_channels': self.config['vit_in_channels'],
            'hidden_channels': self.config['unet_hidden_channels'],
            'normalise': False,
            'num_groups': self.config['unet_num_groups'],
            'groups_mode': self.config['unet_groups_mode'],
            'nonlinearity': 'gelu',
            'final_nonlinearity': 'relu'
        })
        return UNet(unet_config)

    def _build_vit(self) -> ViT:
        """Builds the Vision Transformer (ViT) encoder."""
        vit_config = ViT.default_config()
        vit_config.update({
            'in_channels': self.config['vit_in_channels'],
            'input_height': self.config['vit_in_gsize'],
            'input_width': self.config['vit_in_gsize'],
            'patch_size': self.config['vit_patch_size'],
            'vdim': self.config['vdim'],
            'n_heads': self.config['n_heads'],
            'n_transformer_layers': self.config['vit_layers'],
            'dim_feedforward_transformer': self.config['dim_feedforward_transformer'],
            'activation_transformer': self.config['activation_transformer']
        })
        return ViT(vit_config)

    def _build_mlp_a(self) -> MLP2:
        """Builds the MLP for processing ViT output features."""
        return MLP2(
            self.ff_dim, self.config['mlp_hidden_dim'], self.ff_dim,
            self.config['mlp_activation']
        )

    def _build_patch_encoder(self) -> PatchEncoder:
        """Builds the PatchEncoder for generating decoder queries."""
        patch_encoder_config = PatchEncoder.default_config()
        patch_encoder_config.update({
            'in_dims': self.config['grid_input_channels'],
            'out_dims': self.ff_dim,
        })
        return PatchEncoder(patch_encoder_config)

    def _build_decoder(self) -> nn.TransformerDecoder:
        """Builds the Transformer Decoder module."""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.ff_dim,
            nhead=self.config['n_heads'],
            dim_feedforward=self.config['dim_feedforward_transformer'],
            dropout=0.0,
            activation=self.config['activation_transformer']
        )
        return nn.TransformerDecoder(decoder_layer, self.config['decoder_layers'])

    def _build_predictor(self) -> Predictor:
        """Builds the final multi-headed Predictor module."""
        predictor_config = Predictor.default_config()
        predictor_config.update({
            'in_dims': self.ff_dim,
            'heads': ['selection', 'topology'],
            'nonlinearity': self.config['mlp_activation'],
            'hidden_dim': self.config['mlp_hidden_dim'],
            'topology_size': self.config['topology_size'],
            'sel_out': self.config['sel_out']
        })
        return Predictor(predictor_config)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides the default configuration for the I2TRP model."""
        return {
            'grid_input_channels': 7,
            'include_unet': True,
            'unet_hidden_channels': 32,
            'unet_num_groups': 1,
            'unet_groups_mode': 'old',
            'max_nodes': 50,
            'n_slots': 16,
            'vdim': 64,
            'n_heads': 8,
            'mlp_hidden_dim': 2048,
            'mlp_activation': 'relu',
            'decoder_layers': 6,
            'vit_in_gsize': 288,
            'vit_in_channels': 32,
            'vit_patch_size': 24,
            'vit_layers': 8,
            'dim_feedforward_transformer': 2048,
            'activation_transformer': 'relu',
            'topology_size': 3,
            'sel_out': 'sigmoid'  # 'logits' or 'sigmoid'
        }

    def forward(
        self,
        input_grid: torch.Tensor,
        pos_patches: torch.Tensor,
        node_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        '''
        Performs the forward pass of the model.

        Arguments:
            input_grid (torch.Tensor): The main visual input grid.
                                       Shape: (B, C, H, W).
            pos_patches (torch.Tensor): Image patches corresponding to query nodes.
                                        Shape: (B, N, C, H_patch, W_patch).
            node_mask (torch.Tensor): Mask for the query nodes in the decoder.
                                      Shape: (B, N).
            (Note: Other arguments like pos_lhs, branches_lhs, etc., are currently
            unused in this configuration but are kept for API compatibility).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of predictions, e.g.,
                                     {'selection': ..., 'topology': ...}.
        '''
        # 1. Encode the input grid into visual features.
        grid_features = input_grid
        if self.unet is not None:
            grid_features = self.unet(grid_features)

        # --- Validate grid feature shape before ViT ---
        h, w = grid_features.shape[2:]
        ps = self.config['vit_patch_size']
        gs = self.config['vit_in_gsize']
        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"UNet output shape {grid_features.shape} is not divisible by ViT patch size {ps}")
        if h != gs or w != gs:
            raise ValueError(f"UNet output shape {grid_features.shape} does not match configured ViT input size {gs}")

        # 2. Process grid features through ViT and MLP to create encoder memory.
        # ViT output: (NumPatches, Batch, EmbDim)
        encoder_memory = self.vit(grid_features)
        # Permute to (Batch, NumPatches, EmbDim) for MLP and easier handling.
        encoder_memory = encoder_memory.permute(1, 0, 2)
        encoder_memory = self.mlp_a(encoder_memory)

        # 3. Encode patches to be used as queries (target) for the decoder.
        # PatchEncoder output: (Batch, NumPatches, EmbDim)
        patch_queries = self.patch_encoder(pos_patches)
        # Permute to (NumPatches, Batch, EmbDim) for nn.TransformerDecoder's `tgt` input.
        patch_queries = patch_queries.permute(1, 0, 2)

        # 4. Pass features through the Transformer Decoder.
        # The decoder attends to `encoder_memory` using `patch_queries`.
        # Permute `encoder_memory` to (NumPatches, Batch, EmbDim) for decoder's `memory` input.
        encoder_memory = encoder_memory.permute(1, 0, 2)

        # Create masks for the decoder.
        # `tgt_key_padding_mask` hides padded query elements.
        tgt_mask = (1.0 - node_mask).bool()  # (B, N_queries)

        refined_features = self.decoder(
            tgt=patch_queries,
            memory=encoder_memory,
            tgt_key_padding_mask=tgt_mask
        )
        # Permute output back to (Batch, NumPatches, EmbDim) for the predictor.
        refined_features = refined_features.permute(1, 0, 2)

        # 5. Generate final predictions from the refined features.
        return self.predictor.forward_mlp(refined_features)