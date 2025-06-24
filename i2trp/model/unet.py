import torch
import torch.nn as nn
from typing import List, Dict, Any

from i2trp.model.down_block import DownBlock
from i2trp.model.up_block import UpBlock


class UNet(nn.Module):
    """
    A configurable U-Net architecture for image-to-image translation tasks.

    The U-Net consists of a contracting path (encoder) to capture context and a
    symmetric expanding path (decoder) that enables precise localization. It uses
    skip connections to pass high-resolution features from the encoder to the
    decoder, which is critical for good performance.

    The architecture is built dynamically based on a configuration dictionary,
    allowing for easy customization of depth, channels, and other parameters.
    """
    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__()
        self.config = config or self.default_config()

        # ModuleLists to hold the encoder and decoder blocks
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()

        self._build_network()

    def _build_network(self) -> None:
        """Constructs the U-Net encoder and decoder paths based on the config."""
        
        # --- Channel Dimension Calculation ---
        # Pre-calculate the number of channels at each level of the U-Net.
        # This is more robust than manually tracking 'ff_dim' in a loop.
        down_channels = [self.config['hidden_channels']]
        for _ in range(self.config['depth'] - 1):
            down_channels.append(down_channels[-1] * self.config['mult_factor'])
        # Example for default config: [16, 32, 64, 128]

        # --- Encoder (Down-sampling Path) ---
        in_ch = self.config['in_channels']
        for i in range(self.config['depth']):
            out_ch = down_channels[i]
            
            downblock_config = DownBlock.default_config()
            downblock_config.update({
                'in_channels': in_ch,
                'hidden_channels': out_ch,
                'out_channels': out_ch,
                'nonlinearity': self.config['nonlinearity'],
            })
            
            if self.config['groups_mode'] == 'old':
                downblock_config['num_groups'] = self.config['num_groups'] ** (i + 1)
            elif self.config['groups_mode'] == 'new':
                downblock_config['num_groups'] = self.config['num_groups']
            else:
                raise ValueError(f"Unknown groups_mode: {self.config['groups_mode']}")
            
            self.downblocks.append(DownBlock(downblock_config))
            in_ch = out_ch

        # --- Decoder (Up-sampling Path) ---
        # The first up-block takes the deepest features. Subsequent blocks take
        # the output of the previous up-block concatenated with a skip connection.
        in_ch = down_channels[-1]
        for i in range(self.config['depth']):
            # For all but the first up-block, the input channels are doubled
            # due to concatenation with the skip connection.
            up_in_ch = in_ch if i == 0 else in_ch * 2
            
            # The output channels decrease as we move up the decoder path.
            # The final layer outputs the desired number of 'out_channels'.
            is_last_layer = (i == self.config['depth'] - 1)
            out_ch = (
                self.config['out_channels'] if is_last_layer 
                else down_channels[self.config['depth'] - i - 2]
            )

            upblock_config = UpBlock.default_config()
            upblock_config.update({
                'in_channels': up_in_ch,
                'hidden_channels': out_ch,
                'out_channels': out_ch,
                'nonlinearity': self.config['nonlinearity'],
                'final_nonlinearity': self.config['final_nonlinearity'] if is_last_layer else None,
            })
            
            if self.config['groups_mode'] == 'old':
                upblock_config['num_groups'] = self.config['num_groups'] ** (self.config['depth'] - i)
            elif self.config['groups_mode'] == 'new':
                # Original code had a bug here, referencing 'downblock_config'. Correcting it.
                upblock_config['num_groups'] = self.config['num_groups']
            
            self.upblocks.append(UpBlock(upblock_config))
            in_ch = out_ch


    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides the default configuration for the UNet."""
        return {
            'in_channels': 5,
            'out_channels': 16,
            'hidden_channels': 16,
            'mult_factor': 2,
            'depth': 4,
            'nonlinearity': 'gelu',
            'final_nonlinearity': None,
            'normalise': True,
            'num_groups': 1,
            'groups_mode': 'old'
        }

    def _crop_and_cat(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """
        Crops the skip connection to match the spatial dimensions of the input tensor `x`
        and concatenates them along the channel dimension.

        This is necessary if the convolutions in the DownBlocks are unpadded, leading
        to a reduction in feature map size at each step.
        """
        # Get spatial dimensions
        x_size = x.shape[-1]
        skip_size = skip_connection.shape[-1]

        # Calculate padding needed to center-crop the skip connection
        delta = skip_size - x_size
        crop_start = delta // 2
        crop_end = delta - crop_start

        # Perform the crop. The `or None` handles the case where crop_end is 0.
        cropped_skip = skip_connection[:, :, crop_start : skip_size - crop_end, crop_start : skip_size - crop_end]
        
        return torch.cat([x, cropped_skip], dim=1)

    def forward(self, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Performs the forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            eps (float): A small epsilon value to prevent division by zero during normalization.

        Returns:
            torch.Tensor: The output tensor of shape (B, C_out, H_out, W_out).
        """
        # --- Encoder Path ---
        skips: List[torch.Tensor] = []
        for i, downblock in enumerate(self.downblocks):
            x = downblock(x)
            # Store feature maps for skip connections, excluding the last one (bottleneck)
            if i < self.config['depth'] - 1:
                skips.append(x)

        # --- Decoder Path ---
        # Reverse the skips list to easily pop the correct one
        skips = list(reversed(skips))
        
        for i, upblock in enumerate(self.upblocks):
            # For all but the first up-block, concatenate with the skip connection
            if i > 0:
                skip_connection = skips[i-1]
                x = self._crop_and_cat(x, skip_connection)
            
            x = upblock(x)

        # --- Final Normalization ---
        if self.config['normalise']:
            x_norm = torch.linalg.norm(x, dim=1, keepdim=True)
            x = x / (x_norm + eps)
            
        return x