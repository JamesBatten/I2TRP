import torchvision
import timm
import torch.nn as nn
from typing import Callable, Dict, Optional

class Backbone(nn.Module):
    """
    A configurable backbone module that wraps various pre-built models from
    torchvision and timm, adapting them for a specific number of input channels
    and output feature dimensions (vdim).

    The module uses a factory pattern to create the specified backbone,
    making it easily extensible.
    """
    def __init__(self, config: Optional[Dict] = None):
        super(Backbone, self).__init__()

        if config is None:
            config = self.default_config()
        self.config = config
        
        # This attribute was used in freeze/unfreeze methods but never initialized.
        self.freeze_bn_affine = config.get('freeze_bn_affine', False)

        model_name = config.get('model')
        if not model_name:
            raise ValueError("Model name must be specified in the configuration.")

        # --- Model Factory ---
        # A dictionary mapping model names to their creation methods.
        # This avoids a large and hard-to-maintain if/elif/else block.
        self._MODEL_FACTORIES: Dict[str, Callable[[Dict], nn.Module]] = {
            'resnet18': self._create_resnet18,
            'resnet50': self._create_resnet50,
            'resnext50_32x4d': self._create_resnext50_32x4d,
            'efficientnet_b0': self._create_timm_model,
            'mobilenet_v2': self._create_timm_model,
            'seresnext26d_32x4d': self._create_timm_model,
            'mnasnet1_0': self._create_mnasnet1_0,
        }

        factory = self._MODEL_FACTORIES.get(model_name)
        if factory is None:
            available_models = ", ".join(self._MODEL_FACTORIES.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available models are: {available_models}"
            )

        self.backbone = factory(config)

    def _create_torchvision_model(self, model_fn: Callable[[], nn.Module], fc_in_features: int, config: Dict) -> nn.Module:
        """Helper to create and adapt a standard torchvision model."""
        model = model_fn(weights=None) # Use weights=None for modern torchvision
        
        model.conv1 = nn.Conv2d(
            config['input_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(fc_in_features, config['vdim'])
        return model

    def _create_resnet18(self, config: Dict) -> nn.Module:
        """Creates a ResNet-18 model."""
        return self._create_torchvision_model(torchvision.models.resnet18, 512, config)

    def _create_resnet50(self, config: Dict) -> nn.Module:
        """Creates a ResNet-50 model."""
        return self._create_torchvision_model(torchvision.models.resnet50, 2048, config)
    
    def _create_resnext50_32x4d(self, config: Dict) -> nn.Module:
        """Creates a ResNeXt-50 32x4d model."""
        return self._create_torchvision_model(torchvision.models.resnext50_32x4d, 2048, config)

    def _create_timm_model(self, config: Dict) -> nn.Module:
        """Helper to create a model from the timm library."""
        timm_model_name_map = {
            'efficientnet_b0': 'efficientnet_b0',
            'mobilenet_v2': 'mobilenetv2_100',
            'seresnext26d_32x4d': 'seresnext26d_32x4d',
        }
        model_name = timm_model_name_map[config['model']]
        return timm.create_model(
            model_name,
            pretrained=False,
            num_classes=config['vdim'],
            in_chans=config['input_channels']
        )

    def _create_mnasnet1_0(self, config: Dict) -> nn.Module:
        """
        Creates and adapts an MNASNet 1.0 model.
        Note: This modification is fragile as it depends on the internal
        layer structure of the torchvision implementation.
        """
        model = torchvision.models.mnasnet1_0(weights=None)
        # Modify the first conv layer for custom input channels
        model.layers[0] = nn.Conv2d(
            config['input_channels'], 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Modify the final classifier layer for custom output dimensions
        model.classifier[1] = nn.Linear(1280, config['vdim'])
        return model

    def freeze_batch_norm(self):
        """Sets all BatchNorm2d layers to evaluation mode and optionally freezes their affine parameters."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if self.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def unfreeze_batch_norm(self):
        """Sets all BatchNorm2d layers to training mode and optionally unfreezes their affine parameters."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if self.freeze_bn_affine:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True

    @classmethod
    def default_config(cls) -> Dict:
        """Returns the default configuration dictionary for the Backbone."""
        return {
            'input_channels': 5,
            'vdim': 64,
            'model': 'seresnext26d_32x4d',
            'freeze_bn_affine': False, # Added to fix missing attribute error
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the backbone."""
        return self.backbone.forward(x)