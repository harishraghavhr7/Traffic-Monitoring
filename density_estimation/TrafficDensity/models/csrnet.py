import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from typing import List, Union

class CSRNet(nn.Module):
    """
    CSRNet (Congested Scene Recognition Network) implementation for crowd and vehicle counting.
    CVPR 2018.

    It consists of:
    - A frontend using the first 10 convolutional layers of VGG-16 (with 3 pooling layers).
    - A backend using dilated convolutional layers to capture multi-scale context without resolution loss.
    - An output layer (1x1 conv) producing a single-channel density map.
    """
    def __init__(self, load_weights: bool = True):
        super(CSRNet, self).__init__()
        
        # Frontend features configuration: 'M' is MaxPool2d, integers are Conv2d output channels
        self.frontend_feat: List[Union[int, str]] = [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512
        ]
        
        # Backend features configuration: Dilated convolutions with dilation rate = 2
        self.backend_feat: List[int] = [512, 512, 512, 256, 128, 64]
        
        # Initialize submodules
        self.frontend = self._make_frontend(self.frontend_feat)
        self.backend = self._make_backend(self.backend_feat)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Weight initialization
        if load_weights:
            self._initialize_weights()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
            
        Returns:
            Density map tensor of shape (B, 1, H/8, W/8).
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
        
    def _make_frontend(self, cfg: List[Union[int, str]]) -> nn.Sequential:
        """
        Builds the frontend convolutional layers from the configuration list.
        """
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        return nn.Sequential(*layers)
        
    def _make_backend(self, cfg: List[int]) -> nn.Sequential:
        """
        Builds the backend dilated convolutional layers from the configuration list.
        Dilated layers use dilation=2 and padding=2 to maintain spatial dimensions.
        """
        layers: List[nn.Module] = []
        in_channels = 512  # Input size matching the output of the frontend (conv4_3)
        for v in cfg:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True)
            ]
            in_channels = v
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """
        Loads pretrained VGG-16 weights for the frontend, and initializes backend/output 
        layers using normal distribution (mean=0, std=0.01) and zero bias.
        """
        # Load VGG-16 weights (handles compatibility across older and newer torchvision versions)
        try:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except Exception:
            vgg = vgg16(pretrained=True)
            
        vgg_state_dict = vgg.features.state_dict()
        frontend_state_dict = self.frontend.state_dict()
        
        # Copy matching state dict parameters from pretrained VGG-16
        new_state_dict = {}
        for k, v in vgg_state_dict.items():
            # vgg_state_dict keys are like '0.weight', '0.bias', '2.weight', etc.
            # We copy weights only for layers index < 23 (up to conv4_3 ReLU)
            layer_idx = int(k.split('.')[0])
            if layer_idx < 23:
                new_state_dict[k] = v
                
        # Load the weights into our frontend
        self.frontend.load_state_dict(new_state_dict)
        print("Frontend initialized with pretrained VGG-16 weights successfully.")
        
        # Initialize backend modules using a normal distribution
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
        # Initialize final output layer
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)
        print("Backend and output layer initialized with normal distribution weights (std=0.01).")

if __name__ == "__main__":
    # Instantiation and sanity check
    model = CSRNet(load_weights=True)
    
    # Create a dummy batch: size 1, 3 channels, size 480x640
    dummy_input = torch.randn(1, 3, 480, 640)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    # Assert that the output height and width are exactly 1/8th of input height and width
    assert output.shape[2] == dummy_input.shape[2] // 8, "Output height is incorrect!"
    assert output.shape[3] == dummy_input.shape[3] // 8, "Output width is incorrect!"
    print("CSRNet architecture verification test passed successfully!")
