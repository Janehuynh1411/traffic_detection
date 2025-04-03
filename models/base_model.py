# base_model.py

import torch
import torch.nn as nn
from pytorchvideo.models.hub import i3d_r50
from torchvision.ops import roi_align

# ------------------------------------------------------
# ROI Align Wrapper: Extracts fixed-size features from object regions
# ------------------------------------------------------
class ROI_ALIGN(nn.Module):
    def __init__(self, kernel_size, scale=1.0):
        super().__init__()
        self.roi_align = roi_align  # Use torchvision's built-in ROI Align
        self.kernel = kernel_size   # Output spatial resolution (e.g., 3x3)
        self.scale = scale          # Spatial scaling factor (e.g., 1/64 for 64x downsampled features)

    def forward(self, features, boxes):
        # Extract region-specific features from input feature maps using ROI Align
        return self.roi_align(features, boxes, self.kernel, self.scale, aligned=False)

# ------------------------------------------------------
# Base Class: Loads 3D CNN Backbone and extracts spatiotemporal features
# ------------------------------------------------------
class Base(nn.Module):
    def __init__(self, args, ego_c):
        super().__init__()
        self.args = args
        self.ego_c = ego_c            # Dimensionality for ego-centric embedding

        self.resnet = i3d_r50(True)   # Load pretrained I3D ResNet-50 from PyTorchVideo

        # Placeholder attributes to be set based on selected backbone
        self.resolution = None
        self.resolution3d = None
        self.in_c = None
        self.path_pool = None

        self.set_backbone()  # Configure backbone based on args.backbone

        # Convolutional head for ego-centric feature projection
        self.conv3d_ego = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(self.in_c),
            nn.Conv3d(self.in_c, self.ego_c, (1, 1, 1), stride=1),
            nn.AdaptiveAvgPool3d(output_size=1),  # Reduce to 1x1x1
        )

    # --------------------------------------------------
    # Select and configure backbone architecture
    # --------------------------------------------------
    def set_backbone(self):
        # Each branch loads a different depth or type of 3D CNN
        if self.args.backbone == 'i3d-2':
            self.resnet = self.resnet.blocks[:-2]
            self.in_c = 1024
            self.resolution = (16, 48)
            self.resolution3d = (4, 16, 48)

        elif self.args.backbone == 'i3d-1':
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 2048
            self.resolution = (8, 24)
            self.resolution3d = (4, 8, 24)

        elif self.args.backbone.startswith('x3d'):
            model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

            if self.args.backbone == 'x3d-1':
                # Replace the last block with a custom projection module
                self.projection = nn.Sequential(
                    nn.Conv3d(192, 256, kernel_size=(1, 1, 1), stride=1, bias=False),
                    nn.BatchNorm3d(256),
                    nn.ReLU(),
                    nn.Conv3d(256, 256, kernel_size=(3, 3, 3), dilation=(3, 1, 1), stride=1, padding='same', bias=False),
                    nn.BatchNorm3d(256)
                )
                model.blocks[-1] = self.projection
                self.resnet = model.blocks
                self.in_c = 256
                self.resolution = (8, 24)
                self.resolution3d = (16, 8, 24)

            elif self.args.backbone == 'x3d-2':
                self.resnet = model.blocks[:-1]
                self.in_c = 192
                self.resolution = (8, 24)
                self.resolution3d = (16, 8, 24)

            elif self.args.backbone == 'x3d-3':
                self.resnet = model.blocks[:-2]
                self.in_c = 96
                self.resolution = (16, 48)
                self.resolution3d = (16, 16, 48)

            elif self.args.backbone == 'x3d-4':
                self.resnet = model.blocks[:-3]
                self.in_c = 48
                self.resolution = (32, 96)
                self.resolution3d = (16, 32, 96)

        elif self.args.backbone == 'slowfast':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True).blocks[:-2]
            self.path_pool = nn.AdaptiveAvgPool3d((8, 8, 24))
            self.in_c = 2304
            self.resolution = (8, 24)
            self.resolution3d = (8, 8, 24)

    # --------------------------------------------------
    # Feature extractor that handles both regular and SlowFast backbones
    # --------------------------------------------------
    def extract_features(self, x):
        seq_len = len(x)

        if self.args.backbone == 'slowfast':
            # Create slow pathway: Sample every 4th frame
            slow_x = [x[i] for i in range(0, seq_len, 4)]

            # Convert to tensor: [sequence, batch, channels, height, width] → [batch, channels, sequence, height, width]
            x = torch.stack(x, dim=0).permute(1, 2, 0, 3, 4)
            slow_x = torch.stack(slow_x, dim=0).permute(1, 2, 0, 3, 4)

            x = [slow_x, x]  # SlowFast expects a list of two tensors

            for block in self.resnet:
                x = block(x)

            x[1] = self.path_pool(x[1])  # Pool fast pathway output
            x = torch.cat((x[0], x[1]), dim=1)  # Concatenate slow and fast features

        else:
            # Standard path for I3D or X3D
            x = torch.stack(x, dim=0).permute(1, 2, 0, 3, 4)  # [sequence, batch, C, H, W] → [batch, C, sequence, H, W]

            for block in self.resnet:
                x = block(x)

        return x  # Final spatiotemporal features
