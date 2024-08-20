import torch
import torch.nn as nn

class SpatialUpBlock(nn.Module):
    def __init__(self, x_in_dim, out_dim):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.norm = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.ConvTranspose3d(x_in_dim, out_dim, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, x):
        return self.conv(self.silu(self.norm(x)))    
    

class SpatialBlock(nn.Module):
    def __init__(self, x_in_dim, out_dim, stride):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.bn = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.Conv3d(x_in_dim, out_dim, 3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(self.silu(self.bn(x)))
    

class Spatial3DNet(nn.Module):
        def __init__(self, input_dim=128, dims=(32, 64, 128, 256)):
            super().__init__()
            d0, d1, d2, d3 = dims

            self.init_conv = nn.Conv3d(input_dim, d0, 3, 1, 1)  # 32
            self.conv0 = SpatialBlock(d0,  d0, stride=1)

            self.conv1 = SpatialBlock(d0, d1, stride=2)
            self.conv2_0 = SpatialBlock(d1, d1, stride=1)
            self.conv2_1 = SpatialBlock(d1, d1, stride=1)

            self.conv3 = SpatialBlock(d1, d2, stride=2)
            self.conv4_0 = SpatialBlock(d2, d2, stride=1)
            self.conv4_1 = SpatialBlock(d2, d2, stride=1)

            self.conv5 = SpatialBlock(d2, d3, stride=2)
            self.conv6_0 = SpatialBlock(d3, d3, stride=1)
            self.conv6_1 = SpatialBlock(d3, d3, stride=1)

            self.conv7 = SpatialUpBlock(d3, d2)
            self.conv8 = SpatialUpBlock(d2, d1)
            self.conv9 = SpatialUpBlock(d1, d0)

        def forward(self, x):
    
            x = self.init_conv(x)
            conv0 = self.conv0(x)

            x = self.conv1(conv0)
            x = self.conv2_0(x)
            conv2 = self.conv2_1(x)

            x = self.conv3(conv2)
            x = self.conv4_0(x)
            conv4 = self.conv4_1(x)

            x = self.conv5(conv4)
            x = self.conv6_0(x)
            x = self.conv6_1(x)

            x = conv4 + self.conv7(x)
            x = conv2 + self.conv8(x)
            x = conv0 + self.conv9(x)
            return x