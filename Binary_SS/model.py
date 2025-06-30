import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        current_channels = in_channels
        for feature in features:
            self.down_blocks.append(DoubleConv(current_channels, feature))
            current_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        for feature in reversed(features):
            self.up_blocks.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(feature * 2, feature))

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.up_blocks), 2):
            x = self.up_blocks[i](x)
            skip_conn = skip_connections[i // 2]

            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat((skip_conn, x), dim=1)
            x = self.up_blocks[i + 1](x)

        return self.final_conv(x)

    
def test():
    x = torch.randn((3,1,128,128))
    model = UNet(in_ch=1, out_ch=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape



if __name__ == "__main__":
    test()





 


