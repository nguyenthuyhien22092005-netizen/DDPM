import torch
import torch.nn as nn
import torch.nn.functional as F

## KHỐI DOUBLE CONV (Conv → BN → ReLU)x2
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# Kiến trúc UNet
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        ## Downsampling path (Encoder)
        self.d1 = DoubleConv(1, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        ## Bottleneck giữa encoder và decoder
        self.bottleneck = DoubleConv(512, 1024)

        ## Upsampling path (Decoder)
        ## ConvTranspose2D để phóng to kích thước
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        # DoubleConv sau khi ghép nối skip-connection
        self.c4 = DoubleConv(1024, 512)
        self.c3 = DoubleConv(512, 256)
        self.c2 = DoubleConv(256, 128)
        self.c1 = DoubleConv(128, 64)

        # Output layer 1×1 conv → tạo mask 1 channel
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    # Forward pass
    def forward(self, x):

        #Encoder
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        # Bottleneck
        bn = self.bottleneck(self.pool(d4))

        # Decoder
        u4 = self.u4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.c4(u4)

        u3 = self.u3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.c3(u3)

        u2 = self.u2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.c2(u2)

        u1 = self.u1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.c1(u1)

        # output mask với sigmoid (0–1)
        out = self.output(u1)
        return torch.sigmoid(out)
