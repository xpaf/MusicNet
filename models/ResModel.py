import torch.nn as nn


class ResModel(nn.Module):

    def __init__(self):
        super().__init__()

        conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_block0 = nn.Sequential(
            conv0,
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_block1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = nn.Sequential(
            conv2,
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(128, 128)
        self.res_block3 = ResidualBlock(256, 256)

        self.dense_block1 = nn.Sequential(
            nn.Linear(110592, 1024),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.dense_block2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.dense_block3 = nn.Sequential(
            nn.Linear(256, 10)
        )

        self.flat = nn.Flatten()

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, out):
        out = self.conv_block0(out)
        out = self.res_block1(out)
        out = self.maxpool(out)

        out = self.conv_block1(out)
        out = self.res_block2(out)
        out = self.maxpool(out)

        out = self.conv_block2(out)
        out = self.res_block3(out)
        out = self.maxpool(out)

        out = self.flat(out)
        out = self.dense_block1(out)
        out = self.dense_block2(out)
        out = self.dense_block3(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.relu = nn.ReLU()

    def forward(self, y):
        residual = y
        out = self.conv1(y)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)

        return out
