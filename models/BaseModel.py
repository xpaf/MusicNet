import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

        conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_block1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = nn.Sequential(
            conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_block3 = nn.Sequential(
            conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.dense_block1 = nn.Sequential(
            nn.Linear(110592, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.dense_block2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.dense_block3 = nn.Sequential(
            nn.Linear(256, 10),
        )

        self.flat = nn.Flatten()

    def forward(self, out):
        out = self.conv_block1(out)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.flat(out)
        out = self.dense_block1(out)
        out = self.dense_block2(out)
        out = self.dense_block3(out)

        return out
