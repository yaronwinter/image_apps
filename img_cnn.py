import torch.nn as nn
from collections import OrderedDict

INPUT_IMAGE_SIZE = 32
INPUT_CHANNELS = 3
NUM_CLASSES = 10

class ImgCNN(nn.Module):
    def __init__(self, conv1_channels: int, conv2_channels: int):
        super().__init__()

        # index of conv layers z1 and z2
        self.z1 = 0
        self.z2 = 3

        # feature maps
        self.feature_maps = OrderedDict()

        # switch
        self.pool_locs = OrderedDict()

        self.features = nn.Sequential(
            # Input size: (batch size, 3, 32, 32)
            nn.Conv2d(INPUT_CHANNELS, conv1_channels, 3, padding=1, stride=1),
            # Z1: (batch size, conv1_channels, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            # Max Pool 1 output size: (batch size, conv1_channels, 16, 16)

            # Input size: (batch size, conv1_channels, 16, 16)
            nn.Conv2d(32, conv2_channels, 3, padding=1, stride=1),
            # Z3: (batch size, conv2_channels, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            # Max Pool 2 output size: (batch size, conv2_channels, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Linear(conv2_channels * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, locations = layer(x)
            else:
                x = layer(x)

        # x size: (batch size, 32, 8, 8)
        # Convert to shape (batch size, 32 * 8 * 8)
        # Notice that x.size()[0] == batch size
        x = x.view(x.size()[0], -1)
        logits = self.classifier(x)
        return logits
