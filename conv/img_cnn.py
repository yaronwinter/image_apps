import torch.nn as nn
from collections import OrderedDict

INPUT_IMAGE_SIZE = 32
INPUT_CHANNELS = 3
NUM_CLASSES = 10

class ImgCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=6, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, locations = layer(x)
            else:
                x = layer(x)

        x = x.view(x.size()[0], -1)
        logits = self.classifier(x)
        return logits
