import torch.nn as nn
import torch.nn.functional as F

INPUT_IMAGE_SIZE = 32
INPUT_CHANNELS = 3
NUM_CLASSES = 10

class ImgCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input size: (batch size, 3, 32, 32)
            nn.Conv2d(INPUT_CHANNELS, 32, 3, padding=1, stride=1),
            # Z1: (batch size, 32, 32, 32)
            nn.ReLU(),
            # Input size: (batch size, 32, 32, 32)
            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            # Z2: (batch size, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            # Max Pool 1 output size: (batch size, 32, 16, 16)

            # Input size: (batch size, 32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1, stride=1),
            # Z3: (batch size, 64, 16, 16)
            nn.ReLU(),
            # Input size: (batch size, 64, 16, 16)
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            # Z4: (batch size, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            # Max Pool 2 output size: (batch size, 64, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 4096),
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
