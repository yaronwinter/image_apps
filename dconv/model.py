import torch
import torch.nn as nn

INPUT_IMAGE_SIZE = 32
INPUT_CHANNELS = 3
NUM_CLASSES = 10

class ImgCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_features = nn.Sequential(
            # Z1 Conv Block
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            # Z1 Conv Block
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.deconv_blocks = nn.Sequential(
            # Z2 Deconv Block
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5),

            # Z1 Deconv Block
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=INPUT_CHANNELS, kernel_size=5)
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

        self.conv2deconv = {2:3, 5:0}
        self.unpool2pool = {0:5, 3:2}

    def forward(self, images: torch.Tensor, deconv_layer=5, deconv_channel=-1) -> tuple:
        # Convolution
        features = {}
        locations = {}
        x = images.clone()
        for index, layer in enumerate(self.conv_features):
            if isinstance(layer, nn.MaxPool2d):
                x, location  = layer(x)
                features[index] = x
                locations[index] = location
            else:
                x = layer(x)
                features[index] = x

        # Predict classes
        x = x.view(x.size()[0], -1)
        logits = self.classifier(x)

        # Reconstruct the image.
        if deconv_layer not in self.conv2deconv:
            raise ValueError(f"No image reconstruction is defined for conv layer {deconv_layer}")
        
        init_dconv_layer = self.conv2deconv[deconv_layer]
        init_features = features[deconv_layer]

        if deconv_channel >= 0:
            init_features[:, :deconv_channel, :, :] = 0
            init_features[:, (deconv_channel+1):, :, :] = 0

        y = init_features.clone()
        for index in range(init_dconv_layer, len(self.deconv_blocks)):
            layer = self.deconv_blocks[index]
            if isinstance(layer)



        return logits
