import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dconv import tester
from dconv import model as dconv_model
from conv import img_cnn as conv_model

def show_deconv(model: nn.Module, image: torch.Tensor, device: torch.device):
    _, axs = plt.subplots(2, 1)
    new_img = tester.reconstruct_image(model, image.unsqueeze(0), 5, -1, device).detach()
    orig_img = image.squeeze() / 2 + 0.5
    new_img = new_img.squeeze() / 2 + 0.5
    axs[0].imshow(orig_img.numpy().transpose((1,2,0)))
    axs[0].set_title("original")
    axs[0].axis("off")
    axs[1].imshow(new_img.numpy().transpose((1,2,0)))
    axs[1].set_title("reconstructed")
    axs[1].axis("off")
    plt.show()

def show_layer_channel(model: nn.Module, image: torch.Tensor, layer: int, channels: list, device: torch.device):
    _, axs = plt.subplots(1, len(channels) + 1)
    orig_img = image.squeeze() / 2 + 0.5
    axs[0].imshow(orig_img.numpy().transpose((1,2,0)))
    axs[0].set_title("original")
    axs[0].axis("off")

    for c in range(len(channels)):
        new_img = tester.reconstruct_image(model, image.unsqueeze(0), layer, channels[c], device).detach()
        new_img = new_img.squeeze() / 2 + 0.5
        axs[c + 1].imshow(new_img.numpy().transpose((1,2,0)))
        axs[c + 1].set_title(f"l={layer}, c={channels[c]}")
        axs[c + 1].axis("off")
    plt.show()

def load_dconv_model(model_path: str, device: torch.device) -> nn.Module:
    model = dconv_model.ImgCNN()
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

def load_conv_model(model_path: str, device: torch.device) -> nn.Module:
    model = conv_model.ImgCNN()
    model.load_state_dict(torch.load(model_path))
    return model.to(device)
