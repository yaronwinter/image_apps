import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

def test(model: nn.Module, test_set: DataLoader, device: torch.device) -> float:
    corrects = 0
    evaluated = 0
    model.eval()
    for data in tqdm(test_set):
        images, labels = data
        with torch.no_grad():
            logits, _ = model(images.to(device))
        preds = torch.argmax(logits, dim=1)
        corrects += (preds == labels.to(device)).sum().item()
        evaluated += images.size()[0]
        
    return (corrects / evaluated)

def reconstruct_image(model: nn.Module, image: torch.Tensor, layer: int, channel: int, device: torch.device) -> torch.Tensor:
    model.eval()
    image = image.to(device)
    _, new_image = model(image, layer, channel)
    return new_image
