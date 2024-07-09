import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

def test(model: nn.Module, test_set: DataLoader, device: torch.device) -> float:
    corrects = 0
    evaluated = 0
    model.eval()
    for data in test_set:
        images, labels = data
        with torch.no_grad():
            logits = model(images.to(device))
        preds = torch.argmax(logits, dim=1)
        corrects += (preds == labels).sum().item()
        evaluated += images.size()[0]
        
    return (corrects / evaluated)
