import torchvision
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import util.trainer as trainer

def main():
    model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
    train_loader = trainer.DirDataset()