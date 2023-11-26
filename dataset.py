import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *


def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=data_transform)
    test = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=data_transform)

    return torch.utils.data.ConcatDataset([train, test])
