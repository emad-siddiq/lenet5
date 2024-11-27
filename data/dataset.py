# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_data(batch_size=32, train_val_split=0.9):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # LeNet-5 expects 32x32 images
        transforms.ToTensor(),
    ])
    
    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='.', 
                                 train=True,
                                 transform=transform,
                                 download=True)
    
    test_dataset = datasets.MNIST(root='.', 
                                train=False,
                                transform=transform,
                                download=True)
    
    # Split training data into train and validation sets
    train_size = int(len(train_dataset) * train_val_split)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size,
                            shuffle=True)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False)
    
    test_loader = DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False)
    
    return train_loader, val_loader, test_loader