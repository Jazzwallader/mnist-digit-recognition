import torch
import struct
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define the transformation for training (data augmentation + normalization)
train_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert the tensor image to a PIL image for augmentation
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),    # Convert back to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST stats
])

# For testing, you generally only want to convert to PIL, then to tensor and normalize
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

MNIST_PATH = "./dataset/mnist/"

def load_idx_file(file_path):
    """Loads an IDX file format (MNIST dataset format)."""
    with open(file_path, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
        elif magic == 2049:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic number {magic} in file {file_path}")
    return data

# Load dataset
train_images = load_idx_file(MNIST_PATH + "train-images-idx3-ubyte")
train_labels = load_idx_file(MNIST_PATH + "train-labels-idx1-ubyte")
test_images = load_idx_file(MNIST_PATH + "t10k-images-idx3-ubyte")
test_labels = load_idx_file(MNIST_PATH + "t10k-labels-idx1-ubyte")

class MNISTDataset(Dataset):
    """Custom PyTorch dataset for MNIST."""
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0  # Normalize
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(batch_size=64):
    """Creates DataLoaders for training and testing."""
    train_dataset = MNISTDataset(train_images, train_labels, transform=train_transform)
    test_dataset = MNISTDataset(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader