"""
Data loader module for Cats vs Dogs classification
Handles loading processed images and data augmentation
"""

import os
import numpy as np
from typing import Tuple, Generator
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml


class CatsDogsDataset(Dataset):
    """PyTorch Dataset for Cats vs Dogs classification"""
    
    def __init__(self, data_path: str, split: str, transform=None):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to processed data
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to images
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        class_mapping = {'cat': 0, 'dog': 1}
        
        for class_name, label in class_mapping.items():
            class_path = os.path.join(data_path, split, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.npy'):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(label)
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, label)
        """
        # Load preprocessed image
        image = np.load(self.image_paths[idx])
        
        # Convert from numpy to torch tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def get_data_loaders(config_path: str = "configs/model_config.yaml") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = config['training']['batch_size']
    data_path = "data/processed"
    use_augmentation = config['data']['augmentation']
    
    # Define transforms
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    else:
        train_transform = None
    
    # Create datasets
    train_dataset = CatsDogsDataset(data_path, 'train', transform=train_transform)
    val_dataset = CatsDogsDataset(data_path, 'val', transform=None)
    test_dataset = CatsDogsDataset(data_path, 'test', transform=None)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Test the data loader"""
    train_loader, val_loader, test_loader = get_data_loaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        break


if __name__ == "__main__":
    main()
