"""
CNN model architecture for Cats vs Dogs binary classification
Implements a baseline CNN model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class BaselineCNN(nn.Module):
    """Baseline CNN for binary image classification"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize the CNN model
        
        Args:
            config_path: Path to configuration file
        """
        super(BaselineCNN, self).__init__()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        input_size = config['model']['input_size']  # [224, 224, 3]
        num_classes = config['model']['num_classes']
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_name': 'BaselineCNN'
        }


class SimpleLogisticRegression(nn.Module):
    """Simple logistic regression baseline on flattened pixels"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize logistic regression model
        
        Args:
            config_path: Path to configuration file
        """
        super(SimpleLogisticRegression, self).__init__()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        input_size = config['model']['input_size']
        num_classes = config['model']['num_classes']
        
        # Flatten input: 224 * 224 * 3
        flattened_size = input_size[0] * input_size[1] * input_size[2]
        
        self.fc = nn.Linear(flattened_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_model(model_type: str = "cnn", config_path: str = "configs/model_config.yaml"):
    """
    Factory function to get model
    
    Args:
        model_type: Type of model ('cnn' or 'logistic')
        config_path: Path to configuration file
        
    Returns:
        Model instance
    """
    if model_type == "cnn":
        return BaselineCNN(config_path)
    elif model_type == "logistic":
        return SimpleLogisticRegression(config_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test the model"""
    model = get_model("cnn")
    
    # Print model info
    info = model.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
