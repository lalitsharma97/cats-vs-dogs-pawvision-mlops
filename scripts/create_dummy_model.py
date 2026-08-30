"""
Create dummy model for testing purposes
"""

import os
import torch
import torch.nn as nn
from pathlib import Path

def create_dummy_model():
    """Create a dummy PyTorch model for testing"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("models/saved_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 56 * 56, 128)
            self.fc2 = nn.Linear(128, 2)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 32 * 56 * 56)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create and save the model
    model = DummyModel()
    model_path = models_dir / "best_model.pt"
    
    # Save with proper structure
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': 'dummy_cnn',
        'total_parameters': sum(p.numel() for p in model.parameters())
    }, model_path)
    
    print(f"Dummy model created and saved to {model_path}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    create_dummy_model()