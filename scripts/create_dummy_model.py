"""
Script to create a dummy model for testing the inference service
This creates a minimal trained model for testing purposes
"""

import os
import sys
import torch
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.architecture import get_model

def create_dummy_model():
    """Create a dummy trained model for testing"""
    
    # Ensure directory exists
    os.makedirs("models/saved_models", exist_ok=True)
    
    # Initialize model
    model = get_model("cnn")
    
    # Load config
    with open("configs/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},  # Empty for dummy
        'config': config,
        'epoch': 0,
        'val_accuracy': 0.0
    }
    
    # Save dummy model
    model_path = "models/saved_models/best_model.pt"
    torch.save(checkpoint, model_path)
    
    print(f"Dummy model created at {model_path}")
    print("Note: This is an untrained model for testing purposes only.")
    print("For production use, train the model using: python src/models/train.py")

if __name__ == "__main__":
    create_dummy_model()
