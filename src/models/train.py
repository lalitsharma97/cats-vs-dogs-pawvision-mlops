"""
Training script for Cats vs Dogs classification
Includes MLflow experiment tracking and model saving
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

from src.models.architecture import get_model
from src.data.data_loader import get_data_loaders
from src.monitoring.mlflow_tracker import MLflowTracker


class Trainer:
    """Trainer class for model training with MLflow tracking"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model("cnn", config_path).to(self.device)
        
        # Print model info
        model_info = self.model.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Total parameters: {model_info['total_parameters']:,}")
        
        # Training parameters
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']
        self.batch_size = self.config['training']['batch_size']
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # MLflow setup
        self.mlflow_tracker = MLflowTracker(experiment_name="cats_vs_dogs_classification")
        
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy, all_labels, all_predictions)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_labels, all_predictions
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop with MLflow tracking
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        # Start MLflow run
        self.mlflow_tracker.start_run()
        
        # Log initial parameters
        self.mlflow_tracker.log_params({
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer": "adam",
            "loss_function": "cross_entropy"
        })
        
        # Log model info
        model_info = self.model.get_model_info()
        self.mlflow_tracker.log_params({
            "model_name": model_info['model_name'],
            "total_parameters": model_info['total_parameters']
        })
        
        # Log dataset info
        dataset_info = {
            "total": len(train_loader.dataset) + len(val_loader.dataset),
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": 0,
            "num_classes": 2
        }
        self.mlflow_tracker.log_dataset_info(dataset_info)
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_accuracy = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            
            # Validate
            val_loss, val_accuracy, val_labels, val_predictions = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            # Log metrics
            self.mlflow_tracker.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model("best_model")
                self.mlflow_tracker.log_metrics({"best_val_accuracy": best_val_accuracy})
            
            # Log confusion matrix at end of training
            if epoch == self.epochs - 1:
                self.mlflow_tracker.log_confusion_matrix(
                    val_labels, val_predictions, 
                    class_names=['cat', 'dog']
                )
        
        print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Log the best model
        self.mlflow_tracker.log_model(self.model, "model")
        
        # Log artifacts
        self.mlflow_tracker.log_artifact("configs/model_config.yaml")
        self.mlflow_tracker.log_artifact("requirements.txt")
        
        # Save final model
        self.save_model("final_model")
        
        # End MLflow run
        self.mlflow_tracker.end_run()
    
    def save_model(self, model_name: str):
        """
        Save model to disk
        
        Args:
            model_name: Name for the saved model
        """
        os.makedirs("models/saved_models", exist_ok=True)
        
        # Save as PyTorch model
        model_path = f"models/saved_models/{model_name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, model_path)
        
        print(f"Model saved to {model_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs classifier')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args.config)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
