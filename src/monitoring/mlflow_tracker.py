"""
MLflow experiment tracking module
Automatically logs runs, parameters, metrics, and artifacts during training
"""

import os
import yaml
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path


class MLflowTracker:
    """MLflow experiment tracking class"""
    
    def __init__(self, experiment_name: str = "cats_vs_dogs_classification"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.run = None
        self.run_name = None
        
        # Set MLflow backend store
        mlflow.set_tracking_uri(f"file://{os.path.join(os.getcwd(), 'mlruns')}")
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: str = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Custom run name (auto-generated if None)
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.run = mlflow.start_run(run_name=run_name)
        print(f"Started MLflow run: {run_name}")
        return self.run
    
    def log_params(self, params: dict):
        """
        Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.run:
            mlflow.log_params(params)
            print(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (for time-series metrics)
        """
        if self.run:
            mlflow.log_metrics(metrics, step=step)
            print(f"Logged {len(metrics)} metrics at step {step}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """
        Log a PyTorch model to MLflow
        
        Args:
            model: PyTorch model to log
            artifact_path: Path for the model artifact
        """
        if self.run:
            mlflow.pytorch.log_model(model, artifact_path)
            print(f"Logged model to {artifact_path}")
    
    def log_artifact(self, file_path: str):
        """
        Log an artifact file to MLflow
        
        Args:
            file_path: Path to the artifact file
        """
        if self.run and os.path.exists(file_path):
            mlflow.log_artifact(file_path)
            print(f"Logged artifact: {file_path}")
    
    def log_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Log confusion matrix and classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        if self.run:
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:")
            print(cm)
            
            # Compute classification report
            if class_names is None:
                class_names = [f"class_{i}" for i in range(len(np.unique(y_true)))]
            
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            print("Classification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))
            
            # Log metrics from classification report
            metrics_to_log = {}
            for class_name in class_names:
                if class_name in report:
                    metrics_to_log[f"precision_{class_name}"] = report[class_name]['precision']
                    metrics_to_log[f"recall_{class_name}"] = report[class_name]['recall']
                    metrics_to_log[f"f1_{class_name}"] = report[class_name]['f1-score']
            
            self.log_metrics(metrics_to_log)
    
    def log_training_summary(self, config: dict, model_info: dict, best_val_accuracy: float):
        """
        Log training summary including config and final metrics
        
        Args:
            config: Training configuration
            model_info: Model information
            best_val_accuracy: Best validation accuracy achieved
        """
        if self.run:
            # Log configuration parameters
            training_params = {
                "learning_rate": config.get('training', {}).get('learning_rate', 0.001),
                "batch_size": config.get('training', {}).get('batch_size', 16),
                "epochs": config.get('training', {}).get('epochs', 5),
                "optimizer": config.get('training', {}).get('optimizer', 'adam'),
                "loss_function": config.get('training', {}).get('loss_function', 'cross_entropy')
            }
            self.log_params(training_params)
            
            # Log model parameters
            model_params = {
                "model_name": model_info.get('model_name', 'unknown'),
                "total_parameters": model_info.get('total_parameters', 0)
            }
            self.log_params(model_params)
            
            # Log final metrics
            self.log_metrics({"best_val_accuracy": best_val_accuracy})
    
    def log_dataset_info(self, dataset_info: dict):
        """
        Log dataset information
        
        Args:
            dataset_info: Dictionary with dataset statistics
        """
        if self.run:
            self.log_params({
                "total_images": dataset_info.get('total', 0),
                "train_images": dataset_info.get('train', 0),
                "val_images": dataset_info.get('val', 0),
                "test_images": dataset_info.get('test', 0),
                "num_classes": dataset_info.get('num_classes', 2)
            })
    
    def end_run(self):
        """End the current MLflow run"""
        if self.run:
            mlflow.end_run()
            print(f"Ended MLflow run: {self.run_name}")
            self.run = None
            self.run_name = None


def setup_mlflow_tracking(config_path: str = "configs/model_config.yaml") -> MLflowTracker:
    """
    Setup MLflow tracking with configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MLflowTracker instance
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create tracker
    tracker = MLflowTracker(experiment_name="cats_vs_dogs_classification")
    
    return tracker, config