"""
Metrics calculation utilities for model evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0
    
    return metrics


def calculate_class_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    class_names: List[str] = ['cat', 'dog']) -> Dict[str, Dict[str, float]]:
    """
    Calculate class-specific metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary with class-specific metrics
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    class_metrics = {}
    for class_name in class_names:
        class_metrics[class_name] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1_score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
    
    return class_metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def print_metrics_report(metrics: Dict[str, float], class_metrics: Dict[str, Dict[str, float]]):
    """
    Print a formatted metrics report
    
    Args:
        metrics: Overall metrics dictionary
        class_metrics: Class-specific metrics dictionary
    """
    print("\n" + "="*50)
    print("OVERALL METRICS")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n" + "="*50)
    print("CLASS-SPECIFIC METRICS")
    print("="*50)
    for class_name, metrics_dict in class_metrics.items():
        print(f"\n{class_name.upper()}:")
        for metric_name, value in metrics_dict.items():
            if metric_name != 'support':
                print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  Support: {value}")


def main():
    """Test metrics calculation"""
    # Sample data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.6, 0.9, 0.1, 0.8])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    class_metrics = calculate_class_specific_metrics(y_true, y_pred)
    cm = get_confusion_matrix(y_true, y_pred)
    
    # Print results
    print_metrics_report(metrics, class_metrics)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
