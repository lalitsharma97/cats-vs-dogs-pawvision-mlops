"""
Model Performance Tracking System
Collects and analyzes model performance metrics post-deployment
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

from src.utils.logging import setup_logger

logger = setup_logger("performance_tracker")


class PerformanceTracker:
    """Track and analyze model performance metrics"""
    
    def __init__(self, storage_path: str = "logs/performance_data.json"):
        """
        Initialize performance tracker
        
        Args:
            storage_path: Path to store performance data
        """
        self.storage_path = storage_path
        self.performance_data = {
            "predictions": [],
            "metrics": {},
            "summary": {}
        }
        self._load_data()
    
    def _load_data(self):
        """Load existing performance data from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.performance_data = json.load(f)
                logger.info(f"Loaded existing performance data from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")
    
    def _save_data(self):
        """Save performance data to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            logger.info(f"Saved performance data to {self.storage_path}")
        except Exception as e:
            logger.error(f"Could not save performance data: {e}")
    
    def record_prediction(
        self,
        predicted_class: str,
        confidence: float,
        true_label: Optional[str] = None,
        class_probabilities: Optional[Dict[str, float]] = None,
        processing_time: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record a prediction with optional true label for performance tracking
        
        Args:
            predicted_class: The predicted class
            confidence: Prediction confidence score
            true_label: Optional true label for accuracy calculation
            class_probabilities: Dictionary of class probabilities
            processing_time: Dictionary of timing information
            metadata: Additional metadata about the prediction
        """
        prediction_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "true_label": true_label,
            "class_probabilities": class_probabilities,
            "processing_time": processing_time,
            "metadata": metadata or {}
        }
        
        self.performance_data["predictions"].append(prediction_record)
        self._save_data()
        
        logger.info(f"Recorded prediction: {predicted_class} (confidence: {confidence:.4f})")
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from recorded predictions
        
        Returns:
            Dictionary of performance metrics
        """
        predictions = self.performance_data["predictions"]
        
        if not predictions:
            return {"error": "No predictions recorded yet"}
        
        # Filter predictions with true labels
        labeled_predictions = [p for p in predictions if p.get("true_label")]
        
        metrics = {
            "total_predictions": len(predictions),
            "labeled_predictions": len(labeled_predictions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if labeled_predictions:
            # Calculate accuracy
            correct = sum(1 for p in labeled_predictions if p["predicted_class"] == p["true_label"])
            accuracy = correct / len(labeled_predictions)
            metrics["accuracy"] = accuracy
            
            # Calculate per-class metrics
            class_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
            for p in labeled_predictions:
                true_label = p["true_label"]
                class_metrics[true_label]["total"] += 1
                if p["predicted_class"] == true_label:
                    class_metrics[true_label]["correct"] += 1
            
            metrics["class_accuracy"] = {
                class_name: data["correct"] / data["total"] if data["total"] > 0 else 0
                for class_name, data in class_metrics.items()
            }
            
            # Calculate confidence statistics
            confidences = [p["confidence"] for p in labeled_predictions]
            metrics["confidence_stats"] = {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            }
            
            # Calculate processing time statistics
            processing_times = [p.get("processing_time", {}).get("total", 0) 
                             for p in labeled_predictions if p.get("processing_time")]
            if processing_times:
                metrics["processing_time_stats"] = {
                    "mean": np.mean(processing_times),
                    "std": np.std(processing_times),
                    "min": np.min(processing_times),
                    "max": np.max(processing_times)
                }
        
        self.performance_data["metrics"] = metrics
        self._save_data()
        
        return metrics
    
    def get_summary(self) -> Dict:
        """
        Get a summary of performance data
        
        Returns:
            Summary dictionary
        """
        self.calculate_metrics()
        
        predictions = self.performance_data["predictions"]
        metrics = self.performance_data.get("metrics", {})
        
        summary = {
            "total_predictions": len(predictions),
            "latest_prediction": predictions[-1] if predictions else None,
            "accuracy": metrics.get("accuracy"),
            "class_distribution": self._get_class_distribution(),
            "time_range": self._get_time_range()
        }
        
        self.performance_data["summary"] = summary
        self._save_data()
        
        return summary
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of predicted classes"""
        predictions = self.performance_data["predictions"]
        distribution = defaultdict(int)
        
        for p in predictions:
            distribution[p["predicted_class"]] += 1
        
        return dict(distribution)
    
    def _get_time_range(self) -> Dict:
        """Get time range of predictions"""
        predictions = self.performance_data["predictions"]
        
        if not predictions:
            return {}
        
        timestamps = [p["timestamp"] for p in predictions]
        return {
            "first": min(timestamps),
            "last": max(timestamps),
            "count": len(timestamps)
        }
    
    def export_report(self, output_path: str = "logs/performance_report.json"):
        """
        Export a comprehensive performance report
        
        Args:
            output_path: Path to save the report
        """
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.get_summary(),
            "metrics": self.performance_data.get("metrics", {}),
            "recent_predictions": self.performance_data["predictions"][-10:]  # Last 10 predictions
        }
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report exported to {output_path}")
            return report
        except Exception as e:
            logger.error(f"Could not export performance report: {e}")
            return None
    
    def clear_old_data(self, keep_last_n: int = 1000):
        """
        Clear old prediction data, keeping only the most recent predictions
        
        Args:
            keep_last_n: Number of recent predictions to keep
        """
        if len(self.performance_data["predictions"]) > keep_last_n:
            self.performance_data["predictions"] = self.performance_data["predictions"][-keep_last_n:]
            self._save_data()
            logger.info(f"Cleared old data, keeping last {keep_last_n} predictions")


# Global performance tracker instance
performance_tracker = PerformanceTracker()
