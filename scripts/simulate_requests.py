"""
Simulated Request Collection for Performance Tracking
Generates simulated requests with true labels to track model performance
"""

import os
import time
import random
import logging
from typing import List, Dict
from PIL import Image
import numpy as np
import requests
from datetime import datetime

from src.utils.logging import setup_logger
from src.monitoring.performance_tracker import performance_tracker

logger = setup_logger("simulate_requests")


class RequestSimulator:
    """Simulate requests for performance tracking"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize request simulator
        
        Args:
            api_url: Base URL of the inference API
        """
        self.api_url = api_url
        self.simulated_labels = {
            "cat": ["cat_001.jpg", "cat_002.jpg", "cat_003.jpg"],
            "dog": ["dog_001.jpg", "dog_002.jpg", "dog_003.jpg"]
        }
    
    def create_test_images(self, num_images: int = 10) -> List[str]:
        """
        Create test images for simulation
        
        Args:
            num_images: Number of test images to create
            
        Returns:
            List of image file paths
        """
        image_paths = []
        
        for i in range(num_images):
            # Randomly choose class
            true_class = random.choice(["cat", "dog"])
            filename = f"simulated_{true_class}_{i+1:03d}.jpg"
            filepath = os.path.join("data/raw", filename)
            
            # Create a simple test image
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create image with different colors for different classes
            if true_class == "cat":
                color = (255, 165, 0)  # Orange for cats
            else:
                color = (0, 0, 255)    # Blue for dogs
            
            image = Image.new('RGB', (224, 224), color)
            image.save(filepath)
            
            image_paths.append({
                "path": filepath,
                "true_label": true_class,
                "filename": filename
            })
        
        logger.info(f"Created {num_images} test images")
        return image_paths
    
    def send_prediction_request(self, image_path: str, true_label: str) -> Dict:
        """
        Send a prediction request to the API
        
        Args:
            image_path: Path to the image file
            true_label: True label for the image
            
        Returns:
            Response data from the API
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.api_url}/predict", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # Record prediction with true label for performance tracking
                performance_tracker.record_prediction(
                    predicted_class=result['predicted_class'],
                    confidence=result['confidence'],
                    true_label=true_label,
                    class_probabilities=result['class_probabilities'],
                    processing_time=result.get('processing_time'),
                    metadata={
                        "filename": result['filename'],
                        "simulated": True,
                        "true_label": true_label
                    }
                )
                
                logger.info(f"Request sent: {true_label} -> {result['predicted_class']} (confidence: {result['confidence']:.4f})")
                return result
            else:
                logger.error(f"Request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending prediction request: {e}")
            return None
    
    def run_simulation(self, num_requests: int = 20, delay: float = 1.0) -> Dict:
        """
        Run a simulation of prediction requests
        
        Args:
            num_requests: Number of requests to simulate
            delay: Delay between requests in seconds
            
        Returns:
            Simulation results summary
        """
        logger.info(f"Starting simulation with {num_requests} requests")
        
        # Create test images
        test_images = self.create_test_images(num_requests)
        
        results = {
            "total_requests": num_requests,
            "successful_requests": 0,
            "failed_requests": 0,
            "correct_predictions": 0,
            "start_time": datetime.utcnow().isoformat(),
            "requests": []
        }
        
        # Send requests
        for i, image_info in enumerate(test_images):
            logger.info(f"Processing request {i+1}/{num_requests}")
            
            result = self.send_prediction_request(image_info["path"], image_info["true_label"])
            
            if result:
                results["successful_requests"] += 1
                
                # Check if prediction was correct
                if result['predicted_class'] == image_info["true_label"]:
                    results["correct_predictions"] += 1
                
                results["requests"].append({
                    "true_label": image_info["true_label"],
                    "predicted_class": result['predicted_class'],
                    "confidence": result['confidence'],
                    "correct": result['predicted_class'] == image_info["true_label"]
                })
            else:
                results["failed_requests"] += 1
            
            # Delay between requests
            if delay > 0 and i < len(test_images) - 1:
                time.sleep(delay)
        
        results["end_time"] = datetime.utcnow().isoformat()
        results["accuracy"] = results["correct_predictions"] / results["successful_requests"] if results["successful_requests"] > 0 else 0
        
        logger.info(f"Simulation completed: {results['successful_requests']}/{num_requests} successful, accuracy: {results['accuracy']:.2%}")
        
        return results
    
    def generate_performance_report(self) -> Dict:
        """
        Generate a performance report from the simulation
        
        Returns:
            Performance report
        """
        logger.info("Generating performance report")
        
        # Calculate metrics from performance tracker
        metrics = performance_tracker.calculate_metrics()
        summary = performance_tracker.get_summary()
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "simulation_summary": summary,
            "detailed_metrics": metrics,
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """
        Generate recommendations based on performance metrics
        
        Args:
            metrics: Performance metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            if accuracy < 0.7:
                recommendations.append("Model accuracy is below 70%. Consider retraining with more data.")
            elif accuracy < 0.85:
                recommendations.append("Model accuracy is moderate. Consider fine-tuning hyperparameters.")
            else:
                recommendations.append("Model accuracy is good. Continue monitoring performance.")
        
        if "processing_time_stats" in metrics:
            avg_time = metrics["processing_time_stats"]["mean"]
            if avg_time > 1.0:
                recommendations.append("Average processing time is high. Consider model optimization.")
        
        if "confidence_stats" in metrics:
            avg_confidence = metrics["confidence_stats"]["mean"]
            if avg_confidence < 0.7:
                recommendations.append("Average confidence is low. Model may be uncertain about predictions.")
        
        if not recommendations:
            recommendations.append("Model performance is within acceptable ranges.")
        
        return recommendations


def main():
    """Main function to run the simulation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate requests for performance tracking")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests to simulate")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = RequestSimulator(api_url=args.url)
    
    # Run simulation
    results = simulator.run_simulation(num_requests=args.requests, delay=args.delay)
    
    # Generate performance report
    report = simulator.generate_performance_report()
    
    # Print summary
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(f"Total Requests: {results['total_requests']}")
    print(f"Successful: {results['successful_requests']}")
    print(f"Failed: {results['failed_requests']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Export performance report
    performance_tracker.export_report("logs/simulation_report.json")
    print("\nPerformance report exported to logs/simulation_report.json")


if __name__ == "__main__":
    main()
