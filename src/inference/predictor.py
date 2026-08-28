"""
Model predictor module for Cats vs Dogs classification
Handles model loading and inference logic
"""

import os
import io
import logging
import yaml
import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional

from src.models.architecture import get_model
from src.utils.logging import setup_logger

logger = setup_logger("model_predictor")


class ModelPredictor:
    """Handles model loading and predictions"""
    
    def __init__(self, model_path: str = "models/saved_models/best_model.pt",
                 config_path: str = "configs/model_config.yaml"):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model file
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.device = None
        self.class_names = ['cat', 'dog']
        self.config = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Load configuration
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Initialize model
            self.model = get_model("cnn", self.config_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load trained weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}, using untrained model")
            
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed tensor ready for model
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 224x224
            image = image.resize((224, 224), Image.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        """
        Make prediction on image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_bytes)
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            # Get class name and probabilities
            predicted_class = self.class_names[predicted_class_idx]
            class_probabilities = {
                self.class_names[i]: probabilities[0][i].item() 
                for i in range(len(self.class_names))
            }
            
            # Log prediction
            logger.info(
                f"Prediction: {predicted_class} (confidence: {confidence:.4f}), "
                f"probabilities: {class_probabilities}"
            )
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "predicted_class_idx": predicted_class_idx
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, image_bytes_list: list) -> list:
        """
        Make predictions on multiple images
        
        Args:
            image_bytes_list: List of raw image bytes
            
        Returns:
            List of prediction results
        """
        results = []
        for image_bytes in image_bytes_list:
            try:
                result = self.predict(image_bytes)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting image: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        model_info = self.model.get_model_info()
        return {
            "model_path": self.model_path,
            "config_path": self.config_path,
            "device": str(self.device),
            "class_names": self.class_names,
            "model_info": model_info
        }


# Singleton predictor instance
_predictor_instance = None


def get_predictor() -> ModelPredictor:
    """
    Get singleton predictor instance
    
    Returns:
        ModelPredictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ModelPredictor()
    return _predictor_instance


if __name__ == "__main__":
    # Test the predictor
    predictor = ModelPredictor()
    
    # Print model info
    info = predictor.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test with a dummy image (if available)
    print("\nPredictor is ready for inference.")
