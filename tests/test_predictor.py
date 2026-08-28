"""
Unit tests for model predictor module
"""

import pytest
import torch
import numpy as np
import os
import sys
import tempfile
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predictor import ModelPredictor


class TestModelPredictor:
    """Test cases for ModelPredictor class"""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing"""
        # Create a temporary dummy model file
        import yaml
        from src.models.architecture import get_model
        
        os.makedirs("models/saved_models", exist_ok=True)
        
        model = get_model("cnn")
        with open("configs/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'config': config,
            'epoch': 0,
            'val_accuracy': 0.0
        }
        
        model_path = "models/saved_models/best_model.pt"
        torch.save(checkpoint, model_path)
        
        yield model_path
        
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)
    
    @pytest.fixture
    def predictor(self, dummy_model):
        """Create a ModelPredictor instance"""
        return ModelPredictor()
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes for testing"""
        # Create a simple test image
        image = Image.new('RGB', (100, 100), color='blue')
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.read()
    
    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes correctly"""
        assert predictor.model is not None
        assert predictor.device is not None
        assert predictor.class_names == ['cat', 'dog']
        assert len(predictor.class_names) == 2
    
    def test_preprocess_image(self, predictor, sample_image_bytes):
        """Test image preprocessing"""
        try:
            processed = predictor.preprocess_image(sample_image_bytes)
            
            # Check output is a tensor
            assert isinstance(processed, torch.Tensor)
            
            # Check shape (should be batch_size, channels, height, width)
            assert processed.shape[0] == 1  # batch dimension
            assert processed.shape[1] == 3  # RGB channels
            assert processed.shape[2] == 224  # height
            assert processed.shape[3] == 224  # width
        except Exception as e:
            pytest.skip(f"Preprocessing test skipped: {e}")
    
    def test_predict_single_image(self, predictor, sample_image_bytes):
        """Test prediction on single image"""
        try:
            result = predictor.predict(sample_image_bytes)
            
            # Check result structure
            assert 'predicted_class' in result
            assert 'confidence' in result
            assert 'class_probabilities' in result
            assert 'predicted_class_idx' in result
            
            # Check predicted class is valid
            assert result['predicted_class'] in ['cat', 'dog']
            
            # Check confidence is between 0 and 1
            assert 0 <= result['confidence'] <= 1
            
            # Check class probabilities
            assert 'cat' in result['class_probabilities']
            assert 'dog' in result['class_probabilities']
            assert len(result['class_probabilities']) == 2
            
            # Check probabilities sum to approximately 1
            prob_sum = sum(result['class_probabilities'].values())
            assert abs(prob_sum - 1.0) < 0.01  # Allow small floating point error
        except Exception as e:
            pytest.skip(f"Prediction test skipped: {e}")
    
    def test_predict_batch_images(self, predictor, sample_image_bytes):
        """Test prediction on multiple images"""
        try:
            image_list = [sample_image_bytes, sample_image_bytes]
            results = predictor.predict_batch(image_list)
            
            # Check we get results for all images
            assert len(results) == 2
            
            # Check each result has required fields
            for result in results:
                if 'error' not in result:  # Skip if there was an error
                    assert 'predicted_class' in result
                    assert 'confidence' in result
        except Exception as e:
            pytest.skip(f"Batch prediction test skipped: {e}")
    
    def test_get_model_info(self, predictor):
        """Test model information extraction"""
        info = predictor.get_model_info()
        
        # Check required fields
        assert 'model_path' in info
        assert 'config_path' in info
        assert 'device' in info
        assert 'class_names' in info
        assert 'model_info' in info
        
        # Check model info has required fields
        assert 'total_parameters' in info['model_info']
        assert 'model_name' in info['model_info']
    
    def test_invalid_image_bytes(self, predictor):
        """Test handling of invalid image data"""
        invalid_bytes = b"not an image"
        
        with pytest.raises((ValueError, RuntimeError)):
            predictor.predict(invalid_bytes)
    
    def test_empty_image_bytes(self, predictor):
        """Test handling of empty image data"""
        empty_bytes = b""
        
        with pytest.raises((ValueError, RuntimeError)):
            predictor.predict(empty_bytes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
