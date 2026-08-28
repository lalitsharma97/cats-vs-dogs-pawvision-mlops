"""
Unit tests for inference functions
"""

import pytest
import torch
import numpy as np
import sys
sys.path.append('..')

from src.models.architecture import get_model


class TestModelInference:
    """Test cases for model inference"""
    
    @pytest.fixture
    def model(self):
        """Create a model instance"""
        return get_model("cnn")
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return torch.randn(2, 3, 224, 224)
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly"""
        assert model is not None
        info = model.get_model_info()
        assert info['model_name'] == 'BaselineCNN'
        assert info['total_parameters'] > 0
    
    def test_model_forward_pass(self, model, sample_input):
        """Test forward pass through the model"""
        output = model(sample_input)
        
        # Check output shape
        assert output.shape == (2, 2)  # batch_size, num_classes
        
        # Check output type
        assert isinstance(output, torch.Tensor)
    
    def test_model_output_range(self, model, sample_input):
        """Test that model outputs are reasonable"""
        output = model(sample_input)
        
        # Outputs should be finite
        assert torch.all(torch.isfinite(output))
    
    def test_model_with_different_batch_sizes(self, model):
        """Test model with different batch sizes"""
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            output = model(input_tensor)
            assert output.shape == (batch_size, 2)
    
    def test_model_prediction_classes(self, model, sample_input):
        """Test prediction class extraction"""
        output = model(sample_input)
        predictions = torch.argmax(output, dim=1)
        
        # Predictions should be valid class indices (0 or 1)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_softmax_probabilities(self, model, sample_input):
        """Test softmax probability calculation"""
        output = model(sample_input)
        probabilities = torch.softmax(output, dim=1)
        
        # Probabilities should sum to 1 for each sample
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
        
        # Probabilities should be between 0 and 1
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
    
    def test_logistic_regression_model(self):
        """Test logistic regression baseline model"""
        model = get_model("logistic")
        sample_input = torch.randn(2, 3, 224, 224)
        
        output = model(sample_input)
        assert output.shape == (2, 2)
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type"""
        with pytest.raises(ValueError):
            get_model("invalid_model_type")


class TestModelUtilities:
    """Test cases for model utility functions"""
    
    def test_model_info_extraction(self):
        """Test model information extraction"""
        model = get_model("cnn")
        info = model.get_model_info()
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_name' in info
        assert isinstance(info['total_parameters'], int)
        assert isinstance(info['trainable_parameters'], int)
    
    def test_model_device_compatibility(self):
        """Test model works on different devices"""
        model = get_model("cnn")
        sample_input = torch.randn(2, 3, 224, 224)
        
        # Test CPU
        model_cpu = model.to('cpu')
        input_cpu = sample_input.to('cpu')
        output_cpu = model_cpu(input_cpu)
        assert output_cpu.shape == (2, 2)
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            input_cuda = sample_input.to('cuda')
            output_cuda = model_cuda(input_cuda)
            assert output_cuda.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
