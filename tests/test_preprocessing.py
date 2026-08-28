"""
Unit tests for data preprocessing functions
"""

import pytest
import numpy as np
import os
import tempfile
from PIL import Image
import sys
sys.path.append('..')

from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessor instance"""
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a temporary test image
        img = Image.new('RGB', (100, 100), color='red')
        return img
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test that preprocessor initializes correctly"""
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.train_split == 0.8
        assert preprocessor.val_split == 0.1
        assert preprocessor.test_split == 0.1
    
    def test_load_and_preprocess_image(self, preprocessor, sample_image):
        """Test image loading and preprocessing"""
        # Save sample image to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            sample_image.save(f.name)
            temp_path = f.name
        
        try:
            # Test preprocessing
            processed = preprocessor.load_and_preprocess_image(temp_path)
            
            # Check output shape
            assert processed.shape == (224, 224, 3)
            
            # Check normalization (values should be between 0 and 1)
            assert processed.min() >= 0.0
            assert processed.max() <= 1.0
            
            # Check data type
            assert processed.dtype == np.float32
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_preprocess_non_rgb_image(self, preprocessor):
        """Test handling of non-RGB images"""
        # Create a grayscale image
        gray_img = Image.new('L', (100, 100), color=128)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            gray_img.save(f.name)
            temp_path = f.name
        
        try:
            processed = preprocessor.load_and_preprocess_image(temp_path)
            
            # Should convert to RGB
            assert processed.shape == (224, 224, 3)
        finally:
            os.unlink(temp_path)
    
    def test_preprocess_invalid_image(self, preprocessor):
        """Test handling of invalid image paths"""
        result = preprocessor.load_and_preprocess_image("nonexistent_image.jpg")
        assert result is None
    
    def test_organize_dataset_structure(self, preprocessor):
        """Test dataset directory structure creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = os.path.join(temp_dir, "raw")
            processed_path = os.path.join(temp_dir, "processed")
            
            preprocessor.organize_dataset_structure(raw_path, processed_path)
            
            # Check that directories were created
            assert os.path.exists(os.path.join(processed_path, "train", "cat"))
            assert os.path.exists(os.path.join(processed_path, "train", "dog"))
            assert os.path.exists(os.path.join(processed_path, "val", "cat"))
            assert os.path.exists(os.path.join(processed_path, "val", "dog"))
            assert os.path.exists(os.path.join(processed_path, "test", "cat"))
            assert os.path.exists(os.path.join(processed_path, "test", "dog"))
    
    def test_get_data_statistics(self, preprocessor):
        """Test dataset statistics calculation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data structure
            processed_path = os.path.join(temp_dir, "processed")
            os.makedirs(os.path.join(processed_path, "train", "cat"), exist_ok=True)
            os.makedirs(os.path.join(processed_path, "train", "dog"), exist_ok=True)
            
            # Create dummy files
            np.save(os.path.join(processed_path, "train", "cat", "test1.npy"), np.zeros((224, 224, 3)))
            np.save(os.path.join(processed_path, "train", "cat", "test2.npy"), np.zeros((224, 224, 3)))
            np.save(os.path.join(processed_path, "train", "dog", "test1.npy"), np.zeros((224, 224, 3)))
            
            stats = preprocessor.get_data_statistics(processed_path)
            
            assert stats['train']['cat'] == 2
            assert stats['train']['dog'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
