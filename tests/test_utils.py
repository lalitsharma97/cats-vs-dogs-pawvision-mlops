"""
Unit tests for utility functions
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from src.utils.metrics import (
    calculate_metrics, calculate_class_specific_metrics,
    get_confusion_matrix, print_metrics_report
)


class TestMetricsCalculation:
    """Test cases for metrics calculation"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.6, 0.9, 0.1, 0.8])
        return y_true, y_pred, y_prob
    
    def test_calculate_metrics(self, sample_predictions):
        """Test metrics calculation"""
        y_true, y_pred, y_prob = sample_predictions
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # Check that all expected metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['auc'] <= 1
    
    def test_calculate_metrics_without_probabilities(self, sample_predictions):
        """Test metrics calculation without probabilities"""
        y_true, y_pred, _ = sample_predictions
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should still calculate basic metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_calculate_class_specific_metrics(self, sample_predictions):
        """Test class-specific metrics calculation"""
        y_true, y_pred, _ = sample_predictions
        class_metrics = calculate_class_specific_metrics(y_true, y_pred)
        
        # Check structure
        assert 'cat' in class_metrics
        assert 'dog' in class_metrics
        
        # Check each class has required metrics
        for class_name in ['cat', 'dog']:
            assert 'precision' in class_metrics[class_name]
            assert 'recall' in class_metrics[class_name]
            assert 'f1_score' in class_metrics[class_name]
            assert 'support' in class_metrics[class_name]
    
    def test_get_confusion_matrix(self, sample_predictions):
        """Test confusion matrix generation"""
        y_true, y_pred, _ = sample_predictions
        cm = get_confusion_matrix(y_true, y_pred)
        
        # Check shape (2x2 for binary classification)
        assert cm.shape == (2, 2)
        
        # Check that all values are non-negative
        assert np.all(cm >= 0)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # Perfect predictions should give perfect scores
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_worst_predictions(self):
        """Test metrics with completely wrong predictions"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # Completely wrong predictions should give zero scores
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
    
    def test_metrics_with_imbalanced_data(self):
        """Test metrics with imbalanced classes"""
        y_true = np.array([0, 0, 0, 0, 1])  # 4 cats, 1 dog
        y_pred = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.1, 0.6, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        class_metrics = calculate_class_specific_metrics(y_true, y_pred)
        
        # Check that metrics are calculated correctly despite imbalance
        assert 0 <= metrics['accuracy'] <= 1
        assert class_metrics['cat']['support'] == 4
        assert class_metrics['dog']['support'] == 1


class TestMetricsEdgeCases:
    """Test edge cases for metrics calculation"""
    
    def test_empty_predictions(self):
        """Test with empty arrays"""
        y_true = np.array([])
        y_pred = np.array([])
        
        # This should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate_metrics(y_true, y_pred)
    
    def test_single_class_predictions(self):
        """Test with predictions from only one class"""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        
        metrics = calculate_metrics(y_true, y_pred)
        # Should handle single class case
        assert metrics['accuracy'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
