"""
Unit tests for monitoring module
"""

import pytest
import os
import sys
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.monitoring.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    """Test cases for PerformanceTracker class"""
    
    @pytest.fixture
    def tracker(self):
        """Create a PerformanceTracker instance with temporary storage"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        tracker = PerformanceTracker(storage_path=temp_path)
        yield tracker
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_tracker_initialization(self, tracker):
        """Test that tracker initializes correctly"""
        assert tracker.storage_path.endswith('.json')
        assert 'predictions' in tracker.performance_data
        assert 'metrics' in tracker.performance_data
        assert 'summary' in tracker.performance_data
    
    def test_record_prediction(self, tracker):
        """Test recording a prediction"""
        tracker.record_prediction(
            predicted_class='cat',
            confidence=0.85,
            true_label='cat',
            class_probabilities={'cat': 0.85, 'dog': 0.15},
            processing_time={'total': 0.5}
        )
        
        assert len(tracker.performance_data['predictions']) == 1
        prediction = tracker.performance_data['predictions'][0]
        assert prediction['predicted_class'] == 'cat'
        assert prediction['confidence'] == 0.85
        assert prediction['true_label'] == 'cat'
    
    def test_record_multiple_predictions(self, tracker):
        """Test recording multiple predictions"""
        for i in range(5):
            tracker.record_prediction(
                predicted_class='cat' if i % 2 == 0 else 'dog',
                confidence=0.7 + (i * 0.05),
                true_label='cat' if i % 2 == 0 else 'dog',
                class_probabilities={'cat': 0.7 + (i * 0.05), 'dog': 0.3 - (i * 0.05)}
            )
        
        assert len(tracker.performance_data['predictions']) == 5
    
    def test_calculate_metrics(self, tracker):
        """Test metrics calculation"""
        # Add some labeled predictions
        tracker.record_prediction('cat', 0.9, 'cat', {'cat': 0.9, 'dog': 0.1})
        tracker.record_prediction('dog', 0.8, 'dog', {'cat': 0.2, 'dog': 0.8})
        tracker.record_prediction('cat', 0.7, 'dog', {'cat': 0.7, 'dog': 0.3})
        
        metrics = tracker.calculate_metrics()
        
        assert metrics['total_predictions'] == 3
        assert metrics['labeled_predictions'] == 3
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_calculate_metrics_no_predictions(self, tracker):
        """Test metrics calculation with no predictions"""
        metrics = tracker.calculate_metrics()
        
        assert 'error' in metrics
        assert 'No predictions recorded yet' in metrics['error']
    
    def test_get_summary(self, tracker):
        """Test getting summary"""
        tracker.record_prediction('cat', 0.85, 'cat', {'cat': 0.85, 'dog': 0.15})
        
        summary = tracker.get_summary()
        
        assert 'total_predictions' in summary
        assert summary['total_predictions'] == 1
        assert 'latest_prediction' in summary
        assert 'class_distribution' in summary
    
    def test_class_distribution(self, tracker):
        """Test class distribution calculation"""
        tracker.record_prediction('cat', 0.8, 'cat', {'cat': 0.8, 'dog': 0.2})
        tracker.record_prediction('cat', 0.7, 'cat', {'cat': 0.7, 'dog': 0.3})
        tracker.record_prediction('dog', 0.9, 'dog', {'cat': 0.1, 'dog': 0.9})
        
        summary = tracker.get_summary()
        distribution = summary['class_distribution']
        
        assert distribution['cat'] == 2
        assert distribution['dog'] == 1
    
    def test_export_report(self, tracker):
        """Test exporting performance report"""
        tracker.record_prediction('cat', 0.85, 'cat', {'cat': 0.85, 'dog': 0.15})
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report_path = f.name
        
        try:
            report = tracker.export_report(report_path)
            
            assert report is not None
            assert 'generated_at' in report
            assert 'summary' in report
            assert 'metrics' in report
            assert os.path.exists(report_path)
            
            # Verify file can be read
            with open(report_path, 'r') as f:
                loaded_report = json.load(f)
            assert loaded_report == report
        finally:
            if os.path.exists(report_path):
                os.remove(report_path)
    
    def test_clear_old_data(self, tracker):
        """Test clearing old data"""
        # Add many predictions
        for i in range(15):
            tracker.record_prediction('cat', 0.8, 'cat', {'cat': 0.8, 'dog': 0.2})
        
        assert len(tracker.performance_data['predictions']) == 15
        
        # Clear old data, keeping last 10
        tracker.clear_old_data(keep_last_n=10)
        
        assert len(tracker.performance_data['predictions']) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
