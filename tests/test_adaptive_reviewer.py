"""
Unit tests for AdaptiveReviewer feature importance functionality.
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Import the class to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.adaptive_reviewer import AdaptiveReviewer


class TestFeatureImportance:
    """Test cases for feature importance calculation functionality."""
    
    @pytest.fixture
    def adaptive_reviewer(self):
        """Create AdaptiveReviewer instance for testing."""
        return AdaptiveReviewer()
    
    @pytest.fixture
    def mock_random_forest(self):
        """Create a mock RandomForest model with feature importances."""
        model = Mock(spec=RandomForestClassifier)
        model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])
        return model
    
    @pytest.fixture
    def mock_gradient_boosting(self):
        """Create a mock GradientBoosting model with feature importances."""
        model = Mock(spec=GradientBoostingClassifier)
        model.feature_importances_ = np.array([0.25, 0.35, 0.15, 0.25])
        return model
    
    @pytest.fixture
    def mock_non_tree_model(self):
        """Create a mock non-tree-based model."""
        model = Mock(spec=LogisticRegression)
        # LogisticRegression doesn't have feature_importances_
        return model
    
    @pytest.fixture
    def sample_feature_names(self):
        """Sample feature names for testing."""
        return ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    
    def test_calculate_feature_importance_success(self, adaptive_reviewer, mock_random_forest, sample_feature_names):
        """Test successful feature importance calculation."""
        # Setup
        adaptive_reviewer.models = {'random_forest': mock_random_forest}
        
        # Execute
        result = adaptive_reviewer.calculate_feature_importance('random_forest', sample_feature_names)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 4
        assert result['feature_3'] == 0.4  # Highest importance
        assert result['feature_1'] == 0.3
        assert result['feature_2'] == 0.2
        assert result['feature_4'] == 0.1  # Lowest importance
        
        # Verify descending order
        importance_values = list(result.values())
        assert importance_values == sorted(importance_values, reverse=True)
    
    def test_calculate_feature_importance_invalid_model(self, adaptive_reviewer, sample_feature_names):
        """Test error handling for invalid model name."""
        # Setup
        adaptive_reviewer.models = {'random_forest': Mock()}
        
        # Execute and assert
        with pytest.raises(KeyError) as exc_info:
            adaptive_reviewer.calculate_feature_importance('invalid_model', sample_feature_names)
        
        assert "Model 'invalid_model' not found" in str(exc_info.value)
        assert "Available models: ['random_forest']" in str(exc_info.value)
    
    def test_calculate_feature_importance_mismatched_features(self, adaptive_reviewer, mock_random_forest):
        """Test error handling for mismatched feature names."""
        # Setup
        adaptive_reviewer.models = {'random_forest': mock_random_forest}
        mismatched_features = ['feature_1', 'feature_2']  # Only 2 features, model has 4
        
        # Execute and assert
        with pytest.raises(ValueError) as exc_info:
            adaptive_reviewer.calculate_feature_importance('random_forest', mismatched_features)
        
        assert "Feature names length (2) doesn't match model features (4)" in str(exc_info.value)
    
    def test_calculate_feature_importance_non_tree_model(self, adaptive_reviewer, mock_non_tree_model, sample_feature_names):
        """Test error handling for non-tree-based models."""
        # Setup
        adaptive_reviewer.models = {'logistic_regression': mock_non_tree_model}
        
        # Execute and assert
        with pytest.raises(ValueError) as exc_info:
            adaptive_reviewer.calculate_feature_importance('logistic_regression', sample_feature_names)
        
        assert "Model 'logistic_regression' is not a tree-based model" in str(exc_info.value)
    
    def test_calculate_feature_importance_gradient_boosting(self, adaptive_reviewer, mock_gradient_boosting, sample_feature_names):
        """Test feature importance calculation with GradientBoosting model."""
        # Setup
        adaptive_reviewer.models = {'gradient_boosting': mock_gradient_boosting}
        
        # Execute
        result = adaptive_reviewer.calculate_feature_importance('gradient_boosting', sample_feature_names)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 4
        assert result['feature_2'] == 0.35  # Highest importance
        assert result['feature_1'] == 0.25
        assert result['feature_4'] == 0.25
        assert result['feature_3'] == 0.15  # Lowest importance
    
    def test_get_feature_importance_json_success(self, adaptive_reviewer, mock_random_forest, sample_feature_names):
        """Test successful JSON feature importance generation."""
        # Setup
        adaptive_reviewer.models = {'random_forest': mock_random_forest}
        
        # Execute
        result_json = adaptive_reviewer.get_feature_importance_json('random_forest', sample_feature_names)
        
        # Parse JSON and assert
        result = json.loads(result_json)
        
        assert result['model_name'] == 'random_forest'
        assert result['total_features'] == 4
        assert 'timestamp' in result
        assert 'feature_importance' in result
        assert 'top_features' in result
        assert 'importance_summary' in result
        
        # Check feature importance data
        importance = result['feature_importance']
        assert importance['feature_3'] == 0.4
        assert importance['feature_1'] == 0.3
        
        # Check top features
        assert result['top_features'] == ['feature_3', 'feature_1', 'feature_2', 'feature_4']
        
        # Check summary statistics
        summary = result['importance_summary']
        assert summary['max_importance'] == 0.4
        assert summary['min_importance'] == 0.1
        assert summary['mean_importance'] == 0.25
        assert 'std_importance' in summary
    
    def test_get_feature_importance_json_invalid_model(self, adaptive_reviewer, sample_feature_names):
        """Test JSON generation with invalid model name."""
        # Setup
        adaptive_reviewer.models = {}
        
        # Execute and assert
        with pytest.raises(KeyError):
            adaptive_reviewer.get_feature_importance_json('invalid_model', sample_feature_names)
    
    @patch('core.adaptive_reviewer.logging')
    def test_calculate_feature_importance_logging(self, mock_logging, adaptive_reviewer, mock_random_forest, sample_feature_names):
        """Test that logging is called appropriately."""
        # Setup
        adaptive_reviewer.models = {'random_forest': mock_random_forest}
        mock_logger = Mock()
        mock_logging.getLogger.return_value = mock_logger
        
        # Execute
        adaptive_reviewer.calculate_feature_importance('random_forest', sample_feature_names)
        
        # Assert logging calls
        mock_logging.getLogger.assert_called_with('core.adaptive_reviewer')
        mock_logger.info.assert_called_once()
        mock_logger.debug.assert_called_once()
        
        # Check log messages
        info_call = mock_logger.info.call_args[0][0]
        assert "Calculated feature importance for model 'random_forest'" in info_call
        assert "4 features" in info_call
    
    def test_feature_importance_edge_case_empty_features(self, adaptive_reviewer):
        """Test edge case with empty feature list."""
        # Setup
        model = Mock(spec=RandomForestClassifier)
        model.feature_importances_ = np.array([])
        adaptive_reviewer.models = {'empty_model': model}
        
        # Execute
        result = adaptive_reviewer.calculate_feature_importance('empty_model', [])
        
        # Assert
        assert result == {}
    
    def test_feature_importance_edge_case_single_feature(self, adaptive_reviewer):
        """Test edge case with single feature."""
        # Setup
        model = Mock(spec=RandomForestClassifier)
        model.feature_importances_ = np.array([0.5])
        adaptive_reviewer.models = {'single_model': model}
        
        # Execute
        result = adaptive_reviewer.calculate_feature_importance('single_model', ['single_feature'])
        
        # Assert
        assert result == {'single_feature': 0.5}
    
    @pytest.mark.parametrize("importance_values,expected_order", [
        ([0.1, 0.2, 0.3, 0.4], ['feature_4', 'feature_3', 'feature_2', 'feature_1']),
        ([0.4, 0.3, 0.2, 0.1], ['feature_1', 'feature_2', 'feature_3', 'feature_4']),
        ([0.25, 0.25, 0.25, 0.25], ['feature_1', 'feature_2', 'feature_3', 'feature_4']),  # Ties maintain order
    ])
    def test_feature_importance_sorting_variations(self, adaptive_reviewer, importance_values, expected_order):
        """Test feature importance sorting with various importance distributions."""
        # Setup
        model = Mock(spec=RandomForestClassifier)
        model.feature_importances_ = np.array(importance_values)
        adaptive_reviewer.models = {'test_model': model}
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        # Execute
        result = adaptive_reviewer.calculate_feature_importance('test_model', feature_names)
        
        # Assert
        result_order = list(result.keys())
        assert result_order == expected_order


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
