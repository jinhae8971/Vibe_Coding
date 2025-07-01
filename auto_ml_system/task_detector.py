"""
Task detection module for automatically identifying classification vs regression tasks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from sklearn.preprocessing import LabelEncoder


class TaskDetector:
    """
    Automatic task type detection (classification vs regression)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TaskDetector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('AutoML.TaskDetector')
    
    def detect_task(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the type of machine learning task
        
        Args:
            processed_data: Preprocessed data dictionary
            
        Returns:
            Dictionary containing task information
        """
        X = processed_data['X']
        y = processed_data['y']
        target_type = processed_data.get('target_type', 'unknown')
        
        self.logger.info("Detecting task type and characteristics")
        
        # Basic task info
        task_info = {
            'task_type': target_type,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'target_name': processed_data.get('target_name', 'target')
        }
        
        # Detailed analysis based on task type
        if target_type == 'classification':
            task_info.update(self._analyze_classification_task(y))
        else:
            task_info.update(self._analyze_regression_task(y))
        
        # Dataset size category
        task_info['dataset_size'] = self._categorize_dataset_size(X.shape[0])
        
        # Data complexity analysis
        task_info.update(self._analyze_data_complexity(X, y))
        
        # Recommendations
        task_info['recommendations'] = self._generate_recommendations(task_info)
        
        self.logger.info(f"Detected task: {task_info['task_type']}")
        self.logger.info(f"Dataset size: {task_info['dataset_size']}")
        
        return task_info
    
    def _analyze_classification_task(self, y: pd.Series) -> Dict[str, Any]:
        """Analyze classification-specific characteristics"""
        
        unique_classes = y.nunique()
        class_counts = y.value_counts()
        
        # Determine classification type
        if unique_classes == 2:
            classification_type = 'binary'
        else:
            classification_type = 'multiclass'
        
        # Check class balance
        min_class_ratio = class_counts.min() / len(y)
        max_class_ratio = class_counts.max() / len(y)
        
        if min_class_ratio < 0.1:
            balance_status = 'highly_imbalanced'
        elif min_class_ratio < 0.3:
            balance_status = 'imbalanced'
        else:
            balance_status = 'balanced'
        
        return {
            'classification_type': classification_type,
            'n_classes': unique_classes,
            'class_distribution': class_counts.to_dict(),
            'balance_status': balance_status,
            'min_class_ratio': min_class_ratio,
            'max_class_ratio': max_class_ratio
        }
    
    def _analyze_regression_task(self, y: pd.Series) -> Dict[str, Any]:
        """Analyze regression-specific characteristics"""
        
        # Target distribution analysis
        target_stats = {
            'mean': float(y.mean()),
            'std': float(y.std()),
            'min': float(y.min()),
            'max': float(y.max()),
            'median': float(y.median()),
            'skewness': float(y.skew()),
            'kurtosis': float(y.kurtosis())
        }
        
        # Determine target distribution type
        if abs(target_stats['skewness']) < 0.5:
            distribution_type = 'normal'
        elif target_stats['skewness'] > 1:
            distribution_type = 'right_skewed'
        elif target_stats['skewness'] < -1:
            distribution_type = 'left_skewed'
        else:
            distribution_type = 'moderately_skewed'
        
        # Check for outliers (using IQR method)
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold_low = Q1 - 1.5 * IQR
        outlier_threshold_high = Q3 + 1.5 * IQR
        
        n_outliers = ((y < outlier_threshold_low) | (y > outlier_threshold_high)).sum()
        outlier_ratio = n_outliers / len(y)
        
        return {
            'target_stats': target_stats,
            'distribution_type': distribution_type,
            'n_outliers': int(n_outliers),
            'outlier_ratio': float(outlier_ratio)
        }
    
    def _categorize_dataset_size(self, n_samples: int) -> str:
        """Categorize dataset size"""
        
        if n_samples < 100:
            return 'very_small'
        elif n_samples < 1000:
            return 'small'
        elif n_samples < 10000:
            return 'medium'
        elif n_samples < 100000:
            return 'large'
        else:
            return 'very_large'
    
    def _analyze_data_complexity(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze data complexity characteristics"""
        
        n_samples, n_features = X.shape
        
        # Feature-to-sample ratio
        feature_ratio = n_features / n_samples
        
        # Determine complexity level
        if feature_ratio > 0.5:
            complexity = 'high_dimensional'
        elif feature_ratio > 0.1:
            complexity = 'medium_dimensional'
        else:
            complexity = 'low_dimensional'
        
        # Feature correlation analysis
        if hasattr(X, 'corr'):
            corr_matrix = X.corr().abs()
            high_corr_pairs = (corr_matrix > 0.8).sum().sum() - n_features  # Subtract diagonal
            correlation_level = 'high' if high_corr_pairs > n_features * 0.1 else 'moderate' if high_corr_pairs > 0 else 'low'
        else:
            correlation_level = 'unknown'
        
        # Missing data analysis
        if hasattr(X, 'isnull'):
            missing_ratio = X.isnull().sum().sum() / (n_samples * n_features)
        else:
            missing_ratio = 0.0
        
        return {
            'complexity': complexity,
            'feature_ratio': feature_ratio,
            'correlation_level': correlation_level,
            'missing_data_ratio': missing_ratio
        }
    
    def _generate_recommendations(self, task_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate recommendations based on task analysis"""
        
        recommendations = {
            'data_synthesis': [],
            'models': [],
            'preprocessing': [],
            'evaluation': []
        }
        
        task_type = task_info['task_type']
        dataset_size = task_info['dataset_size']
        
        # Data synthesis recommendations
        if dataset_size in ['very_small', 'small']:
            if task_type == 'classification':
                if task_info.get('balance_status') == 'imbalanced':
                    recommendations['data_synthesis'].extend(['SMOTE', 'ADASYN', 'BorderlineSMOTE'])
                else:
                    recommendations['data_synthesis'].extend(['CTGAN', 'Copulas', 'SMOTE'])
            else:
                recommendations['data_synthesis'].extend(['Copulas', 'CTGAN'])
        
        # Model recommendations
        if dataset_size in ['very_small', 'small']:
            if task_type == 'classification':
                recommendations['models'].extend(['RandomForest', 'SVM', 'LightGBM'])
            else:
                recommendations['models'].extend(['RandomForest', 'SVR', 'LightGBM'])
        else:
            recommendations['models'].extend(['XGBoost', 'LightGBM', 'CatBoost'])
        
        # Preprocessing recommendations
        if task_info.get('complexity') == 'high_dimensional':
            recommendations['preprocessing'].extend(['feature_selection', 'pca'])
        
        if task_type == 'regression' and task_info.get('distribution_type') in ['right_skewed', 'left_skewed']:
            recommendations['preprocessing'].append('target_transformation')
        
        # Evaluation recommendations
        if task_type == 'classification':
            if task_info.get('balance_status') == 'imbalanced':
                recommendations['evaluation'].extend(['f1_score', 'roc_auc', 'precision_recall'])
            else:
                recommendations['evaluation'].extend(['accuracy', 'f1_score', 'roc_auc'])
        else:
            recommendations['evaluation'].extend(['rmse', 'mae', 'r2'])
        
        return recommendations