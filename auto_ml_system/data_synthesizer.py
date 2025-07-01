"""
Data synthesis module for generating synthetic data using various methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, f1_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import synthesis libraries with error handling
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False

try:
    from copulas.single_table import GaussianCopula
    COPULAS_AVAILABLE = True
except ImportError:
    COPULAS_AVAILABLE = False


class DataSynthesizer:
    """
    Data synthesis class for generating synthetic data using various methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataSynthesizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('AutoML.DataSynthesizer')
        self.synthesis_methods = self._get_available_methods()
        
    def _get_available_methods(self) -> List[str]:
        """Get list of available synthesis methods"""
        methods = ['none']  # Always include baseline
        
        if IMBLEARN_AVAILABLE:
            methods.extend(['smote', 'adasyn', 'borderline'])
        
        if CTGAN_AVAILABLE:
            methods.append('ctgan')
        
        if COPULAS_AVAILABLE:
            methods.append('copulas')
        
        return methods
    
    def optimize_synthesis(self, processed_data: Dict[str, Any], task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize data synthesis methods
        
        Args:
            processed_data: Preprocessed data dictionary
            task_info: Task information dictionary
            
        Returns:
            Dictionary containing synthesis results
        """
        X = processed_data['X']
        y = processed_data['y']
        task_type = task_info['task_type']
        
        self.logger.info("Starting data synthesis optimization")
        
        synthesis_results = {
            'original_data': {
                'X': X.copy(),
                'y': y.copy(),
                'method': 'none',
                'performance': None
            }
        }
        
        # Evaluate baseline (no synthesis)
        baseline_score = self._evaluate_synthesis(X, y, task_type)
        synthesis_results['original_data']['performance'] = baseline_score
        
        self.logger.info(f"Baseline performance: {baseline_score:.4f}")
        
        # Test different synthesis methods
        for method in self.synthesis_methods:
            if method == 'none':
                continue
                
            try:
                self.logger.info(f"Testing synthesis method: {method}")
                
                # Generate synthetic data
                X_syn, y_syn = self._generate_synthetic_data(X, y, method, task_info)
                
                if X_syn is not None and y_syn is not None:
                    # Evaluate synthesized data
                    syn_score = self._evaluate_synthesis(X_syn, y_syn, task_type)
                    
                    synthesis_results[method] = {
                        'X': X_syn,
                        'y': y_syn,
                        'method': method,
                        'performance': syn_score
                    }
                    
                    self.logger.info(f"{method} performance: {syn_score:.4f}")
                else:
                    self.logger.warning(f"Failed to generate data with {method}")
                    
            except Exception as e:
                self.logger.error(f"Error with {method}: {str(e)}")
        
        # Find best synthesis method
        best_method = max(synthesis_results.keys(), 
                         key=lambda k: synthesis_results[k]['performance'])
        
        synthesis_results['best_method'] = best_method
        synthesis_results['best_performance'] = synthesis_results[best_method]['performance']
        
        self.logger.info(f"Best synthesis method: {best_method} "
                        f"(performance: {synthesis_results['best_performance']:.4f})")
        
        return synthesis_results
    
    def _generate_synthetic_data(self, X: pd.DataFrame, y: pd.Series, 
                                method: str, task_info: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Generate synthetic data using specified method"""
        
        try:
            if method == 'smote' and IMBLEARN_AVAILABLE:
                return self._smote_synthesis(X, y, task_info)
            elif method == 'adasyn' and IMBLEARN_AVAILABLE:
                return self._adasyn_synthesis(X, y, task_info)
            elif method == 'borderline' and IMBLEARN_AVAILABLE:
                return self._borderline_synthesis(X, y, task_info)
            elif method == 'ctgan' and CTGAN_AVAILABLE:
                return self._ctgan_synthesis(X, y, task_info)
            elif method == 'copulas' and COPULAS_AVAILABLE:
                return self._copulas_synthesis(X, y, task_info)
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error in {method} synthesis: {str(e)}")
            return None, None
    
    def _smote_synthesis(self, X: pd.DataFrame, y: pd.Series, 
                        task_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data using SMOTE"""
        
        if task_info['task_type'] != 'classification':
            return None, None
        
        # Calculate target sample size (2x original)
        target_size = len(X) * 2
        
        smote = SMOTE(
            sampling_strategy='auto',
            random_state=self.config.get('random_state', 42),
            k_neighbors=min(5, len(X) - 1)
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Limit to target size
        if len(X_resampled) > target_size:
            indices = np.random.choice(len(X_resampled), target_size, replace=False)
            X_resampled = X_resampled.iloc[indices]
            y_resampled = y_resampled.iloc[indices]
        
        return X_resampled, y_resampled
    
    def _adasyn_synthesis(self, X: pd.DataFrame, y: pd.Series, 
                         task_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data using ADASYN"""
        
        if task_info['task_type'] != 'classification':
            return None, None
        
        adasyn = ADASYN(
            sampling_strategy='auto',
            random_state=self.config.get('random_state', 42),
            n_neighbors=min(5, len(X) - 1)
        )
        
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        # Limit size
        target_size = len(X) * 2
        if len(X_resampled) > target_size:
            indices = np.random.choice(len(X_resampled), target_size, replace=False)
            X_resampled = X_resampled.iloc[indices]
            y_resampled = y_resampled.iloc[indices]
        
        return X_resampled, y_resampled
    
    def _borderline_synthesis(self, X: pd.DataFrame, y: pd.Series, 
                             task_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data using BorderlineSMOTE"""
        
        if task_info['task_type'] != 'classification':
            return None, None
        
        borderline = BorderlineSMOTE(
            sampling_strategy='auto',
            random_state=self.config.get('random_state', 42),
            k_neighbors=min(5, len(X) - 1)
        )
        
        X_resampled, y_resampled = borderline.fit_resample(X, y)
        
        # Limit size
        target_size = len(X) * 2
        if len(X_resampled) > target_size:
            indices = np.random.choice(len(X_resampled), target_size, replace=False)
            X_resampled = X_resampled.iloc[indices]
            y_resampled = y_resampled.iloc[indices]
        
        return X_resampled, y_resampled
    
    def _ctgan_synthesis(self, X: pd.DataFrame, y: pd.Series, 
                        task_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data using CTGAN"""
        
        # Combine X and y for CTGAN
        data = pd.concat([X, y.rename('target')], axis=1)
        
        # Initialize CTGAN
        ctgan = CTGAN(
            epochs=50,  # Reduced for speed
            batch_size=min(500, len(data)),
            verbose=False
        )
        
        # Fit CTGAN
        ctgan.fit(data)
        
        # Generate synthetic data (2x original size)
        target_size = len(data) * 2
        synthetic_data = ctgan.sample(target_size)
        
        # Split back to X and y
        X_syn = synthetic_data.drop('target', axis=1)
        y_syn = synthetic_data['target']
        
        return X_syn, y_syn
    
    def _copulas_synthesis(self, X: pd.DataFrame, y: pd.Series, 
                          task_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data using Gaussian Copula"""
        
        # Combine X and y for Copula
        data = pd.concat([X, y.rename('target')], axis=1)
        
        # Initialize Gaussian Copula
        copula = GaussianCopula()
        
        # Fit copula
        copula.fit(data)
        
        # Generate synthetic data (2x original size)
        target_size = len(data) * 2
        synthetic_data = copula.sample(target_size)
        
        # Split back to X and y
        X_syn = synthetic_data.drop('target', axis=1)
        y_syn = synthetic_data['target']
        
        return X_syn, y_syn
    
    def _evaluate_synthesis(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
        """Evaluate the quality of synthesized data using cross-validation"""
        
        try:
            if task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.config.get('random_state', 42),
                    n_jobs=-1
                )
                scorer = make_scorer(f1_score, average='weighted')
            else:
                model = RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.config.get('random_state', 42),
                    n_jobs=-1
                )
                scorer = make_scorer(mean_squared_error, greater_is_better=False)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=min(5, len(X)), 
                scoring=scorer, n_jobs=-1
            )
            
            # Return mean score (absolute value for negative scores)
            return abs(cv_scores.mean())
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {str(e)}")
            return 0.0