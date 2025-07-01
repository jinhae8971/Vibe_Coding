"""
Model training module with hyperparameter optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries with error handling
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.metrics import classification_report, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class ModelTrainer:
    """
    Model training class with automatic hyperparameter optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('AutoML.ModelTrainer')
        self.available_models = self._get_available_models()
        
    def _get_available_models(self) -> Dict[str, List[str]]:
        """Get available models based on installed libraries"""
        models = {
            'classification': [],
            'regression': []
        }
        
        if SKLEARN_AVAILABLE:
            models['classification'].extend(['rf', 'svm'])
            models['regression'].extend(['rf', 'svr'])
        
        if XGB_AVAILABLE:
            models['classification'].append('xgb')
            models['regression'].append('xgb')
        
        if LGB_AVAILABLE:
            models['classification'].append('lgb')
            models['regression'].append('lgb')
        
        if CATBOOST_AVAILABLE:
            models['classification'].append('catboost')
            models['regression'].append('catboost')
        
        return models
    
    def train_models(self, synthesis_results: Dict[str, Any], task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter optimization
        
        Args:
            synthesis_results: Results from data synthesis
            task_info: Task information
            
        Returns:
            Dictionary containing trained models and results
        """
        task_type = task_info['task_type']
        self.logger.info(f"Starting model training for {task_type} task")
        
        model_results = {}
        
        # Get available models for this task type
        available_models = self.available_models.get(task_type, [])
        
        if not available_models:
            raise ValueError(f"No models available for {task_type} task")
        
        # Train models on each synthesis method's data
        for synthesis_method, synthesis_data in synthesis_results.items():
            if synthesis_method in ['best_method', 'best_performance']:
                continue
                
            X = synthesis_data['X']
            y = synthesis_data['y']
            
            self.logger.info(f"Training models on {synthesis_method} data")
            
            method_results = {}
            
            for model_name in available_models:
                try:
                    self.logger.info(f"Training {model_name} on {synthesis_method}")
                    
                    # Get optimized model
                    best_model, best_params, cv_score = self._optimize_model(
                        X, y, model_name, task_type
                    )
                    
                    method_results[model_name] = {
                        'model': best_model,
                        'params': best_params,
                        'cv_score': cv_score,
                        'synthesis_method': synthesis_method
                    }
                    
                    self.logger.info(f"{model_name} CV score: {cv_score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {str(e)}")
            
            model_results[synthesis_method] = method_results
        
        return model_results
    
    def _optimize_model(self, X: pd.DataFrame, y: pd.Series, 
                       model_name: str, task_type: str) -> Tuple[Any, Dict, float]:
        """Optimize hyperparameters for a specific model"""
        
        if OPTUNA_AVAILABLE:
            return self._optuna_optimization(X, y, model_name, task_type)
        else:
            return self._default_training(X, y, model_name, task_type)
    
    def _optuna_optimization(self, X: pd.DataFrame, y: pd.Series, 
                            model_name: str, task_type: str) -> Tuple[Any, Dict, float]:
        """Hyperparameter optimization using Optuna"""
        
        def objective(trial):
            params = self._get_param_space(trial, model_name, task_type)
            model = self._create_model(model_name, task_type, params)
            
            # Cross-validation
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=min(5, len(X)), shuffle=True, 
                                   random_state=self.config.get('random_state', 42))
                scoring = 'f1_weighted'
            else:
                cv = KFold(n_splits=min(5, len(X)), shuffle=True, 
                         random_state=self.config.get('random_state', 42))
                scoring = 'neg_mean_squared_error'
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.get('random_state', 42))
        )
        
        # Optimize
        n_trials = min(self.config.get('optimization_trials', 100), 50)  # Limit for speed
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best model
        best_params = study.best_params
        best_model = self._create_model(model_name, task_type, best_params)
        best_model.fit(X, y)
        
        return best_model, best_params, study.best_value
    
    def _default_training(self, X: pd.DataFrame, y: pd.Series, 
                         model_name: str, task_type: str) -> Tuple[Any, Dict, float]:
        """Default training without hyperparameter optimization"""
        
        default_params = self._get_default_params(model_name, task_type)
        model = self._create_model(model_name, task_type, default_params)
        
        # Cross-validation
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=min(5, len(X)), shuffle=True, 
                               random_state=self.config.get('random_state', 42))
            scoring = 'f1_weighted'
        else:
            cv = KFold(n_splits=min(5, len(X)), shuffle=True, 
                     random_state=self.config.get('random_state', 42))
            scoring = 'neg_mean_squared_error'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_score = scores.mean()
        
        # Fit final model
        model.fit(X, y)
        
        return model, default_params, cv_score
    
    def _create_model(self, model_name: str, task_type: str, params: Dict):
        """Create model instance with given parameters"""
        
        if model_name == 'rf':
            if task_type == 'classification':
                return RandomForestClassifier(**params)
            else:
                return RandomForestRegressor(**params)
        
        elif model_name == 'svm':
            if task_type == 'classification':
                return SVC(**params)
            else:
                return SVR(**params)
        
        elif model_name == 'xgb' and XGB_AVAILABLE:
            if task_type == 'classification':
                return xgb.XGBClassifier(**params)
            else:
                return xgb.XGBRegressor(**params)
        
        elif model_name == 'lgb' and LGB_AVAILABLE:
            if task_type == 'classification':
                return lgb.LGBMClassifier(**params)
            else:
                return lgb.LGBMRegressor(**params)
        
        elif model_name == 'catboost' and CATBOOST_AVAILABLE:
            if task_type == 'classification':
                return cb.CatBoostClassifier(**params)
            else:
                return cb.CatBoostRegressor(**params)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _get_param_space(self, trial, model_name: str, task_type: str) -> Dict:
        """Define hyperparameter search space for Optuna"""
        
        if model_name == 'rf':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.config.get('random_state', 42)
            }
        
        elif model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'random_state': self.config.get('random_state', 42)
            }
        
        elif model_name == 'xgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.config.get('random_state', 42)
            }
        
        elif model_name == 'lgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.config.get('random_state', 42),
                'verbose': -1
            }
        
        elif model_name == 'catboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'random_state': self.config.get('random_state', 42),
                'verbose': False
            }
        
        return {}
    
    def _get_default_params(self, model_name: str, task_type: str) -> Dict:
        """Get default parameters when optimization is not available"""
        
        base_params = {'random_state': self.config.get('random_state', 42)}
        
        if model_name == 'rf':
            base_params.update({'n_estimators': 100, 'max_depth': 10})
        elif model_name in ['lgb', 'catboost']:
            base_params.update({'verbose': -1 if model_name == 'lgb' else False})
        
        return base_params