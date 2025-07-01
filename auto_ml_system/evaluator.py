"""
Model evaluation and visualization module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

# Import libraries with error handling
try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    from sklearn.model_selection import learning_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelEvaluator:
    """
    Model evaluation and visualization class
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelEvaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('AutoML.ModelEvaluator')
        
    def evaluate_models(self, model_results: Dict[str, Any], task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all trained models and select the best one
        
        Args:
            model_results: Results from model training
            task_info: Task information
            
        Returns:
            Dictionary containing evaluation results
        """
        task_type = task_info['task_type']
        self.logger.info(f"Evaluating models for {task_type} task")
        
        evaluation_results = {}
        best_score = -np.inf
        best_model_info = None
        
        # Evaluate each model
        for synthesis_method, method_models in model_results.items():
            for model_name, model_info in method_models.items():
                try:
                    # Calculate detailed metrics
                    metrics = self._calculate_metrics(
                        model_info['model'], 
                        model_info.get('X_test'),
                        model_info.get('y_test'),
                        task_type
                    )
                    
                    # Use CV score as primary metric
                    primary_score = model_info['cv_score']
                    
                    evaluation_key = f"{synthesis_method}_{model_name}"
                    evaluation_results[evaluation_key] = {
                        'model_name': model_name,
                        'synthesis_method': synthesis_method,
                        'cv_score': primary_score,
                        'params': model_info['params'],
                        'metrics': metrics,
                        'model': model_info['model']
                    }
                    
                    # Track best model
                    if primary_score > best_score:
                        best_score = primary_score
                        best_model_info = evaluation_results[evaluation_key].copy()
                        best_model_info['performance'] = primary_score
                    
                    self.logger.info(f"{evaluation_key} - Score: {primary_score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {synthesis_method}_{model_name}: {str(e)}")
        
        # Add best model information
        evaluation_results['best_model'] = best_model_info
        evaluation_results['task_info'] = task_info
        
        if best_model_info:
            self.logger.info(f"Best model: {best_model_info['model_name']} "
                           f"with {best_model_info['synthesis_method']} "
                           f"(score: {best_score:.4f})")
        
        return evaluation_results
    
    def _calculate_metrics(self, model, X_test: Optional[pd.DataFrame], 
                          y_test: Optional[pd.Series], task_type: str) -> Dict[str, float]:
        """Calculate detailed performance metrics"""
        
        if X_test is None or y_test is None:
            return {}
        
        try:
            y_pred = model.predict(X_test)
            
            if task_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted')
                }
                
                # Add ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                    except:
                        pass
                        
            else:  # regression
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def generate_visualizations(self, results: Dict[str, Any], output_path: Path):
        """Generate comprehensive visualizations"""
        
        self.logger.info("Generating visualizations")
        
        try:
            # Create visualization directory
            viz_path = output_path / 'visualizations'
            viz_path.mkdir(exist_ok=True)
            
            # Generate different types of visualizations
            self._plot_model_comparison(results, viz_path)
            self._plot_feature_importance(results, viz_path)
            self._plot_prediction_analysis(results, viz_path)
            
            if SHAP_AVAILABLE:
                self._generate_shap_plots(results, viz_path)
            
            self.logger.info(f"Visualizations saved to {viz_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
    
    def _plot_model_comparison(self, results: Dict[str, Any], output_path: Path):
        """Plot model performance comparison"""
        
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            eval_results = results['evaluation_results']
            
            # Extract performance data
            model_names = []
            scores = []
            synthesis_methods = []
            
            for key, result in eval_results.items():
                if key in ['best_model', 'task_info']:
                    continue
                    
                model_names.append(result['model_name'])
                scores.append(result['cv_score'])
                synthesis_methods.append(result['synthesis_method'])
            
            if not model_names:
                return
            
            # Create comparison plot
            plt.figure(figsize=(12, 8))
            
            # Create a bar plot
            x_pos = np.arange(len(model_names))
            bars = plt.bar(x_pos, scores, alpha=0.7)
            
            # Color bars by synthesis method
            unique_methods = list(set(synthesis_methods))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_methods)))
            method_colors = {method: color for method, color in zip(unique_methods, colors)}
            
            for bar, method in zip(bars, synthesis_methods):
                bar.set_color(method_colors[method])
            
            plt.xlabel('Models')
            plt.ylabel('CV Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x_pos, [f"{name}\n({method})" for name, method in zip(model_names, synthesis_methods)], 
                      rotation=45, ha='right')
            
            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=method_colors[method], label=method) 
                             for method in unique_methods]
            plt.legend(handles=legend_elements, title='Synthesis Method')
            
            plt.tight_layout()
            plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting model comparison: {str(e)}")
    
    def _plot_feature_importance(self, results: Dict[str, Any], output_path: Path):
        """Plot feature importance for the best model"""
        
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            best_model = results['evaluation_results']['best_model']['model']
            
            # Get feature importance
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_names = results['data_info'].get('feature_names', 
                                                        [f'Feature_{i}' for i in range(len(importances))])
                
                # Sort by importance
                indices = np.argsort(importances)[::-1]
                
                # Plot top 20 features
                top_n = min(20, len(importances))
                
                plt.figure(figsize=(10, 8))
                plt.bar(range(top_n), importances[indices[:top_n]])
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title('Feature Importance (Top 20)')
                plt.xticks(range(top_n), 
                          [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                           for i in indices[:top_n]], 
                          rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
    
    def _plot_prediction_analysis(self, results: Dict[str, Any], output_path: Path):
        """Plot prediction analysis"""
        
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            task_type = results['task_info']['task_type']
            
            if task_type == 'regression':
                self._plot_regression_analysis(results, output_path)
            else:
                self._plot_classification_analysis(results, output_path)
                
        except Exception as e:
            self.logger.error(f"Error plotting prediction analysis: {str(e)}")
    
    def _plot_regression_analysis(self, results: Dict[str, Any], output_path: Path):
        """Plot regression-specific analysis"""
        
        # This would require actual test predictions
        # For now, create a placeholder
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Regression Analysis\n(Requires test data)', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path / 'regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_classification_analysis(self, results: Dict[str, Any], output_path: Path):
        """Plot classification-specific analysis"""
        
        # This would require actual test predictions
        # For now, create a placeholder
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Classification Analysis\n(Requires test data)', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path / 'classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_shap_plots(self, results: Dict[str, Any], output_path: Path):
        """Generate SHAP explainability plots"""
        
        try:
            best_model = results['evaluation_results']['best_model']['model']
            
            # Create placeholder for SHAP analysis
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'SHAP Analysis\n(Requires sample data)', 
                        ha='center', va='center', fontsize=16)
                plt.axis('off')
                plt.savefig(output_path / 'shap_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error generating SHAP plots: {str(e)}")