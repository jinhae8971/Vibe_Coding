"""
Core AutoML Pipeline for automatic task detection and optimization
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .data_processor import DataProcessor
from .task_detector import TaskDetector
from .data_synthesizer import DataSynthesizer
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .utils import setup_logging, save_results, load_config


class AutoMLPipeline:
    """
    Main AutoML Pipeline class that orchestrates the entire process
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AutoML Pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logging(self.config.get('log_level', 'INFO'))
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.task_detector = TaskDetector(self.config)
        self.data_synthesizer = DataSynthesizer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
        self.results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'optimization_trials': 100,
            'synthesis_methods': ['smote', 'adasyn', 'borderline', 'ctgan', 'copulas'],
            'models': {
                'classification': ['rf', 'xgb', 'lgb', 'catboost', 'svm'],
                'regression': ['rf', 'xgb', 'lgb', 'catboost', 'svr']
            },
            'log_level': 'INFO'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            default_config.update(user_config)
            
        return default_config
    
    def run_batch_processing(self, input_folder: str, output_folder: str) -> Dict[str, Any]:
        """
        Run batch processing on multiple datasets
        
        Args:
            input_folder: Path to folder containing input datasets
            output_folder: Path to folder for saving results
            
        Returns:
            Dictionary containing results for all processed files
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting batch processing from {input_folder} to {output_folder}")
        
        batch_results = {}
        supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        
        # Find all supported data files
        data_files = []
        for ext in supported_formats:
            data_files.extend(list(input_path.glob(f"*{ext}")))
        
        if not data_files:
            raise ValueError(f"No supported data files found in {input_folder}")
        
        self.logger.info(f"Found {len(data_files)} data files to process")
        
        for file_path in data_files:
            try:
                self.logger.info(f"Processing file: {file_path.name}")
                
                # Create output subfolder for this file
                file_output_path = output_path / file_path.stem
                file_output_path.mkdir(exist_ok=True)
                
                # Process single file
                result = self.run_single_file(str(file_path), str(file_output_path))
                batch_results[file_path.name] = result
                
                self.logger.info(f"Completed processing: {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {str(e)}")
                batch_results[file_path.name] = {'error': str(e)}
        
        # Save batch summary
        self._save_batch_summary(batch_results, output_path)
        
        return batch_results
    
    def run_single_file(self, file_path: str, output_folder: str) -> Dict[str, Any]:
        """
        Process a single data file
        
        Args:
            file_path: Path to input data file
            output_folder: Path to folder for saving results
            
        Returns:
            Dictionary containing processing results
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting processing of {file_path}")
        
        # Step 1: Load and preprocess data
        self.logger.info("Step 1: Loading and preprocessing data")
        data = self.data_processor.load_data(file_path)
        processed_data = self.data_processor.preprocess(data)
        
        # Step 2: Detect task type
        self.logger.info("Step 2: Detecting task type")
        task_info = self.task_detector.detect_task(processed_data)
        
        # Step 3: Data synthesis optimization
        self.logger.info("Step 3: Optimizing data synthesis methods")
        synthesis_results = self.data_synthesizer.optimize_synthesis(
            processed_data, task_info
        )
        
        # Step 4: Model training and optimization
        self.logger.info("Step 4: Training and optimizing models")
        model_results = self.model_trainer.train_models(
            synthesis_results, task_info
        )
        
        # Step 5: Evaluation and selection
        self.logger.info("Step 5: Evaluating and selecting best model")
        evaluation_results = self.evaluator.evaluate_models(
            model_results, task_info
        )
        
        # Compile results
        results = {
            'file_path': file_path,
            'task_info': task_info,
            'data_info': {
                'original_shape': data.shape,
                'processed_shape': processed_data['X'].shape,
                'target_shape': processed_data['y'].shape,
                'feature_names': processed_data.get('feature_names', [])
            },
            'synthesis_results': synthesis_results,
            'model_results': model_results,
            'evaluation_results': evaluation_results,
            'best_model': evaluation_results['best_model'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(results, output_path)
        
        self.logger.info(f"Processing completed for {file_path}")
        return results
    
    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save processing results to files"""
        
        # Save detailed results as YAML
        with open(output_path / 'results.yaml', 'w') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        # Save summary as JSON for easy parsing
        summary = {
            'task_type': results['task_info']['task_type'],
            'best_model': results['best_model']['model_name'],
            'best_synthesis': results['best_model']['synthesis_method'],
            'performance': results['best_model']['performance'],
            'data_shape': results['data_info']['processed_shape']
        }
        
        import json
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate and save visualizations
        self.evaluator.generate_visualizations(results, output_path)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def _save_batch_summary(self, batch_results: Dict[str, Any], output_path: Path):
        """Save batch processing summary"""
        
        summary = {
            'total_files': len(batch_results),
            'successful': sum(1 for r in batch_results.values() if 'error' not in r),
            'failed': sum(1 for r in batch_results.values() if 'error' in r),
            'results_by_file': {}
        }
        
        for filename, result in batch_results.items():
            if 'error' not in result:
                summary['results_by_file'][filename] = {
                    'task_type': result['task_info']['task_type'],
                    'best_model': result['best_model']['model_name'],
                    'performance': result['best_model']['performance']
                }
            else:
                summary['results_by_file'][filename] = {'error': result['error']}
        
        # Save batch summary
        import json
        with open(output_path / 'batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Batch summary saved to {output_path / 'batch_summary.json'}")