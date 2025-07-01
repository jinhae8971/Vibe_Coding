"""
Utility functions for AutoML System
"""

import logging
import json
import yaml
import os
from typing import Any, Dict
from pathlib import Path


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'automl.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('AutoML')


def save_results(results: Dict[str, Any], output_path: Path, filename: str = 'results'):
    """Save results to multiple formats"""
    
    # Save as YAML
    with open(output_path / f'{filename}.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    # Save as JSON
    with open(output_path / f'{filename}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Configuration file must be YAML or JSON format")


def ensure_directory(path: str) -> Path:
    """Ensure directory exists and return Path object"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)