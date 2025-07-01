"""
Automatic ML System for Small Dataset Processing
Auto-detection of classification/regression tasks with data synthesis optimization
"""

__version__ = "1.0.0"
__author__ = "Auto ML System"

from .core import AutoMLPipeline
from .data_processor import DataProcessor
from .task_detector import TaskDetector
from .data_synthesizer import DataSynthesizer
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator

__all__ = [
    "AutoMLPipeline",
    "DataProcessor", 
    "TaskDetector",
    "DataSynthesizer",
    "ModelTrainer",
    "ModelEvaluator"
]