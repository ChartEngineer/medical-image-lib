"""
Medical Image Classification Library

A Python library for training image classification models with limited medical image data.
Focuses on transfer learning, data augmentation, and few-shot learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Emmanuel Sande"

from .models import MedicalImageClassifier
from .data_loader import MedicalDataLoader
from .augmentation import MedicalAugmentation
from .training import ModelTrainer
from .evaluation import ModelEvaluator

__all__ = [
    'MedicalImageClassifier',
    'MedicalDataLoader', 
    'MedicalAugmentation',
    'ModelTrainer',
    'ModelEvaluator'
]

