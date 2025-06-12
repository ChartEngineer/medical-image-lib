"""
Data Loader Module for Medical Images

This module handles loading and preprocessing of medical image datasets
with support for various formats and limited data scenarios.
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MedicalDataLoader:
    """
    A class for loading and preprocessing medical image datasets.
    """
    
    def __init__(self, target_size=(224, 224), batch_size=32, validation_split=0.2):
        """
        Initialize the data loader.
        
        Args:
            target_size (tuple): Target image size
            batch_size (int): Batch size for training
            validation_split (float): Validation split ratio
        """
        self.target_size = target_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.class_names = None
        self.num_classes = None
        
    def load_from_directory(self, data_dir, subset=None):
        """
        Load images from directory structure.
        
        Args:
            data_dir (str): Path to data directory
            subset (str): 'training' or 'validation'
            
        Returns:
            tf.data.Dataset: Loaded dataset
        """
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.validation_split if subset else None,
            subset=subset,
            seed=42,
            image_size=self.target_size,
            batch_size=self.batch_size,
            label_mode='categorical'
        )
        
        if self.class_names is None:
            self.class_names = dataset.class_names
            self.num_classes = len(self.class_names)
            print(f"Found {self.num_classes} classes: {self.class_names}")
        
        return dataset
    
    def create_synthetic_medical_data(self, num_samples_per_class=100, num_classes=2):
        """
        Create synthetic medical image data for demonstration.
        
        Args:
            num_samples_per_class (int): Number of samples per class
            num_classes (int): Number of classes
            
        Returns:
            tuple: (X, y) arrays
        """
        np.random.seed(42)
        
        X = []
        y = []
        
        for class_idx in range(num_classes):
            for _ in range(num_samples_per_class):
                # Create synthetic medical image
                if class_idx == 0:  # Normal
                    # Create normal-looking image (smoother patterns)
                    img = self._create_normal_image()
                else:  # Abnormal
                    # Create abnormal-looking image (with artifacts)
                    img = self._create_abnormal_image()
                
                X.append(img)
                y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"Created synthetic dataset: {X.shape}, Classes: {num_classes}")
        return X, y
    
    def _create_normal_image(self):
        """Create a synthetic normal medical image."""
        # Create base image with smooth gradients
        img = np.zeros(self.target_size + (3,), dtype=np.uint8)
        
        # Add smooth circular patterns (simulating normal tissue)
        center_x, center_y = self.target_size[0] // 2, self.target_size[1] // 2
        y, x = np.ogrid[:self.target_size[0], :self.target_size[1]]
        
        # Multiple circular patterns
        for i in range(3):
            cx = center_x + np.random.randint(-50, 50)
            cy = center_y + np.random.randint(-50, 50)
            radius = np.random.randint(30, 80)
            
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            intensity = np.random.randint(100, 180)
            img[mask] = [intensity, intensity, intensity]
        
        # Add some noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = np.clip(img + noise, 0, 255)
        
        return img
    
    def _create_abnormal_image(self):
        """Create a synthetic abnormal medical image."""
        # Start with normal image
        img = self._create_normal_image()
        
        # Add abnormal features
        # Add bright spots (simulating lesions)
        for _ in range(np.random.randint(1, 4)):
            x = np.random.randint(20, self.target_size[0] - 20)
            y = np.random.randint(20, self.target_size[1] - 20)
            size = np.random.randint(5, 15)
            
            cv2.circle(img, (y, x), size, (255, 255, 255), -1)
        
        # Add irregular shapes
        for _ in range(np.random.randint(1, 3)):
            pts = np.random.randint(0, min(self.target_size), (6, 2))
            cv2.fillPoly(img, [pts], (200, 200, 200))
        
        return img
    
    def preprocess_images(self, images, normalize=True):
        """
        Preprocess images for training.
        
        Args:
            images (np.array): Input images
            normalize (bool): Whether to normalize pixel values
            
        Returns:
            np.array: Preprocessed images
        """
        if normalize:
            images = images.astype(np.float32) / 255.0
        
        return images
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (np.array): Features
            y (np.array): Labels
            test_size (float): Test set proportion
            val_size (float): Validation set proportion
            
        Returns:
            tuple: Split datasets
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_tf_dataset(self, X, y, shuffle=True, augment=False):
        """
        Create TensorFlow dataset from arrays.
        
        Args:
            X (np.array): Images
            y (np.array): Labels
            shuffle (bool): Whether to shuffle data
            augment (bool): Whether to apply augmentation
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Convert labels to categorical if needed
        if len(y.shape) == 1:
            y = to_categorical(y, num_classes=self.num_classes)
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.batch_size)
        
        if augment:
            dataset = dataset.map(self._augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_batch(self, images, labels):
        """Apply augmentation to a batch of images."""
        # Random rotation
        images = tf.image.rot90(images, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random flip
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        
        # Random brightness and contrast
        images = tf.image.random_brightness(images, 0.1)
        images = tf.image.random_contrast(images, 0.9, 1.1)
        
        return images, labels
    
    def visualize_samples(self, X, y, num_samples=8, class_names=None):
        """
        Visualize sample images from the dataset.
        
        Args:
            X (np.array): Images
            y (np.array): Labels
            num_samples (int): Number of samples to show
            class_names (list): Class names for labels
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            img = X[idx]
            label = y[idx] if len(y.shape) == 1 else np.argmax(y[idx])
            
            axes[i].imshow(img)
            axes[i].set_title(f"{class_names[label]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self, y):
        """
        Get class distribution statistics.
        
        Args:
            y (np.array): Labels
            
        Returns:
            dict: Class distribution
        """
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)
        
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print("Class distribution:")
        for class_idx, count in distribution.items():
            class_name = self.class_names[class_idx] if self.class_names else f"Class {class_idx}"
            print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
        
        return distribution
    
    def save_dataset(self, X, y, save_dir):
        """
        Save dataset to disk.
        
        Args:
            X (np.array): Images
            y (np.array): Labels
            save_dir (str): Directory to save data
        """
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'images.npy'), X)
        np.save(os.path.join(save_dir, 'labels.npy'), y)
        
        print(f"Dataset saved to {save_dir}")
    
    def load_dataset(self, save_dir):
        """
        Load dataset from disk.
        
        Args:
            save_dir (str): Directory containing saved data
            
        Returns:
            tuple: (X, y) arrays
        """
        X = np.load(os.path.join(save_dir, 'images.npy'))
        y = np.load(os.path.join(save_dir, 'labels.npy'))
        
        print(f"Dataset loaded from {save_dir}: {X.shape}")
        return X, y

def create_few_shot_dataset(X, y, n_way=2, k_shot=5, n_query=10):
    """
    Create a few-shot learning dataset.
    
    Args:
        X (np.array): Images
        y (np.array): Labels
        n_way (int): Number of classes per episode
        k_shot (int): Number of support examples per class
        n_query (int): Number of query examples per class
        
    Returns:
        tuple: Support and query sets
    """
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    
    # Select random classes
    unique_classes = np.unique(y)
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)
    
    support_X, support_y = [], []
    query_X, query_y = [], []
    
    for i, class_idx in enumerate(selected_classes):
        class_indices = np.where(y == class_idx)[0]
        selected_indices = np.random.choice(class_indices, k_shot + n_query, replace=False)
        
        # Support set
        support_indices = selected_indices[:k_shot]
        support_X.extend(X[support_indices])
        support_y.extend([i] * k_shot)
        
        # Query set
        query_indices = selected_indices[k_shot:]
        query_X.extend(X[query_indices])
        query_y.extend([i] * n_query)
    
    return (np.array(support_X), np.array(support_y)), (np.array(query_X), np.array(query_y))

