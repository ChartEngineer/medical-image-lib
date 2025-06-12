"""
Medical Image Models Module

This module provides pre-configured deep learning models optimized for medical image classification
with limited data. Includes transfer learning capabilities and model architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np

class MedicalImageClassifier:
    """
    A comprehensive medical image classifier with transfer learning capabilities.
    """
    
    def __init__(self, model_name='resnet50', num_classes=2, input_shape=(224, 224, 3), 
                 weights='imagenet', fine_tune_layers=None):
        """
        Initialize the medical image classifier.
        
        Args:
            model_name (str): Base model architecture
            num_classes (int): Number of output classes
            input_shape (tuple): Input image shape
            weights (str): Pre-trained weights to use
            fine_tune_layers (int): Number of layers to fine-tune
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.fine_tune_layers = fine_tune_layers
        self.model = None
        self.history = None
        
    def create_base_model(self):
        """
        Create the base pre-trained model.
        
        Returns:
            keras.Model: Base model without top layers
        """
        if self.model_name.lower() == 'resnet50':
            base_model = applications.ResNet50(
                weights=self.weights,
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name.lower() == 'vgg16':
            base_model = applications.VGG16(
                weights=self.weights,
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name.lower() == 'efficientnetb0':
            base_model = applications.EfficientNetB0(
                weights=self.weights,
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name.lower() == 'densenet121':
            base_model = applications.DenseNet121(
                weights=self.weights,
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return base_model
    
    def build_model(self, dropout_rate=0.5, l2_reg=0.01):
        """
        Build the complete model with custom top layers.
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            l2_reg (float): L2 regularization strength
            
        Returns:
            keras.Model: Complete model
        """
        # Create base model
        base_model = self.create_base_model()
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom top layers
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers with regularization
        x = layers.Dense(
            512, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'top_3_accuracy']
        
        # Create model
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        self.base_model = base_model
        
        return model
    
    def enable_fine_tuning(self, learning_rate=0.0001):
        """
        Enable fine-tuning of the base model.
        
        Args:
            learning_rate (float): Learning rate for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model must be built before enabling fine-tuning")
        
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        if self.fine_tune_layers:
            for layer in self.base_model.layers[:-self.fine_tune_layers]:
                layer.trainable = False
        
        # Recompile with lower learning rate
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'top_3_accuracy']
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        print(f"Fine-tuning enabled with learning rate: {learning_rate}")
    
    def create_few_shot_model(self, support_set_size=5):
        """
        Create a model optimized for few-shot learning.
        
        Args:
            support_set_size (int): Number of examples per class in support set
            
        Returns:
            keras.Model: Few-shot learning model
        """
        # Create a siamese network for few-shot learning
        base_model = self.create_base_model()
        base_model.trainable = False
        
        # Feature extraction branch
        feature_extractor = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu')
        ])
        
        # Input for query image
        query_input = keras.Input(shape=self.input_shape, name='query')
        query_features = feature_extractor(query_input)
        
        # Input for support set
        support_input = keras.Input(shape=(support_set_size,) + self.input_shape, name='support')
        
        # Process support set
        support_features = []
        for i in range(support_set_size):
            support_img = support_input[:, i, :, :, :]
            support_feat = feature_extractor(support_img)
            support_features.append(support_feat)
        
        # Compute similarities
        similarities = []
        for support_feat in support_features:
            # Cosine similarity
            similarity = layers.Dot(axes=1, normalize=True)([query_features, support_feat])
            similarities.append(similarity)
        
        # Combine similarities
        combined_similarities = layers.Concatenate()(similarities)
        
        # Classification based on similarities
        output = layers.Dense(self.num_classes, activation='softmax')(combined_similarities)
        
        few_shot_model = keras.Model([query_input, support_input], output)
        few_shot_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return few_shot_model
    
    def get_model_summary(self):
        """
        Get model summary and architecture information.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"error": "Model not built yet"}
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        return {
            "model_name": self.model_name,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "layers": len(self.model.layers)
        }
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_single_image(self, image, preprocess=True):
        """
        Predict on a single image.
        
        Args:
            image (np.array): Input image
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            np.array: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        if preprocess:
            # Resize and normalize
            image = tf.image.resize(image, self.input_shape[:2])
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.expand_dims(image, 0)
        
        predictions = self.model.predict(image)
        return predictions[0]

def create_custom_cnn(input_shape, num_classes, architecture='simple'):
    """
    Create a custom CNN architecture for medical images.
    
    Args:
        input_shape (tuple): Input image shape
        num_classes (int): Number of output classes
        architecture (str): Architecture type ('simple', 'medium', 'complex')
        
    Returns:
        keras.Model: Custom CNN model
    """
    inputs = keras.Input(shape=input_shape)
    
    if architecture == 'simple':
        # Simple CNN for small datasets
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
    elif architecture == 'medium':
        # Medium complexity CNN
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
    else:  # complex
        # More complex CNN with residual connections
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        for filters in [64, 128, 256]:
            shortcut = x
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1)(shortcut)
            
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D()(x)
        
        x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

