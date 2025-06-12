"""
Example: Chest X-ray Classification

This example demonstrates how to use the medical image library
for chest X-ray classification with limited data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from medical_image_lib import MedicalImageClassifier, MedicalDataLoader

def main():
    """
    Main function demonstrating chest X-ray classification.
    """
    print("Chest X-ray Classification Example")
    print("="*50)
    
    # Initialize data loader
    data_loader = MedicalDataLoader(target_size=(224, 224), batch_size=16)
    
    # Create synthetic chest X-ray data (in real scenario, load actual data)
    print("Creating synthetic chest X-ray dataset...")
    X, y = data_loader.create_synthetic_medical_data(
        num_samples_per_class=50,  # Limited data scenario
        num_classes=2  # Normal vs Pneumonia
    )
    
    # Set class names
    data_loader.class_names = ['Normal', 'Pneumonia']
    data_loader.num_classes = 2
    
    # Preprocess images
    X = data_loader.preprocess_images(X, normalize=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
    
    # Show class distribution
    print("\nTraining set class distribution:")
    data_loader.get_class_distribution(y_train)
    
    # Visualize samples
    print("\nVisualizing sample images...")
    data_loader.visualize_samples(X_train[:8], y_train[:8], class_names=data_loader.class_names)
    
    # Initialize classifier with transfer learning
    print("\nInitializing classifier with ResNet50 backbone...")
    classifier = MedicalImageClassifier(
        model_name='resnet50',
        num_classes=2,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    
    # Build model
    model = classifier.build_model(dropout_rate=0.5, l2_reg=0.01)
    print(f"\nModel built successfully!")
    
    # Show model summary
    model_info = classifier.get_model_summary()
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Create TensorFlow datasets
    train_dataset = data_loader.create_tf_dataset(X_train, y_train, shuffle=True, augment=True)
    val_dataset = data_loader.create_tf_dataset(X_val, y_val, shuffle=False, augment=False)
    
    # Train model (initial training with frozen base)
    print("\nTraining model (Phase 1: Frozen base model)...")
    
    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    # Initial training
    history1 = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Enable fine-tuning
    print("\nEnabling fine-tuning (Phase 2: Unfrozen base model)...")
    classifier.enable_fine_tuning(learning_rate=0.0001)
    
    # Fine-tuning
    history2 = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_dataset = data_loader.create_tf_dataset(X_test, y_test, shuffle=False, augment=False)
    test_results = model.evaluate(test_dataset, verbose=1)
    
    print(f"\nTest Results:")
    print(f"Loss: {test_results[0]:.4f}")
    print(f"Accuracy: {test_results[1]:.4f}")
    if len(test_results) > 2:
        print(f"Precision: {test_results[2]:.4f}")
        print(f"Recall: {test_results[3]:.4f}")
    
    # Plot training history
    plot_training_history(history1, history2)
    
    # Make predictions on test set
    print("\nMaking predictions on test samples...")
    predictions = model.predict(test_dataset)
    
    # Show some predictions
    show_predictions(X_test[:4], y_test[:4], predictions[:4], data_loader.class_names)
    
    # Save model
    model_path = 'models/chest_xray_classifier.h5'
    os.makedirs('models', exist_ok=True)
    classifier.save_model(model_path)
    print(f"\nModel saved to {model_path}")

def plot_training_history(history1, history2):
    """Plot training history for both phases."""
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning Start')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning Start')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def show_predictions(images, true_labels, predictions, class_names):
    """Show predictions for sample images."""
    plt.figure(figsize=(12, 8))
    
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        
        true_class = class_names[true_labels[i]]
        pred_prob = predictions[i][0] if len(predictions[i]) == 1 else np.max(predictions[i])
        pred_class = class_names[1] if (len(predictions[i]) == 1 and pred_prob > 0.5) else class_names[np.argmax(predictions[i])]
        
        plt.title(f'True: {true_class}\nPred: {pred_class} ({pred_prob:.3f})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

