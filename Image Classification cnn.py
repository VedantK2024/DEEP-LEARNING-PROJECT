"""
Image Classification using Deep Learning
=========================================

CNN Model for MNIST Digit Classification using TensorFlow/Keras

This script implements a Convolutional Neural Network (CNN) for classifying
handwritten digits from the MNIST dataset.


"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print(" " * 15 + "IMAGE CLASSIFICATION MODEL")
print(" " * 10 + "CNN for MNIST Digit Recognition")
print("=" * 70)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


class MNISTImageClassifier:
    """
    CNN-based Image Classifier for MNIST digits.
    """
    
    def __init__(self, model_name="mnist_cnn"):
        """Initialize the classifier."""
        self.model_name = model_name
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.class_names = [str(i) for i in range(10)]
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset."""
        print("\n" + "=" * 70)
        print("STEP 1: Loading and Preprocessing Data")
        print("=" * 70)
        
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        print(f"\nAfter preprocessing:")
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Visualize sample images
        self._visualize_samples()
        
    def _visualize_samples(self, n_samples=16):
        """Visualize sample images from the dataset."""
        print("\nVisualizing sample images...")
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle('Sample MNIST Images', fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.x_train[i].squeeze(), cmap='gray')
            ax.set_title(f'Label: {self.y_train[i]}', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        print("✓ Sample images saved to 'sample_images.png'")
        plt.close()
        
    def build_model(self, architecture='standard'):
        """
        Build CNN model architecture.
        
        Args:
            architecture: 'standard' or 'advanced'
        """
        print("\n" + "=" * 70)
        print("STEP 2: Building Model Architecture")
        print("=" * 70)
        
        if architecture == 'standard':
            model = self._build_standard_cnn()
        else:
            model = self._build_advanced_cnn()
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Save model architecture visualization
        keras.utils.plot_model(
            self.model,
            to_file='model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=150
        )
        print("\n✓ Model architecture saved to 'model_architecture.png'")
        
    def _build_standard_cnn(self):
        """Build standard CNN architecture."""
        print("Building Standard CNN...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def _build_advanced_cnn(self):
        """Build advanced CNN with batch normalization."""
        print("Building Advanced CNN with Batch Normalization...")
        
        model = models.Sequential([
            # First Block
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def train_model(self, epochs=20, batch_size=128, validation_split=0.1):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data for validation
        """
        print("\n" + "=" * 70)
        print("STEP 3: Training Model")
        print("=" * 70)
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Validation Split: {validation_split}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training completed!")
        
    def evaluate_model(self):
        """Evaluate model on test data."""
        print("\n" + "=" * 70)
        print("STEP 4: Evaluating Model")
        print("=" * 70)
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Make predictions
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_classes, 
                                   target_names=self.class_names))
        
        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'total_parameters': self.model.count_params(),
            'epochs_trained': len(self.history.history['loss'])
        }
        
        with open('model_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✓ Results saved to 'model_results.json'")
        
        return y_pred_classes
    
    def visualize_training_history(self):
        """Visualize training history."""
        print("\nGenerating training history visualizations...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history saved to 'training_history.png'")
        plt.close()
        
    def visualize_confusion_matrix(self, y_pred_classes):
        """Visualize confusion matrix."""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(self.y_test, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - MNIST Digit Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved to 'confusion_matrix.png'")
        plt.close()
        
    def visualize_predictions(self, n_samples=16):
        """Visualize predictions on test samples."""
        print("\nGenerating prediction visualizations...")
        
        # Get predictions
        predictions = self.model.predict(self.x_test[:n_samples], verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.x_test[i].squeeze(), cmap='gray')
            
            true_label = self.y_test[i]
            pred_label = predicted_classes[i]
            confidence = predictions[i][pred_label] * 100
            
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%',
                        color=color, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ Predictions saved to 'predictions_visualization.png'")
        plt.close()
        
    def visualize_feature_maps(self):
        """Visualize convolutional layer feature maps."""
        print("\nGenerating feature map visualizations...")
        
        # Create model that outputs feature maps
        layer_outputs = [layer.output for layer in self.model.layers[:6]]  # First few layers
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
        
        # Get activations for a sample image
        sample_image = np.expand_dims(self.x_test[0], 0)
        activations = activation_model.predict(sample_image, verbose=0)
        
        # Visualize first convolutional layer
        first_layer_activation = activations[0]
        
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle('Feature Maps - First Convolutional Layer', 
                    fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            if i < first_layer_activation.shape[-1]:
                ax.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
                ax.set_title(f'Filter {i+1}', fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('feature_maps.png', dpi=300, bbox_inches='tight')
        print("✓ Feature maps saved to 'feature_maps.png'")
        plt.close()
        
    def save_model(self):
        """Save the trained model."""
        self.model.save(f'{self.model_name}_final.h5')
        print(f"\n✓ Model saved to '{self.model_name}_final.h5'")
        
    def run_complete_pipeline(self, architecture='standard', epochs=15):
        """Run the complete training and evaluation pipeline."""
        self.load_and_preprocess_data()
        self.build_model(architecture=architecture)
        self.train_model(epochs=epochs)
        y_pred = self.evaluate_model()
        self.visualize_training_history()
        self.visualize_confusion_matrix(y_pred)
        self.visualize_predictions()
        self.visualize_feature_maps()
        self.save_model()
        
        print("\n" + "=" * 70)
        print(" " * 20 + "PIPELINE COMPLETED!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  📊 sample_images.png - Sample training images")
        print("  🏗️  model_architecture.png - Model architecture diagram")
        print("  📈 training_history.png - Training progress")
        print("  🎯 confusion_matrix.png - Classification confusion matrix")
        print("  🔮 predictions_visualization.png - Model predictions")
        print("  🧠 feature_maps.png - Convolutional feature maps")
        print("  💾 mnist_cnn_final.h5 - Trained model file")
        print("  📄 model_results.json - Evaluation metrics")
        print("=" * 70)


if __name__ == "__main__":
    # Create and run classifier
    classifier = MNISTImageClassifier()
    classifier.run_complete_pipeline(architecture='standard', epochs=15)
