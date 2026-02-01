"""
Natural Language Processing - Sentiment Analysis
================================================

LSTM Model for Text Sentiment Classification using TensorFlow/Keras

This script implements a deep learning model for sentiment analysis
on movie reviews (IMDB dataset).

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
from wordcloud import WordCloud

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print(" " * 15 + "NLP SENTIMENT ANALYSIS MODEL")
print(" " * 10 + "LSTM for Movie Review Classification")
print("=" * 70)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


class SentimentAnalysisLSTM:
    """
    LSTM-based Sentiment Analysis for Text Classification.
    """
    
    def __init__(self, model_name="sentiment_lstm", max_words=10000, max_len=200):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name for saving the model
            max_words: Maximum number of words in vocabulary
            max_len: Maximum sequence length
        """
        self.model_name = model_name
        self.max_words = max_words
        self.max_len = max_len
        self.model = None
        self.history = None
        self.tokenizer = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.class_names = ['Negative', 'Positive']
        
    def load_and_preprocess_data(self):
        """Load and preprocess IMDB dataset."""
        print("\n" + "=" * 70)
        print("STEP 1: Loading and Preprocessing Data")
        print("=" * 70)
        
        # Load IMDB dataset
        print("Loading IMDB movie review dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
            num_words=self.max_words
        )
        
        print(f"Training samples: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Vocabulary size: {self.max_words}")
        
        # Get word index
        word_index = keras.datasets.imdb.get_word_index()
        self.reverse_word_index = {v: k for k, v in word_index.items()}
        
        # Pad sequences
        print("\nPadding sequences...")
        x_train = pad_sequences(x_train, maxlen=self.max_len, padding='post')
        x_test = pad_sequences(x_test, maxlen=self.max_len, padding='post')
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nClass distribution in training:")
        for label, count in zip(unique, counts):
            print(f"  {self.class_names[label]}: {count} ({count/len(y_train)*100:.1f}%)")
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Visualize data
        self._visualize_data_distribution()
        self._visualize_word_clouds()
        
    def decode_review(self, encoded_review):
        """Decode an encoded review back to text."""
        return ' '.join([self.reverse_word_index.get(i - 3, '?') for i in encoded_review])
    
    def _visualize_data_distribution(self):
        """Visualize data distribution."""
        print("\nGenerating data distribution visualizations...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        axes[0].bar([self.class_names[i] for i in unique], counts, color=['#ff6b6b', '#4ecdc4'])
        axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Reviews', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Sequence length distribution
        lengths = [len([w for w in seq if w != 0]) for seq in self.x_train]
        axes[1].hist(lengths, bins=50, color='#95e1d3', edgecolor='black')
        axes[1].axvline(self.max_len, color='red', linestyle='--', linewidth=2, label=f'Max Length ({self.max_len})')
        axes[1].set_title('Review Length Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Words', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Data distribution saved to 'data_distribution.png'")
        plt.close()
        
    def _visualize_word_clouds(self):
        """Generate word clouds for positive and negative reviews."""
        print("\nGenerating word clouds...")
        
        # Sample reviews from each class
        positive_reviews = [self.decode_review(self.x_train[i]) 
                          for i in range(len(self.y_train)) if self.y_train[i] == 1][:500]
        negative_reviews = [self.decode_review(self.x_train[i]) 
                          for i in range(len(self.y_train)) if self.y_train[i] == 0][:500]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Positive word cloud
        positive_text = ' '.join(positive_reviews)
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Greens').generate(positive_text)
        axes[0].imshow(wordcloud_pos, interpolation='bilinear')
        axes[0].set_title('Positive Reviews - Word Cloud', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Negative word cloud
        negative_text = ' '.join(negative_reviews)
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Reds').generate(negative_text)
        axes[1].imshow(wordcloud_neg, interpolation='bilinear')
        axes[1].set_title('Negative Reviews - Word Cloud', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('word_clouds.png', dpi=300, bbox_inches='tight')
        print("✓ Word clouds saved to 'word_clouds.png'")
        plt.close()
        
    def build_model(self, architecture='lstm'):
        """
        Build model architecture.
        
        Args:
            architecture: 'lstm', 'bidirectional_lstm', or 'gru'
        """
        print("\n" + "=" * 70)
        print("STEP 2: Building Model Architecture")
        print("=" * 70)
        
        if architecture == 'lstm':
            model = self._build_lstm()
        elif architecture == 'bidirectional_lstm':
            model = self._build_bidirectional_lstm()
        else:
            model = self._build_gru()
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Save architecture visualization
        keras.utils.plot_model(
            self.model,
            to_file='nlp_model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=150
        )
        print("\n✓ Model architecture saved to 'nlp_model_architecture.png'")
        
    def _build_lstm(self):
        """Build standard LSTM model."""
        print("Building LSTM Model...")
        
        model = models.Sequential([
            layers.Embedding(self.max_words, 128, input_length=self.max_len),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def _build_bidirectional_lstm(self):
        """Build bidirectional LSTM model."""
        print("Building Bidirectional LSTM Model...")
        
        model = models.Sequential([
            layers.Embedding(self.max_words, 128, input_length=self.max_len),
            layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def _build_gru(self):
        """Build GRU model."""
        print("Building GRU Model...")
        
        model = models.Sequential([
            layers.Embedding(self.max_words, 128, input_length=self.max_len),
            layers.GRU(64, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def train_model(self, epochs=10, batch_size=128, validation_split=0.2):
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
                patience=3,
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
                patience=2,
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
        results = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {results[0]:.4f}")
        print(f"  Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
        print(f"  Test Precision: {results[2]:.4f}")
        print(f"  Test Recall: {results[3]:.4f}")
        
        # Make predictions
        y_pred_prob = self.model.predict(self.x_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))
        
        # Save results
        results_dict = {
            'test_loss': float(results[0]),
            'test_accuracy': float(results[1]),
            'test_precision': float(results[2]),
            'test_recall': float(results[3]),
            'total_parameters': self.model.count_params(),
            'epochs_trained': len(self.history.history['loss'])
        }
        
        with open('nlp_model_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print("\n✓ Results saved to 'nlp_model_results.json'")
        
        return y_pred, y_pred_prob
    
    def visualize_training_history(self):
        """Visualize training history."""
        print("\nGenerating training history visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nlp_training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history saved to 'nlp_training_history.png'")
        plt.close()
        
    def visualize_confusion_matrix(self, y_pred):
        """Visualize confusion matrix."""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Sentiment Analysis', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Sentiment', fontsize=12)
        plt.ylabel('True Sentiment', fontsize=12)
        plt.tight_layout()
        plt.savefig('nlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved to 'nlp_confusion_matrix.png'")
        plt.close()
        
    def visualize_roc_curve(self, y_pred_prob):
        """Visualize ROC curve."""
        print("\nGenerating ROC curve...")
        
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Sentiment Analysis', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('nlp_roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved to 'nlp_roc_curve.png'")
        plt.close()
        
    def visualize_sample_predictions(self, n_samples=10):
        """Visualize sample predictions."""
        print("\nGenerating sample prediction visualizations...")
        
        # Get random samples
        indices = np.random.choice(len(self.x_test), n_samples, replace=False)
        
        predictions = self.model.predict(self.x_test[indices], verbose=0)
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(14, n_samples * 2))
        fig.suptitle('Sample Review Predictions', fontsize=16, fontweight='bold')
        
        for i, (idx, ax) in enumerate(zip(indices, axes)):
            review_text = self.decode_review(self.x_test[idx])
            true_sentiment = self.class_names[self.y_test[idx]]
            pred_sentiment = self.class_names[1 if predictions[i] > 0.5 else 0]
            confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]
            
            # Truncate review for display
            display_text = review_text[:200] + '...' if len(review_text) > 200 else review_text
            
            color = 'green' if true_sentiment == pred_sentiment else 'red'
            
            ax.text(0.05, 0.7, f"Review: {display_text}", fontsize=9, wrap=True, va='top')
            ax.text(0.05, 0.3, f"True: {true_sentiment} | Predicted: {pred_sentiment} ({confidence*100:.1f}%)",
                   fontsize=10, fontweight='bold', color=color)
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('nlp_sample_predictions.png', dpi=300, bbox_inches='tight')
        print("✓ Sample predictions saved to 'nlp_sample_predictions.png'")
        plt.close()
        
    def save_model(self):
        """Save the trained model."""
        self.model.save(f'{self.model_name}_final.h5')
        print(f"\n✓ Model saved to '{self.model_name}_final.h5'")
        
    def run_complete_pipeline(self, architecture='bidirectional_lstm', epochs=10):
        """Run the complete training and evaluation pipeline."""
        self.load_and_preprocess_data()
        self.build_model(architecture=architecture)
        self.train_model(epochs=epochs)
        y_pred, y_pred_prob = self.evaluate_model()
        self.visualize_training_history()
        self.visualize_confusion_matrix(y_pred)
        self.visualize_roc_curve(y_pred_prob)
        self.visualize_sample_predictions()
        self.save_model()
        
        print("\n" + "=" * 70)
        print(" " * 20 + "PIPELINE COMPLETED!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  📊 data_distribution.png - Data analysis")
        print("  ☁️  word_clouds.png - Word clouds for sentiments")
        print("  🏗️  nlp_model_architecture.png - Model architecture")
        print("  📈 nlp_training_history.png - Training metrics")
        print("  🎯 nlp_confusion_matrix.png - Confusion matrix")
        print("  📉 nlp_roc_curve.png - ROC curve")
        print("  🔮 nlp_sample_predictions.png - Sample predictions")
        print("  💾 sentiment_lstm_final.h5 - Trained model")
        print("  📄 nlp_model_results.json - Evaluation metrics")
        print("=" * 70)


if __name__ == "__main__":
    # Create and run sentiment analyzer
    analyzer = SentimentAnalysisLSTM()
    analyzer.run_complete_pipeline(architecture='bidirectional_lstm', epochs=10)
