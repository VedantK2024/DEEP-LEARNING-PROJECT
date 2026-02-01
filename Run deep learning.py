"""
Deep Learning Models - Main Runner
===================================

Run either Image Classification (CNN) or NLP Sentiment Analysis (LSTM)

"""

import sys
import argparse


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 80)
    print(" " * 25 + "DEEP LEARNING MODELS")
    print(" " * 20 + "TensorFlow/Keras Implementation")
    print("=" * 80)
    print("\nAvailable Models:")
    print("  1. Image Classification - CNN for MNIST Digit Recognition")
    print("  2. NLP Sentiment Analysis - LSTM for Movie Review Classification")
    print("=" * 80)


def run_image_classification(epochs=15, architecture='standard'):
    """Run image classification model."""
    from image_classification_cnn import MNISTImageClassifier
    
    print("\n🖼️  Starting Image Classification Model...")
    classifier = MNISTImageClassifier()
    classifier.run_complete_pipeline(architecture=architecture, epochs=epochs)


def run_nlp_sentiment(epochs=10, architecture='bidirectional_lstm'):
    """Run NLP sentiment analysis model."""
    from nlp_sentiment_analysis import SentimentAnalysisLSTM
    
    print("\n📝 Starting NLP Sentiment Analysis Model...")
    analyzer = SentimentAnalysisLSTM()
    analyzer.run_complete_pipeline(architecture=architecture, epochs=epochs)


def run_both_models():
    """Run both models sequentially."""
    print("\n🚀 Running BOTH models sequentially...\n")
    
    print("\n" + "=" * 80)
    print("PART 1: IMAGE CLASSIFICATION")
    print("=" * 80)
    run_image_classification(epochs=10, architecture='standard')
    
    print("\n\n" + "=" * 80)
    print("PART 2: NLP SENTIMENT ANALYSIS")
    print("=" * 80)
    run_nlp_sentiment(epochs=8, architecture='bidirectional_lstm')
    
    print("\n\n" + "=" * 80)
    print("✅ BOTH MODELS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nCheck the generated PNG and H5 files for results and trained models.")


def interactive_mode():
    """Run in interactive mode."""
    print_banner()
    
    print("\nSelect an option:")
    print("  1 - Run Image Classification (CNN)")
    print("  2 - Run NLP Sentiment Analysis (LSTM)")
    print("  3 - Run BOTH models")
    print("  4 - Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\nImage Classification Options:")
            print("  Architectures: 'standard' or 'advanced'")
            arch = input("Choose architecture (default: standard): ").strip() or 'standard'
            epochs = input("Number of epochs (default: 15): ").strip() or '15'
            run_image_classification(epochs=int(epochs), architecture=arch)
            break
            
        elif choice == '2':
            print("\nNLP Sentiment Analysis Options:")
            print("  Architectures: 'lstm', 'bidirectional_lstm', or 'gru'")
            arch = input("Choose architecture (default: bidirectional_lstm): ").strip() or 'bidirectional_lstm'
            epochs = input("Number of epochs (default: 10): ").strip() or '10'
            run_nlp_sentiment(epochs=int(epochs), architecture=arch)
            break
            
        elif choice == '3':
            run_both_models()
            break
            
        elif choice == '4':
            print("\nExiting... Goodbye! 👋")
            sys.exit(0)
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run Deep Learning Models')
    parser.add_argument('--model', type=str, choices=['image', 'nlp', 'both'],
                       help='Model to run: image, nlp, or both')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--architecture', type=str,
                       help='Model architecture variant')
    
    args = parser.parse_args()
    
    if args.model is None:
        # Interactive mode
        interactive_mode()
    else:
        # Command-line mode
        print_banner()
        
        if args.model == 'image':
            arch = args.architecture or 'standard'
            run_image_classification(epochs=args.epochs, architecture=arch)
        
        elif args.model == 'nlp':
            arch = args.architecture or 'bidirectional_lstm'
            run_nlp_sentiment(epochs=args.epochs, architecture=arch)
        
        elif args.model == 'both':
            run_both_models()


if __name__ == "__main__":
    main()
