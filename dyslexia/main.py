import argparse
import torch
from models.vit_model import DyslexiaViT
from src.data_processing import get_data_loaders
from src.training import train_model
from src.evaluation import evaluate_model, plot_confusion_matrix, visualize_attention

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DyslexiaViT(num_classes=2, pretrained=True)
    
    if args.mode == 'train':
        # Get data loaders
        train_loader, val_loader = get_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Train model
        best_model_state = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
        
        # Load best model
        model.load_state_dict(best_model_state)
        
    elif args.mode == 'evaluate':
        # Load trained model
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)
        
        # Get test loader
        _, test_loader = get_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device)
        print("\nClassification Report:")
        print(results['classification_report'])
        
        # Plot confusion matrix
        plot_confusion_matrix(results['confusion_matrix'])
        
    elif args.mode == 'visualize':
        # Load trained model
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)
        
        # Visualize attention
        visualize_attention(model, args.image_path, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dyslexia Detection using Vision Transformer')
    
    # Common arguments
    parser.add_argument('--mode', type=str, required=True,
                      choices=['train', 'evaluate', 'visualize'],
                      help='Mode to run the script in')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training/evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for training')
    
    # Evaluation/Visualization arguments
    parser.add_argument('--model_path', type=str,
                      help='Path to the trained model')
    parser.add_argument('--image_path', type=str,
                      help='Path to the image for visualization')
    
    args = parser.parse_args()
    main(args) 