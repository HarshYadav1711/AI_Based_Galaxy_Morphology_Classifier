"""
Training script for galaxy morphology classifier.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from model import get_model, count_parameters
from data_loader import load_dataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, class_names):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Get unique classes present in the data
    unique_labels = sorted(set(all_labels + all_preds))
    num_classes = len(class_names)
    
    # Ensure all class indices are included (0 to num_classes-1)
    all_class_indices = list(range(num_classes))
    
    # Detailed metrics - explicitly specify labels to include all classes
    report = classification_report(all_labels, all_preds, 
                                   labels=all_class_indices,
                                   target_names=class_names, 
                                   output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=all_class_indices)
    
    return epoch_loss, epoch_acc, report, cm


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Galaxy Morphology Classifier')
    parser.add_argument('--data_dir', type=str, default='data/galaxies',
                       help='Directory containing galaxy images')
    parser.add_argument('--model', type=str, default='lightweight',
                       choices=['lightweight', 'efficient'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    print('Loading dataset...')
    train_loader, val_loader, class_names = load_dataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
    print(f'Classes: {class_names}')
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Create model
    model = get_model(model_name=args.model, num_classes=len(class_names))
    print(f'Model: {args.model}')
    print(f'Parameters: {count_parameters(model):,}')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Track learning rate for logging
    current_lr = args.lr
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    if args.resume:
        print(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
        current_lr = optimizer.param_groups[0]['lr']  # Restore learning rate
    
    # Training loop
    print('\nStarting training...')
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, report, cm = validate(model, val_loader, criterion, device, class_names)
        
        # Update learning rate
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log learning rate changes
        if current_lr != old_lr:
            print(f'Learning rate reduced from {old_lr:.6f} to {current_lr:.6f}')
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Check which classes are present in validation set
        present_classes = []
        missing_classes = []
        for class_name in class_names:
            if class_name in report:
                support = report[class_name].get('support', 0)
                if support > 0:
                    present_classes.append(class_name)
                else:
                    missing_classes.append(class_name)
        
        if missing_classes:
            print(f'\nWarning: Some classes missing in validation set: {missing_classes}')
            print(f'Present classes: {present_classes}')
        
        print('\nPer-Class Metrics:')
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                # Check if class had any samples
                support = metrics.get('support', 0)
                if support > 0:
                    print(f'{class_name}: Precision={metrics["precision"]:.4f}, '
                          f'Recall={metrics["recall"]:.4f}, '
                          f'F1={metrics["f1-score"]:.4f}, '
                          f'Support={int(support)}')
                else:
                    print(f'{class_name}: No samples in validation set')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'class_names': class_names
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'New best model saved! Val Acc: {val_acc:.4f}')
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
    
    # Final evaluation
    print('\n' + '='*50)
    print('Training Complete!')
    print('='*50)
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print('\nFinal Evaluation on Validation Set:')
    val_loss, val_acc, report, cm = validate(model, val_loader, criterion, device, class_names)
    print(f'Best Val Accuracy: {val_acc:.4f}')
    
    # Plot results
    plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))
    plot_confusion_matrix(cm, class_names, os.path.join(args.save_dir, 'confusion_matrix.png'))
    
    print(f'\nModel saved to: {os.path.join(args.save_dir, "best_model.pth")}')


if __name__ == '__main__':
    main()

