"""
Inference script for galaxy morphology classification.
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import numpy as np
from torchvision import transforms

from model import get_model


def load_model(checkpoint_path, device, model_name='lightweight'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    class_names = checkpoint.get('class_names', ['spiral', 'elliptical', 'irregular'])
    num_classes = len(class_names)
    
    model = get_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names


def preprocess_image(image_path, image_size=224):
    """Preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image


def predict(model, image_tensor, device, class_names):
    """Make prediction on a single image."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    # Get all class probabilities
    all_probs = probabilities[0].cpu().numpy()
    class_probs = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}
    
    return predicted_class, confidence_score, class_probs


def predict_batch(model, image_paths, device, class_names, image_size=224):
    """Make predictions on multiple images."""
    results = []
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
            all_probs = probabilities[0].cpu().numpy()
            class_probs = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}
            
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'probabilities': class_probs
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Galaxy Morphology Classifier Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--model', type=str, default='lightweight',
                       choices=['lightweight', 'efficient'],
                       help='Model architecture')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save predictions (CSV format)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model, class_names = load_model(args.checkpoint, device, args.model)
    print(f'Model loaded. Classes: {class_names}')
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file {args.image} not found.")
            return
        
        print(f'\nPredicting on: {args.image}')
        image_tensor, original_image = preprocess_image(args.image, args.image_size)
        predicted_class, confidence, class_probs = predict(model, image_tensor, device, class_names)
        
        print(f'\nPrediction: {predicted_class}')
        print(f'Confidence: {confidence:.4f}')
        print('\nClass Probabilities:')
        for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
            print(f'  {class_name}: {prob:.4f}')
    
    # Batch prediction
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory {args.image_dir} not found.")
            return
        
        # Find all images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_paths = [os.path.join(args.image_dir, f) 
                      for f in os.listdir(args.image_dir)
                      if f.lower().endswith(image_extensions)]
        
        if len(image_paths) == 0:
            print(f"No images found in {args.image_dir}")
            return
        
        print(f'\nProcessing {len(image_paths)} images...')
        results = predict_batch(model, image_paths, device, class_names, args.image_size)
        
        # Print results
        print('\nPredictions:')
        print('-' * 80)
        for result in results:
            if 'error' in result:
                print(f"{result['image_path']}: ERROR - {result['error']}")
            else:
                print(f"{os.path.basename(result['image_path'])}: "
                      f"{result['predicted_class']} (confidence: {result['confidence']:.4f})")
        
        # Save to file if requested
        if args.output:
            import pandas as pd
            df_data = []
            for result in results:
                if 'error' not in result:
                    row = {
                        'image': os.path.basename(result['image_path']),
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence']
                    }
                    row.update({f'prob_{k}': v for k, v in result['probabilities'].items()})
                    df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(args.output, index=False)
            print(f'\nResults saved to {args.output}')
    
    else:
        print("Please provide either --image or --image_dir for prediction.")


if __name__ == '__main__':
    main()

