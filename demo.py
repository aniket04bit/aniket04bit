#!/usr/bin/env python3
"""
Demo script for SVTRv2 Scene Text Recognition Model

This script demonstrates how to use the PyTorch implementation of SVTRv2 
with RCTC decoder for scene text recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import json
from typing import List, Dict, Tuple

from svtrv2_model import create_svtrv2_model, SVTRv2Config


class TextRecognitionDataset(Dataset):
    """
    Simple dataset class for text recognition
    You can replace this with your own dataset implementation
    """
    def __init__(self, image_paths: List[str], labels: List[str], char_to_idx: Dict[str, int], 
                 max_length: int = 25, img_height: int = 32, img_width: int = 128):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        self.img_height = img_height
        self.img_width = img_width
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = image.resize((self.img_width, self.img_height))
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # CHW format
        except:
            # Create dummy image if loading fails
            image = torch.randn(3, self.img_height, self.img_width)
        
        # Encode label
        label = self.labels[idx]
        encoded_label = []
        for char in label:
            if char in self.char_to_idx:
                encoded_label.append(self.char_to_idx[char])
            else:
                encoded_label.append(self.char_to_idx.get('<UNK>', 1))  # Unknown character
        
        # Pad or truncate to max_length
        if len(encoded_label) > self.max_length:
            encoded_label = encoded_label[:self.max_length]
        else:
            encoded_label.extend([0] * (self.max_length - len(encoded_label)))  # 0 for padding
        
        return image, torch.tensor(encoded_label, dtype=torch.long)


def create_character_dict(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create character dictionary from text samples"""
    chars = set()
    for text in texts:
        chars.update(text)
    
    # Add special tokens
    char_list = ['<BLANK>', '<UNK>'] + sorted(list(chars))
    
    char_to_idx = {char: idx for idx, char in enumerate(char_list)}
    idx_to_char = {idx: char for idx, char in enumerate(char_list)}
    
    return char_to_idx, idx_to_char


def generate_synthetic_data(num_samples: int = 100) -> Tuple[List[np.ndarray], List[str]]:
    """Generate synthetic text recognition data for demo"""
    import random
    import string
    
    images = []
    labels = []
    
    # Character set for synthetic data
    chars = string.ascii_letters + string.digits
    
    for _ in range(num_samples):
        # Generate random text
        text_length = random.randint(3, 8)
        text = ''.join(random.choices(chars, k=text_length))
        labels.append(text)
        
        # Generate synthetic image (just noise for demo)
        image = np.random.rand(32, 128, 3) * 255
        images.append(image.astype(np.uint8))
    
    return images, labels


def train_model(model, train_loader, val_loader, num_epochs: int = 10, lr: float = 0.001):
    """Simple training loop for demonstration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    model.set_training_mode(True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            output = model(images, targets)
            loss = output['loss']
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}')
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    output = model(images, targets)
                    val_loss += output['loss'].item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f'Validation loss: {avg_val_loss:.4f}')
    
    return model


def inference_demo(model, test_images: List[np.ndarray], idx_to_char: Dict[int, str]):
    """Demonstration of model inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.set_training_mode(False)
    
    predictions = []
    
    with torch.no_grad():
        for image in test_images:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
                if len(image.shape) == 3:
                    image = image.permute(2, 0, 1)  # HWC to CHW
            
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            output = model(image)
            predicted_indices = output['predictions'][0]  # First (and only) item in batch
            
            # Decode to text
            predicted_text = ''.join([idx_to_char.get(idx, '<UNK>') for idx in predicted_indices])
            predictions.append(predicted_text)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='SVTRv2 Text Recognition Demo')
    parser.add_argument('--mode', choices=['train', 'inference', 'demo'], default='demo',
                        help='Mode to run: train, inference, or demo')
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running SVTRv2 Demo with Synthetic Data...")
        
        # Generate synthetic data
        images, labels = generate_synthetic_data(num_samples=200)
        
        # Create character dictionary
        char_to_idx, idx_to_char = create_character_dict(labels)
        
        print(f"Generated {len(images)} synthetic samples")
        print(f"Character vocabulary size: {len(char_to_idx)}")
        print(f"Sample labels: {labels[:5]}")
        
        # Create model
        config = {'num_classes': len(char_to_idx)}
        model = create_svtrv2_model(config)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create datasets (split 80/20 for train/val)
        split_idx = int(0.8 * len(images))
        
        # Save synthetic images temporarily for dataset
        import tempfile
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        for i, img in enumerate(images):
            img_path = os.path.join(temp_dir, f"img_{i}.png")
            Image.fromarray(img).save(img_path)
            image_paths.append(img_path)
        
        train_dataset = TextRecognitionDataset(
            image_paths[:split_idx], 
            labels[:split_idx], 
            char_to_idx
        )
        val_dataset = TextRecognitionDataset(
            image_paths[split_idx:], 
            labels[split_idx:], 
            char_to_idx
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train model
        print("\nStarting training...")
        trained_model = train_model(model, train_loader, val_loader, 
                                  num_epochs=args.num_epochs, lr=args.lr)
        
        # Test inference
        print("\nTesting inference...")
        test_images = images[split_idx:split_idx+5]  # Take 5 test samples
        test_labels = labels[split_idx:split_idx+5]
        
        predictions = inference_demo(trained_model, test_images, idx_to_char)
        
        print("\nInference Results:")
        for i, (true_label, pred_label) in enumerate(zip(test_labels, predictions)):
            print(f"Sample {i+1}: True='{true_label}' | Predicted='{pred_label}'")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
    elif args.mode == 'train':
        if not args.data_path:
            print("Please provide --data_path for training mode")
            return
        
        print("Training mode - implement your dataset loading here")
        # You would implement your actual dataset loading here
        
    elif args.mode == 'inference':
        if not args.model_path:
            print("Please provide --model_path for inference mode")
            return
        
        print("Inference mode - implement your inference pipeline here")
        # You would implement your inference pipeline here
    
    print("Demo completed!")


if __name__ == "__main__":
    main()