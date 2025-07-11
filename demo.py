#!/usr/bin/env python3
"""
Demo script for PP-OCRv3 Text Recognition Model
This script demonstrates the basic functionality of the model.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pp_ocrv3_rec import pp_ocrv3_rec_english
from utils.postprocess import greedy_decode


def create_synthetic_text_image(text, img_size=(48, 320), font_size=32):
    """Create a synthetic text image for testing."""
    img = Image.new('RGB', (img_size[1], img_size[0]), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fall back to default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Get text size
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width = len(text) * 10
        text_height = 15
    
    # Center the text
    x = (img_size[1] - text_width) // 2
    y = (img_size[0] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    return img


def test_model_forward():
    """Test basic model forward pass."""
    print("Testing model forward pass...")
    
    # Create model
    model = pp_ocrv3_rec_english()
    model.eval()
    
    # Create synthetic input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 48, 320)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    if isinstance(output, tuple):
        features, predictions = output
        print(f"Features shape: {features.shape}")
        print(f"Predictions shape: {predictions.shape}")
    else:
        predictions = output
        print(f"Predictions shape: {predictions.shape}")
    
    # Test decoding
    decoded = greedy_decode(predictions)
    print(f"Decoded sequences length: {[len(seq) for seq in decoded]}")
    
    print("✓ Model forward pass test passed!")
    return True


def test_text_generation_and_recognition():
    """Test with synthetic text images."""
    print("\nTesting text generation and recognition...")
    
    # Create model
    model = pp_ocrv3_rec_english()
    model.eval()
    
    # Test texts
    test_texts = ["HELLO", "WORLD", "PYTORCH", "OCR", "123", "ABC123"]
    
    results = []
    
    for text in test_texts:
        # Create synthetic image
        img = create_synthetic_text_image(text)
        
        # Convert to tensor
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        # Normalize (simple normalization)
        img_array = (img_array - 0.5) / 0.5
        input_tensor = torch.tensor(img_array).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        if isinstance(output, tuple):
            predictions = output[1]
        else:
            predictions = output
        
        # Decode
        decoded = greedy_decode(predictions)
        
        # Convert to characters (simple mapping for demo)
        if len(decoded[0]) > 0:
            # This is a simplified character mapping for demo
            # In practice, you'd use the actual character dictionary
            pred_chars = []
            for idx in decoded[0]:
                if idx < len(model.character_list):
                    pred_chars.append(model.character_list[idx])
            pred_text = ''.join(pred_chars)
        else:
            pred_text = ""
        
        results.append((text, pred_text))
        print(f"Ground truth: '{text}' -> Prediction: '{pred_text}'")
    
    print("✓ Text generation and recognition test completed!")
    return results


def visualize_model_architecture():
    """Visualize model architecture information."""
    print("\nModel Architecture Information:")
    print("-" * 50)
    
    model = pp_ocrv3_rec_english()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Print model structure
    print("\nModel Structure:")
    print(model)
    
    return model


def benchmark_model_speed():
    """Benchmark model inference speed."""
    print("\nBenchmarking model speed...")
    
    model = pp_ocrv3_rec_english()
    model.eval()
    
    # Warm up
    dummy_input = torch.randn(1, 3, 48, 320)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    import time
    
    num_runs = 100
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, 48, 320)
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = batch_size / avg_time
        
        print(f"Batch size {batch_size}: {avg_time*1000:.2f} ms/batch, {throughput:.2f} images/sec")
    
    print("✓ Speed benchmark completed!")


def create_demo_visualization():
    """Create a visualization of the demo results."""
    print("\nCreating demo visualization...")
    
    # Create synthetic images with different texts
    test_texts = ["HELLO", "WORLD", "PyTorch", "OCR2024"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    model = pp_ocrv3_rec_english()
    model.eval()
    
    for i, text in enumerate(test_texts):
        # Create image
        img = create_synthetic_text_image(text, font_size=24)
        
        # Show original image
        axes[i].imshow(img)
        axes[i].set_title(f"Input: '{text}'")
        axes[i].axis('off')
        
        # Process with model (simplified)
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5
        input_tensor = torch.tensor(img_array).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        if isinstance(output, tuple):
            predictions = output[1]
        else:
            predictions = output
        
        # Visualize predictions (heatmap)
        pred_array = predictions.squeeze().numpy()
        
        axes[i + 4].imshow(pred_array.T, aspect='auto', cmap='viridis')
        axes[i + 4].set_title(f"Predictions\nShape: {pred_array.shape}")
        axes[i + 4].set_xlabel('Time steps')
        axes[i + 4].set_ylabel('Character classes')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Demo visualization saved as 'demo_results.png'")


def main():
    parser = argparse.ArgumentParser(description='PP-OCRv3 Demo Script')
    parser.add_argument('--test', nargs='*', 
                       choices=['forward', 'text', 'arch', 'speed', 'viz'],
                       default=['forward', 'text', 'arch'],
                       help='Tests to run')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=" * 60)
    print("PP-OCRv3 Text Recognition Model Demo")
    print("=" * 60)
    
    try:
        if 'forward' in args.test:
            test_model_forward()
        
        if 'text' in args.test:
            test_text_generation_and_recognition()
        
        if 'arch' in args.test:
            visualize_model_architecture()
        
        if 'speed' in args.test:
            benchmark_model_speed()
        
        if 'viz' in args.test:
            create_demo_visualization()
        
        print("\n" + "=" * 60)
        print("✓ All demo tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())