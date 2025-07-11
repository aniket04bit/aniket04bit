#!/usr/bin/env python3
"""
Simple example script for PP-OCRv3 Text Recognition
This script shows how to quickly test the model with minimal setup.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.pp_ocrv3_rec import pp_ocrv3_rec_english
from utils.postprocess import greedy_decode


def create_test_image(text="HELLO", size=(48, 320)):
    """Create a simple test image with text."""
    img = Image.new('RGB', (size[1], size[0]), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a better font
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (center)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width = len(text) * 10
        text_height = 20
    
    x = (size[1] - text_width) // 2
    y = (size[0] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    return img


def preprocess_image(img):
    """Simple preprocessing for the image."""
    # Convert PIL to numpy
    img_array = np.array(img)
    
    # Convert to CHW format and normalize
    img_array = img_array.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    # Normalize to [-1, 1]
    img_array = (img_array - 0.5) / 0.5
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(img_array).unsqueeze(0)
    
    return tensor


def predict_text(model, image_tensor):
    """Predict text from image tensor."""
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        output = model(image_tensor)
        
        # Handle model output
        if isinstance(output, tuple):
            predictions = output[1]  # (features, predictions)
        else:
            predictions = output
        
        # Decode predictions using greedy search
        decoded = greedy_decode(predictions)
        
        # Convert indices to characters
        if len(decoded[0]) > 0:
            chars = []
            for idx in decoded[0]:
                if idx < len(model.character_list):
                    chars.append(model.character_list[idx])
            predicted_text = ''.join(chars)
        else:
            predicted_text = ""
        
        return predicted_text


def main():
    print("🚀 PP-OCRv3 Text Recognition Example")
    print("=" * 40)
    
    # 1. Create model
    print("📦 Loading model...")
    model = pp_ocrv3_rec_english()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    print(f"📏 Character set size: {len(model.character_list)}")
    
    # 2. Test with different texts
    test_texts = ["HELLO", "WORLD", "PyTorch", "OCR", "2024", "TEST123"]
    
    print("\n🧪 Testing predictions:")
    print("-" * 40)
    
    for text in test_texts:
        # Create test image
        img = create_test_image(text)
        
        # Preprocess
        img_tensor = preprocess_image(img)
        
        # Predict
        predicted = predict_text(model, img_tensor)
        
        # Show result
        match = "✅" if predicted.upper() == text.upper() else "❌"
        print(f"{match} '{text}' -> '{predicted}'")
    
    print("\n" + "=" * 40)
    print("🎉 Example completed!")
    
    # 3. Demonstrate with custom image (if you have one)
    print("\n💡 To use your own image:")
    print("   from PIL import Image")
    print("   img = Image.open('your_image.jpg')")
    print("   img = img.resize((320, 48))  # Resize to model input")
    print("   tensor = preprocess_image(img)")
    print("   result = predict_text(model, tensor)")
    print("   print(f'Predicted: {result}')")


if __name__ == "__main__":
    main()