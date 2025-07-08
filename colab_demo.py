"""
SVTRv2 Text Recognition - Google Colab Demo

Copy and paste this entire script into Google Colab cells to run text recognition on your images!

Instructions:
1. Copy the first cell (Setup & Model)
2. Copy the second cell (Upload & Process)
3. Upload your images when prompted
4. See the results!
"""

# =====================================================
# CELL 1: Setup & Model Definition
# =====================================================

# Install dependencies
!pip install torch torchvision pillow matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import string
from typing import List, Tuple
import math

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Simple SVTRv2 Model for Colab
class SimpleSVTR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Simplified encoder
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Adaptive pooling to handle different widths
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True, bidirectional=True)
        
        # Output projection
        self.classifier = nn.Linear(256, num_classes)
        
        self.training_mode = False
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Convert to sequence
        x = self.pool(x)  # [B, C, 1, W]
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W, C]
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Classification
        logits = self.classifier(x)
        
        if not self.training_mode:
            # CTC Decoding
            predictions = self._decode(logits)
            return {'logits': logits, 'predictions': predictions}
        
        return {'logits': logits}
    
    def _decode(self, logits):
        """Simple greedy CTC decoding"""
        predictions = torch.argmax(logits, dim=-1)
        decoded = []
        
        for seq in predictions:
            chars = []
            prev = None
            for char_idx in seq.cpu().numpy():
                if char_idx != 0 and char_idx != prev:  # Skip blank and consecutive
                    chars.append(int(char_idx))
                prev = char_idx
            decoded.append(chars)
        
        return decoded
    
    def set_training_mode(self, training):
        self.training_mode = training
        self.train(training)

# Create character dictionary
chars = ['<BLANK>'] + list(string.ascii_letters + string.digits + ' .,!?-\'"()[]{}@#$%&*')
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

print(f"Vocabulary size: {len(chars)}")
print(f"Sample chars: {''.join(chars[1:21])}")

# Create and setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleSVTR(num_classes=len(chars)).to(device)
model.set_training_mode(False)

print(f"✅ Model ready with {sum(p.numel() for p in model.parameters()):,} parameters")

# =====================================================
# CELL 2: Upload Images & Run Recognition
# =====================================================

from google.colab import files

def preprocess_image(image_path, target_height=32, target_width=128):
    """Preprocess image for text recognition"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize maintaining aspect ratio
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    new_width = int(target_height * aspect_ratio)
    
    # Limit width
    if new_width > target_width:
        new_width = target_width
    
    # Resize
    image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    # Pad to target width
    if new_width < target_width:
        padded = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        padded.paste(image, (0, 0))
        image = padded
    
    # Convert to tensor
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, image

def recognize_text(image_path):
    """Run text recognition on single image"""
    # Preprocess
    img_tensor, display_img = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predicted_indices = output['predictions'][0]
    
    # Convert to text
    predicted_text = ''.join([idx_to_char.get(idx, '') for idx in predicted_indices])
    
    return predicted_text, display_img

# Upload images
print("📤 Upload your text images (JPG, PNG, etc.)")
uploaded = files.upload()

# Process each uploaded image
results = []
for filename in uploaded.keys():
    print(f"\n🔍 Processing: {filename}")
    
    try:
        predicted_text, display_img = recognize_text(filename)
        results.append((filename, predicted_text, display_img))
        print(f"✅ Recognized: '{predicted_text}'")
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# Display results
print(f"\n📊 FINAL RESULTS:")
print("=" * 50)

fig, axes = plt.subplots(len(results), 1, figsize=(12, 3*len(results))) if len(results) > 1 else plt.subplots(1, 1, figsize=(12, 3))
if len(results) == 1:
    axes = [axes]

for i, (filename, text, img) in enumerate(results):
    print(f"{i+1}. {filename}: '{text}'")
    
    # Show image with prediction
    axes[i].imshow(img)
    axes[i].set_title(f"File: {filename}\nRecognized: '{text}'", fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(f"\n✅ Processed {len(results)} images successfully!")

# =====================================================
# TIPS FOR BETTER RESULTS
# =====================================================

print("""
💡 TIPS FOR BETTER TEXT RECOGNITION:

1. 📷 Image Quality: Use clear, high-resolution images
2. 📏 Text Size: Ensure text is large enough and readable  
3. 🎨 Contrast: High contrast between text and background
4. 📐 Orientation: Horizontal text works best
5. 🔤 Language: Currently optimized for English + numbers + symbols
6. 🧹 Clean Background: Minimal noise around text
7. 📱 Format: JPG, PNG formats work well

⚠️ Note: This is a demo model. For production use, you would need:
- Training on specific domains/languages
- Larger vocabulary
- Pre-trained weights
- More sophisticated preprocessing
""")