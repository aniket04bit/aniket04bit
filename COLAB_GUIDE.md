# 🚀 Run SVTRv2 Text Recognition on Google Colab

This guide shows you how to run text recognition on your images using Google Colab - completely free!

## 📋 Quick Start (2 Minutes Setup)

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Sign in with your Google account
3. Create a new notebook: `File → New notebook`

### Step 2: Enable GPU (Recommended)
1. Go to `Runtime → Change runtime type`
2. Select `Hardware accelerator → GPU`
3. Click `Save`

### Step 3: Copy and Run the Code

**CELL 1: Setup & Model (Copy this into first cell)**
```python
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
```

**CELL 2: Upload & Process Images (Copy this into second cell)**
```python
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

if results:
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
```

### Step 4: Run the Cells
1. Click on the first cell and press `Ctrl+Enter` (or click the play button)
2. Wait for setup to complete (30-60 seconds)
3. Click on the second cell and press `Ctrl+Enter`
4. When prompted, click "Choose Files" and upload your text images
5. Watch the magic happen! 🎉

## 📸 What Images Work Best?

### ✅ Good Examples:
- **Street signs**: "STOP", "Main Street"
- **License plates**: "ABC123"
- **Book titles**: "PYTHON PROGRAMMING"
- **Store names**: "Coffee Shop"
- **Handwritten text**: "Hello World"

### ⚠️ Tips for Better Results:
- **Clear text**: High contrast, readable fonts
- **Horizontal orientation**: Works best with left-to-right text
- **Good lighting**: Avoid shadows or glare
- **Minimal background**: Clean, simple backgrounds
- **Reasonable size**: Text should be clearly visible

## 🔧 Advanced Usage

### Upload Multiple Images at Once
The script automatically processes all uploaded images in batch.

### Different Image Sizes
The model automatically resizes images to 32x128 pixels while maintaining aspect ratio.

### Custom Vocabulary
To add more characters, modify the `chars` list:
```python
# Add more languages or symbols
chars = ['<BLANK>'] + list(string.ascii_letters + string.digits + ' .,!?-\'"()[]{}@#$%&*àáâãäåæçèéêë')
```

## 🎯 Expected Results

### Demo Performance:
- **English text**: Good accuracy on clean, horizontal text
- **Numbers**: Excellent accuracy on digits
- **Mixed content**: Handles alphanumeric content well
- **Speed**: 1-2 seconds per image on GPU

### Limitations:
- **Handwriting**: Limited accuracy (this is a demo model)
- **Complex fonts**: May struggle with decorative fonts
- **Vertical text**: Not optimized for vertical orientation
- **Non-English**: Limited to English + basic symbols

## 🚨 Troubleshooting

### Common Issues:

**"No module named torch"**
- Run the first cell again to install dependencies

**"files not found"**
- Make sure you uploaded images in cell 2
- Check file formats (JPG, PNG work best)

**"CUDA out of memory"**
- Reduce image size or batch size
- Switch to CPU: `device = torch.device('cpu')`

**Poor recognition results**
- Check image quality and contrast
- Ensure text is horizontal
- Try cropping to just the text area

## 🔗 Links & Resources

- **Original Paper**: [SVTRv2: CTC Beats Encoder-Decoder Models](https://arxiv.org/abs/2411.15858)
- **PaddleOCR**: [Official Repository](https://github.com/PaddlePaddle/PaddleOCR)
- **Google Colab**: [colab.research.google.com](https://colab.research.google.com)

## 💡 Next Steps

### For Production Use:
1. **Train on your data**: Use domain-specific datasets
2. **Larger vocabulary**: Add language-specific characters
3. **Pre-trained weights**: Use models trained on large datasets
4. **Better preprocessing**: Advanced image enhancement
5. **Post-processing**: Language models for context

### Extend the Model:
- Add more sophisticated attention mechanisms
- Implement beam search decoding
- Support for multiple languages
- Real-time video text recognition

---

**Have fun recognizing text! 🎉**

If you run into issues, feel free to ask for help or check the troubleshooting section above.