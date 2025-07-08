# SVTRv2 with RCTC Decoder - PyTorch Implementation

A PyTorch implementation of SVTRv2 (Scene Text Recognition with Vision Transformer v2) with RCTC (Residual CTC) decoder, converted from the PaddleOCR configuration.

## Overview

This implementation provides a complete scene text recognition model based on the SVTRv2 architecture with the following key components:

- **SVTRv2LNConvTwo33 Encoder**: A hierarchical vision transformer with mixed attention mechanisms (Conv, FGlobal, Global)
- **RCTC Decoder**: Residual CTC decoder that enhances feature representation before CTC alignment
- **Modular Design**: Easy to customize and extend for different datasets and use cases

## Architecture Details

### SVTRv2 Encoder

The encoder follows a hierarchical design with three stages:

| Stage | Dimensions | Depth | Heads | Mixer Pattern |
|-------|------------|-------|-------|---------------|
| 1 | 128 | 6 | 4 | Conv×6 |
| 2 | 256 | 6 | 8 | Conv×2, FGlobal×1, Global×3 |
| 3 | 384 | 6 | 12 | Global×6 |

**Key Features:**
- **Conv Mixer**: Local feature interaction using depthwise convolutions
- **FGlobal Mixer**: Fast global attention with spatial reduction
- **Global Mixer**: Full self-attention for long-range dependencies
- **Hierarchical Processing**: Progressive feature refinement across stages

### RCTC Decoder

The RCTC (Residual CTC) decoder enhances the standard CTC approach with:

- **Residual Blocks**: Feature enhancement through residual connections
- **Sequence Encoder**: Bidirectional LSTM for temporal modeling
- **Dual Output Paths**: Main path + residual path for improved gradient flow
- **Adaptive Pooling**: Converts 2D features to sequence format

## File Structure

```
├── svtrv2_encoder.py      # SVTRv2LNConvTwo33 encoder implementation
├── rctc_decoder.py        # RCTC decoder and CTC loss implementation
├── svtrv2_model.py        # Complete model combining encoder + decoder
├── demo.py                # Demo script with training/inference examples
└── README.md              # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd svtrv2-pytorch

# Install dependencies
pip install torch torchvision numpy pillow matplotlib

# Optional: Install additional dependencies for demo
pip install opencv-python
```

## Quick Start

### 1. Basic Model Usage

```python
from svtrv2_model import create_svtrv2_model
import torch

# Create model with default configuration
model = create_svtrv2_model()

# Create sample input (batch_size=2, channels=3, height=32, width=128)
images = torch.randn(2, 3, 32, 128)

# Training mode
model.set_training_mode(True)
targets = torch.randint(1, 1000, (2, 25))  # Random target sequences
output = model(images, targets)
print(f"Training loss: {output['loss']}")

# Inference mode
model.set_training_mode(False)
with torch.no_grad():
    output = model(images)
    predictions = output['predictions']
    print(f"Predictions: {predictions}")
```

### 2. Custom Configuration

```python
# Custom model configuration
config = {
    'encoder_dims': [64, 128, 256],      # Smaller model
    'encoder_depths': [4, 4, 4],         # Fewer layers
    'num_classes': 1000,                 # Custom vocabulary size
    'max_text_length': 20,               # Shorter sequences
}

model = create_svtrv2_model(config)
```

### 3. Run Demo

```bash
# Run demo with synthetic data
python demo.py --mode demo --num_epochs 5 --batch_size 16

# Training on custom dataset
python demo.py --mode train --data_path /path/to/dataset

# Inference with trained model
python demo.py --mode inference --model_path /path/to/model.pth
```

## Model Configuration

The model supports extensive customization through configuration parameters:

### Image Input
- `img_size`: Input image dimensions (default: (32, 128))
- `in_channels`: Input channels (default: 3)

### Encoder Settings
- `encoder_dims`: Channel dimensions for each stage
- `encoder_depths`: Number of blocks per stage
- `encoder_num_heads`: Number of attention heads per stage
- `encoder_mixer`: Mixer patterns for each block
- `encoder_local_k`: Local kernel sizes for conv mixers
- `encoder_sub_k`: Spatial reduction ratios
- `use_pos_embed`: Whether to use positional embeddings

### Decoder Settings
- `num_classes`: Vocabulary size (number of characters)
- `max_text_length`: Maximum text sequence length
- `decoder_hidden_channels`: Hidden dimensions in decoder
- `decoder_sequence_hidden_size`: LSTM hidden size
- `decoder_sequence_layers`: Number of LSTM layers
- `decoder_num_residual_blocks`: Number of residual blocks

## Training

### Basic Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_svtrv2_model(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.00065, weight_decay=0.05)

# Training loop
model.set_training_mode(True)
for epoch in range(num_epochs):
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(images, targets)
        loss = output['loss']
        loss.backward()
        optimizer.step()
```

### Recommended Training Settings

Based on the original PaddleOCR configuration:

- **Optimizer**: AdamW with lr=0.00065, weight_decay=0.05
- **Scheduler**: OneCycleLR with warmup
- **Batch Size**: 256 per GPU (adjust based on memory)
- **Mixed Precision**: Recommended for faster training
- **Gradient Clipping**: Optional, helps with stability

## Inference

### Basic Inference

```python
model.set_training_mode(False)
model.eval()

with torch.no_grad():
    images = preprocess_images(image_paths)  # Your preprocessing
    output = model(images)
    predictions = output['predictions']
    
    # Convert predictions to text
    texts = decode_predictions(predictions, idx_to_char)
```

### CTC Decoding

The model includes greedy CTC decoding by default. For better results, you can implement:

- **Beam Search**: For better accuracy (not included)
- **Language Model Integration**: For context-aware decoding
- **Custom Post-processing**: For domain-specific requirements

## Model Variants

You can create different model sizes by adjusting the configuration:

### Tiny Model (Fast Inference)
```python
tiny_config = {
    'encoder_dims': [64, 128, 192],
    'encoder_depths': [3, 3, 3],
    'decoder_hidden_channels': 128,
    'decoder_sequence_hidden_size': 128,
}
```

### Large Model (High Accuracy)
```python
large_config = {
    'encoder_dims': [192, 384, 576],
    'encoder_depths': [8, 8, 8],
    'decoder_hidden_channels': 512,
    'decoder_sequence_hidden_size': 512,
}
```

## Performance Considerations

### Memory Usage
- The model uses attention mechanisms that scale quadratically with sequence length
- Use gradient checkpointing for training with limited memory
- Consider reducing image width for longer sequences

### Speed Optimization
- Use mixed precision training (AMP)
- Optimize batch size for your hardware
- Consider TensorRT for inference optimization

### Accuracy Tips
- Use appropriate data augmentation
- Fine-tune on domain-specific data
- Experiment with different mixer patterns
- Consider ensemble methods for critical applications

## Differences from PaddleOCR

This PyTorch implementation maintains the same architectural principles as the original PaddleOCR version while adapting to PyTorch conventions:

1. **Framework Differences**: PyTorch-style modules and training loops
2. **Implementation Details**: Some low-level optimizations may differ
3. **Default Parameters**: Adjusted for typical PyTorch workflows
4. **Additional Features**: More flexible configuration options

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Beam search CTC decoding
- [ ] Additional mixer types
- [ ] Pre-trained model weights
- [ ] Data loading utilities
- [ ] Evaluation metrics
- [ ] Export to ONNX/TensorRT

## License

This implementation is provided for research and educational purposes. Please refer to the original PaddleOCR license for commercial usage guidelines.

## Citation

If you use this implementation, please cite the original SVTRv2 paper:

```bibtex
@article{du2024svtrv2,
    title={SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
    author={Du, Yongkun and Chen, Zhineng and Xie, Hongtao and Jia, Caiyan and Jiang, Yu-Gang},
    journal={arXiv preprint arXiv:2411.15858},
    year={2024}
}
```

## Acknowledgments

- Original SVTRv2 implementation by PaddleOCR team
- Vision Transformer implementations that inspired this work
- PyTorch community for excellent documentation and tools
