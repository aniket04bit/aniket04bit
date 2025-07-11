# PP-OCRv3 Text Recognition - PyTorch Implementation

A complete PyTorch implementation of the **en_PP-OCRv3_rec_slim** text recognition model from PaddleOCR. This implementation combines the lightweight LCNet backbone with SVTR (Scene Text Recognition with a Single Visual Model) transformer blocks for efficient and accurate text recognition.

## 🏗️ Architecture Overview

The PP-OCRv3 text recognition model consists of three main components:

1. **LCNet Backbone**: A lightweight CNN backbone that extracts features from input images
2. **SVTR Transformer**: Global Mix Blocks that capture long-range dependencies in text sequences
3. **CTC Head**: Connectionist Temporal Classification for sequence-to-sequence prediction without explicit alignment

### Key Features

- **Lightweight**: Optimized for mobile and edge deployment
- **High Accuracy**: Combines CNN efficiency with transformer effectiveness  
- **CTC-based**: No need for character-level alignment during training
- **Flexible**: Supports custom character dictionaries and multiple languages
- **Production-ready**: Includes training, evaluation, and inference pipelines

## 📁 Project Structure

```
pp-ocrv3-pytorch/
├── models/
│   ├── __init__.py
│   ├── lcnet.py              # LCNet backbone implementation
│   ├── svtr_components.py    # SVTR transformer components
│   ├── svtr_lcnet.py         # Combined SVTR-LCNet model
│   └── pp_ocrv3_rec.py       # Main PP-OCRv3 model with CTC head
├── utils/
│   ├── __init__.py
│   ├── transform.py          # Data preprocessing and augmentation
│   ├── postprocess.py        # CTC decoding and postprocessing
│   └── losses.py             # Loss functions including CTC loss
├── data/
│   └── dataset.py            # Dataset classes for training and evaluation
├── train.py                  # Training script
├── inference.py              # Inference script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd pp-ocrv3-pytorch
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dataset Preparation

Prepare your dataset in the following format:

**Directory structure**:
```
dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── train_labels.txt
└── val_labels.txt
```

**Label file format**:
```
img1.jpg	Hello World
img2.jpg	PyTorch OCR
img3.jpg	Text Recognition
```

### Training

**Basic training command**:
```bash
python train.py \
    --data_dir ./dataset/images \
    --train_label ./dataset/train_labels.txt \
    --val_label ./dataset/val_labels.txt \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

**Advanced training with custom settings**:
```bash
python train.py \
    --data_dir ./dataset/images \
    --train_label ./dataset/train_labels.txt \
    --val_label ./dataset/val_labels.txt \
    --character_dict ./custom_chars.txt \
    --output_dir ./output \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --img_height 48 \
    --img_width 320 \
    --device cuda \
    --num_workers 8
```

### Inference

**Single image inference**:
```bash
python inference.py \
    --model_path ./output/best_model.pth \
    --image_path ./test_image.jpg \
    --save_result ./result.png
```

**Batch inference**:
```python
from inference import batch_inference

batch_inference(
    model_path='./output/best_model.pth',
    image_folder='./test_images/',
    output_file='./results.txt'
)
```

## 🛠️ Model Architecture Details

### LCNet Backbone

The LCNet (Lightweight CNN) backbone features:
- Depthwise separable convolutions
- Hard-Swish activation functions
- Squeeze-and-Excitation (SE) modules
- Efficient architecture with multiple scaling factors

### SVTR Components

The SVTR transformer includes:
- **Global Mix Blocks**: Combine local convolution and global attention
- **Local and Global Attention**: Capture both local features and long-range dependencies
- **Position Encoding**: Learned positional embeddings for sequence modeling
- **Layer Normalization**: Pre-norm architecture for stable training

### CTC Head

The CTC (Connectionist Temporal Classification) head:
- Maps feature sequences to character probabilities
- Handles variable-length sequences without explicit alignment
- Includes blank token for handling repetitions and empty positions

## 📊 Model Variants

Three model sizes are available:

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| **Tiny** | ~2M | Fast | Good | Mobile/Edge |
| **Small** | ~5M | Medium | Better | Balanced |
| **Base** | ~12M | Slower | Best | Server |

## 🎯 Training Features

### Data Augmentation
- **TIA (Text Image Augmentation)**: Distortion and perspective changes
- **Random Cropping**: Improves robustness to different image sizes
- **Color Jittering**: Handles various lighting conditions

### Training Strategies
- **CosineAnnealingLR**: Smooth learning rate scheduling
- **AdamW Optimizer**: Better generalization than Adam
- **Gradient Clipping**: Stable training for RNN components
- **Mixed Precision**: Faster training with FP16 (optional)

### Loss Functions
- **CTC Loss**: Primary loss for sequence learning
- **Focal Loss**: Handles class imbalance (optional)
- **Distillation Loss**: Knowledge transfer from larger models

## 🔧 Customization

### Adding New Languages

1. **Prepare character dictionary**:
```
# characters.txt
a
b
c
...
```

2. **Update model configuration**:
```python
model = build_pp_ocrv3_rec_model(
    character_dict_path='./characters.txt',
    use_space_char=True  # For languages with spaces
)
```

### Custom Model Configuration

```python
# Custom backbone configuration
backbone_config = {
    'dims': 64,        # Transformer dimension
    'depths': 2,       # Number of transformer layers  
    'num_heads': 8,    # Attention heads
    'mixer': ['Global'] * 2,  # Attention types
    'img_size': [48, 320],    # Input image size
    'out_channels': 256,      # Output feature channels
}

model = PP_OCRv3_Rec(backbone_config=backbone_config)
```

## 📈 Performance Tips

### Training Optimization
1. **Batch Size**: Use largest batch size that fits in memory
2. **Learning Rate**: Start with 1e-3, reduce if loss doesn't decrease
3. **Image Size**: Larger images = better accuracy but slower training
4. **Data Quality**: Clean, properly labeled data is crucial

### Inference Optimization
1. **TensorRT**: Use TensorRT for faster GPU inference
2. **ONNX**: Convert to ONNX for cross-platform deployment
3. **Quantization**: INT8 quantization for mobile deployment
4. **Batch Inference**: Process multiple images together

## 🧪 Evaluation Metrics

The model supports various evaluation metrics:
- **Character Accuracy**: Percentage of correctly recognized characters
- **Word Accuracy**: Percentage of completely correct words
- **Edit Distance**: Levenshtein distance between prediction and ground truth
- **BLEU Score**: For sequence-to-sequence evaluation

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --batch_size 8

# Use gradient accumulation
python train.py --batch_size 16 --accumulate_grad_batches 2
```

**2. Model Not Converging**
```bash
# Check learning rate
python train.py --lr 5e-4

# Verify data format
python train.py --log_freq 10  # More frequent logging
```

**3. Poor Accuracy**
- Verify label file format
- Check character dictionary completeness  
- Increase training data diversity
- Adjust image preprocessing parameters

## 📚 References

1. **PP-OCRv3 Paper**: [PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System](https://arxiv.org/abs/2206.03001)
2. **SVTR Paper**: [Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)
3. **LCNet Paper**: [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/abs/2109.15099)
4. **CTC Paper**: [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PaddleOCR Team**: For the original PP-OCRv3 implementation
- **PyTorch Team**: For the excellent deep learning framework
- **Open Source Community**: For various components and inspirations

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](../../issues) page
2. Read the documentation thoroughly
3. Create a new issue with detailed information

---

**Happy Text Recognition! 🎉**
