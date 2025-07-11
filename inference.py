import os
import sys
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import pp_ocrv3_rec_english
from utils.transform import resize_norm_img
from utils.postprocess import CTCLabelDecode, greedy_decode, beam_search_decode


def parse_args():
    parser = argparse.ArgumentParser(description='PP-OCRv3 Text Recognition Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--character_dict', type=str, help='Character dictionary file')
    parser.add_argument('--img_height', type=int, default=48, help='Image height')
    parser.add_argument('--img_width', type=int, default=320, help='Image width')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--use_beam_search', action='store_true', help='Use beam search decoding')
    parser.add_argument('--beam_width', type=int, default=10, help='Beam width for beam search')
    parser.add_argument('--save_result', type=str, help='Path to save result image')
    return parser.parse_args()


class PP_OCRv3_Predictor:
    def __init__(self, model_path, character_dict_path=None, 
                 img_size=[48, 320], device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # Build model
        self.model = pp_ocrv3_rec_english(img_size=img_size)
        
        # Load trained weights
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Build postprocessor
        self.postprocessor = CTCLabelDecode(
            character_dict_path=character_dict_path,
            use_space_char=False
        )
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Character set size: {len(self.postprocessor.character)}")

    def preprocess(self, image):
        """Preprocess input image."""
        if isinstance(image, str):
            # Load image from path
            image = cv2.imread(image)
            if image is None:
                image = np.array(Image.open(image).convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize and normalize
        norm_img, valid_ratio = resize_norm_img(image, [3] + self.img_size, padding=True)
        
        # Convert to tensor
        norm_img = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0)
        
        return norm_img, valid_ratio

    def predict(self, image, use_beam_search=False, beam_width=10):
        """Predict text from image."""
        # Preprocess
        input_tensor, valid_ratio = self.preprocess(image)
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            if self.model.head.return_feats:
                features, predictions = self.model(input_tensor)
            else:
                predictions = self.model(input_tensor)
        
        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[1]  # Get predictions from (features, predictions)
        
        # Convert to numpy
        predictions_np = predictions.detach().cpu()
        
        # Decode
        if use_beam_search:
            decoded_indices = beam_search_decode(predictions_np, beam_width=beam_width)
        else:
            decoded_indices = greedy_decode(predictions_np)
        
        # Convert indices to text
        pred_texts = []
        confidences = []
        
        for batch_idx, indices in enumerate(decoded_indices):
            if len(indices) > 0:
                # Get characters
                chars = []
                char_confidences = []
                
                for idx in indices:
                    if idx < len(self.postprocessor.character):
                        chars.append(self.postprocessor.character[idx])
                        # Get confidence from softmax probabilities
                        prob = torch.softmax(predictions_np[batch_idx], dim=-1)
                        max_conf = torch.max(prob, dim=-1)[0]
                        char_confidences.append(max_conf.mean().item())
                
                pred_text = ''.join(chars)
                avg_confidence = np.mean(char_confidences) if char_confidences else 0.0
            else:
                pred_text = ""
                avg_confidence = 0.0
            
            pred_texts.append(pred_text)
            confidences.append(avg_confidence)
        
        return pred_texts, confidences

    def visualize_result(self, image_path, pred_text, confidence, save_path=None):
        """Visualize prediction result."""
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            image = np.array(Image.open(image_path).convert('RGB'))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Show original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Show preprocessed image
        plt.subplot(1, 2, 2)
        preprocessed, _ = self.preprocess(image_path)
        preprocessed_img = preprocessed.squeeze().permute(1, 2, 0).numpy()
        # Denormalize
        preprocessed_img = (preprocessed_img * 0.5 + 0.5) * 255
        preprocessed_img = np.clip(preprocessed_img, 0, 255).astype(np.uint8)
        plt.imshow(preprocessed_img)
        plt.title(f'Preprocessed Image\nPrediction: "{pred_text}"\nConfidence: {confidence:.3f}')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Result saved to {save_path}")
        
        plt.show()


def main():
    args = parse_args()
    
    # Create predictor
    predictor = PP_OCRv3_Predictor(
        model_path=args.model_path,
        character_dict_path=args.character_dict,
        img_size=[args.img_height, args.img_width],
        device=args.device
    )
    
    # Predict
    print(f"Processing image: {args.image_path}")
    pred_texts, confidences = predictor.predict(
        args.image_path,
        use_beam_search=args.use_beam_search,
        beam_width=args.beam_width
    )
    
    # Print results
    for i, (text, conf) in enumerate(zip(pred_texts, confidences)):
        print(f"Prediction {i+1}: '{text}' (confidence: {conf:.3f})")
    
    # Visualize result
    if len(pred_texts) > 0:
        predictor.visualize_result(
            args.image_path, 
            pred_texts[0], 
            confidences[0],
            save_path=args.save_result
        )


def batch_inference(model_path, image_folder, output_file, character_dict=None):
    """Batch inference on a folder of images."""
    predictor = PP_OCRv3_Predictor(
        model_path=model_path,
        character_dict_path=character_dict
    )
    
    results = []
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            pred_texts, confidences = predictor.predict(image_path)
            if len(pred_texts) > 0:
                results.append({
                    'image': image_file,
                    'prediction': pred_texts[0],
                    'confidence': confidences[0]
                })
                print(f"{image_file}: '{pred_texts[0]}' ({confidences[0]:.3f})")
            else:
                results.append({
                    'image': image_file,
                    'prediction': '',
                    'confidence': 0.0
                })
                print(f"{image_file}: No prediction")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results.append({
                'image': image_file,
                'prediction': 'ERROR',
                'confidence': 0.0
            })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("image\tprediction\tconfidence\n")
        for result in results:
            f.write(f"{result['image']}\t{result['prediction']}\t{result['confidence']:.3f}\n")
    
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()