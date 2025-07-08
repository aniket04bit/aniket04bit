import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from svtrv2_encoder import SVTRv2LNConvTwo33
from rctc_decoder import RCTCDecoder, CTCLoss


class SVTRv2Model(nn.Module):
    """
    Complete SVTRv2 model with RCTC decoder for scene text recognition.
    
    This model follows the PaddleOCR configuration:
    - SVTRv2LNConvTwo33 encoder with specified dimensions and mixer patterns
    - RCTCDecoder for CTC-based text recognition
    - Support for training and inference modes
    """
    def __init__(
        self,
        # Image input configuration
        img_size: Tuple[int, int] = (32, 128),
        in_channels: int = 3,
        
        # Encoder configuration (from YAML)
        encoder_dims: List[int] = [128, 256, 384],
        encoder_depths: List[int] = [6, 6, 6],
        encoder_num_heads: List[int] = [4, 8, 12],
        encoder_mixer: List[List[str]] = [
            ['Conv','Conv','Conv','Conv','Conv','Conv'],
            ['Conv','Conv','FGlobal','Global','Global','Global'],
            ['Global','Global','Global','Global','Global','Global']
        ],
        encoder_local_k: List[List[int]] = [[5, 5], [5, 5], [-1, -1]],
        encoder_sub_k: List[List[int]] = [[1, 1], [2, 1], [-1, -1]],
        use_pos_embed: bool = False,
        
        # Decoder configuration
        num_classes: int = 6625,  # Character dictionary size
        max_text_length: int = 25,
        decoder_hidden_channels: int = 256,
        decoder_sequence_hidden_size: int = 256,
        decoder_sequence_layers: int = 2,
        decoder_num_residual_blocks: int = 2,
        dropout: float = 0.1,
        
        # Training configuration
        training: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_text_length = max_text_length
        self.training_mode = training
        
        # SVTRv2 Encoder
        self.encoder = SVTRv2LNConvTwo33(
            img_size=img_size,
            in_channels=in_channels,
            dims=encoder_dims,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            mixer=encoder_mixer,
            local_k=encoder_local_k,
            sub_k=encoder_sub_k,
            last_stage=False,
            feat2d=True,
            use_pos_embed=use_pos_embed
        )
        
        # RCTC Decoder
        self.decoder = RCTCDecoder(
            in_channels=encoder_dims[-1],  # Last encoder dimension
            hidden_channels=decoder_hidden_channels,
            num_classes=num_classes,
            num_residual_blocks=decoder_num_residual_blocks,
            sequence_hidden_size=decoder_sequence_hidden_size,
            sequence_layers=decoder_sequence_layers,
            dropout=dropout,
            max_text_length=max_text_length
        )
        
        # CTC Loss for training
        self.ctc_loss = CTCLoss(blank_idx=0, zero_infinity=True)
        
    def forward(self, images, targets=None):
        """
        Forward pass through the model
        
        Args:
            images: Input images [B, C, H, W]
            targets: Target sequences for training [B, max_text_length] (optional)
            
        Returns:
            Dict containing:
            - 'logits': Character predictions [B, T, num_classes]
            - 'loss': CTC loss if targets provided
            - 'predictions': Decoded predictions if in inference mode
        """
        # Encode features
        encoder_features = self.encoder(images)  # [B, C, H, W]
        
        # Decode to character logits
        logits = self.decoder(encoder_features)  # [B, T, num_classes]
        
        result = {'logits': logits}
        
        # Compute loss if targets are provided (training mode)
        if targets is not None and self.training_mode:
            input_lengths = self.decoder.get_input_lengths(logits)
            target_lengths = (targets != 0).sum(dim=1)
            
            loss = self.ctc_loss(logits, targets, input_lengths, target_lengths)
            result['loss'] = loss
        
        # Generate predictions if in inference mode
        if not self.training_mode:
            with torch.no_grad():
                predictions = self.decode_predictions(logits)
                result['predictions'] = predictions
        
        return result
    
    def decode_predictions(self, logits, method='greedy'):
        """
        Decode CTC logits to text predictions
        
        Args:
            logits: Character logits [B, T, num_classes]
            method: Decoding method ('greedy' or 'beam_search')
            
        Returns:
            List of decoded sequences for each batch item
        """
        if method == 'greedy':
            return self._greedy_decode(logits)
        else:
            raise NotImplementedError(f"Decoding method '{method}' not implemented")
    
    def _greedy_decode(self, logits):
        """Greedy CTC decoding"""
        # Get most likely characters
        predictions = torch.argmax(logits, dim=-1)  # [B, T]
        
        decoded_sequences = []
        for batch_idx in range(predictions.size(0)):
            sequence = predictions[batch_idx].cpu().numpy()
            
            # Remove consecutive duplicates and blank tokens (assuming blank=0)
            decoded = []
            prev_char = None
            
            for char in sequence:
                if char != 0 and char != prev_char:  # Skip blanks and consecutive duplicates
                    decoded.append(int(char))
                prev_char = char
            
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def set_training_mode(self, training: bool):
        """Set training mode"""
        self.training_mode = training
        self.train(training)


class SVTRv2Config:
    """Configuration class for SVTRv2 model to match PaddleOCR settings"""
    
    # Default configuration matching the YAML file
    DEFAULT_CONFIG = {
        # Global settings
        'img_size': (32, 128),  # Height, Width
        'in_channels': 3,
        'max_text_length': 25,
        'character_dict_path': './tools/utils/EN_symbol_dict.txt',
        'use_space_char': False,
        
        # Encoder settings
        'encoder_dims': [128, 256, 384],
        'encoder_depths': [6, 6, 6],
        'encoder_num_heads': [4, 8, 12],
        'encoder_mixer': [
            ['Conv','Conv','Conv','Conv','Conv','Conv'],
            ['Conv','Conv','FGlobal','Global','Global','Global'],
            ['Global','Global','Global','Global','Global','Global']
        ],
        'encoder_local_k': [[5, 5], [5, 5], [-1, -1]],
        'encoder_sub_k': [[1, 1], [2, 1], [-1, -1]],
        'use_pos_embed': False,
        'last_stage': False,
        'feat2d': True,
        
        # Decoder settings
        'num_classes': 6625,  # Will be updated based on character dict
        'decoder_hidden_channels': 256,
        'decoder_sequence_hidden_size': 256,
        'decoder_sequence_layers': 2,
        'decoder_num_residual_blocks': 2,
        'dropout': 0.1,
        
        # Training settings
        'use_amp': True,
        'cal_metric_during_train': True,
    }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        config = cls.DEFAULT_CONFIG.copy()
        config.update(config_dict)
        return config
    
    @classmethod
    def load_character_dict(cls, dict_path: str):
        """Load character dictionary and return vocab size"""
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                characters = f.read().strip().split('\n')
            return len(characters) + 1  # +1 for blank token
        except FileNotFoundError:
            print(f"Character dict file not found: {dict_path}")
            return 6625  # Default size


def create_svtrv2_model(config: Optional[Dict] = None, character_dict_path: Optional[str] = None) -> SVTRv2Model:
    """
    Factory function to create SVTRv2 model with configuration
    
    Args:
        config: Configuration dictionary
        character_dict_path: Path to character dictionary file
        
    Returns:
        Configured SVTRv2Model instance
    """
    if config is None:
        config = SVTRv2Config.DEFAULT_CONFIG.copy()
    else:
        full_config = SVTRv2Config.DEFAULT_CONFIG.copy()
        full_config.update(config)
        config = full_config
    
    # Update vocab size based on character dictionary
    if character_dict_path:
        config['num_classes'] = SVTRv2Config.load_character_dict(character_dict_path)
    
    model = SVTRv2Model(
        img_size=config['img_size'],
        in_channels=config['in_channels'],
        encoder_dims=config['encoder_dims'],
        encoder_depths=config['encoder_depths'],
        encoder_num_heads=config['encoder_num_heads'],
        encoder_mixer=config['encoder_mixer'],
        encoder_local_k=config['encoder_local_k'],
        encoder_sub_k=config['encoder_sub_k'],
        use_pos_embed=config['use_pos_embed'],
        num_classes=config['num_classes'],
        max_text_length=config['max_text_length'],
        decoder_hidden_channels=config['decoder_hidden_channels'],
        decoder_sequence_hidden_size=config['decoder_sequence_hidden_size'],
        decoder_sequence_layers=config['decoder_sequence_layers'],
        decoder_num_residual_blocks=config['decoder_num_residual_blocks'],
        dropout=config['dropout'],
        training=True
    )
    
    return model


if __name__ == "__main__":
    # Test the complete model
    print("Testing SVTRv2 Model...")
    
    # Create model with default configuration
    model = create_svtrv2_model()
    
    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 32, 128)
    targets = torch.randint(1, 1000, (batch_size, 25))  # Random targets
    
    print(f"Input images shape: {images.shape}")
    print(f"Target sequences shape: {targets.shape}")
    
    # Training mode
    model.set_training_mode(True)
    train_output = model(images, targets)
    
    print("\nTraining mode output:")
    print(f"Logits shape: {train_output['logits'].shape}")
    print(f"Loss: {train_output['loss'].item():.4f}")
    
    # Inference mode
    model.set_training_mode(False)
    with torch.no_grad():
        inference_output = model(images)
    
    print("\nInference mode output:")
    print(f"Logits shape: {inference_output['logits'].shape}")
    print(f"Predictions: {inference_output['predictions']}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with different input sizes
    print(f"\nTesting different input sizes...")
    test_sizes = [(32, 64), (32, 128), (32, 256)]
    
    for h, w in test_sizes:
        test_input = torch.randn(1, 3, h, w)
        with torch.no_grad():
            output = model(test_input)
        print(f"Input size ({h}, {w}) -> Output logits shape: {output['logits'].shape}")