import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """Residual block for feature enhancement in RCTC decoder"""
    def __init__(self, in_channels: int, hidden_channels: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
            
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Ensure input/output channels match for residual connection
        self.identity = nn.Identity() if in_channels == in_channels else nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.identity(residual)
        out = self.relu(out)
        
        return out


class SequenceEncoder(nn.Module):
    """Sequence encoder with BiLSTM for CTC processing"""
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [B, T, C]
        output, _ = self.bilstm(x)
        output = self.dropout(output)
        return output  # [B, T, 2*hidden_size]


class RCTCDecoder(nn.Module):
    """
    Residual CTC Decoder that combines residual blocks with CTC for text recognition.
    
    The RCTC decoder processes 2D feature maps from the encoder, applies residual 
    feature enhancement, converts to sequence format, and outputs character predictions
    for CTC loss computation.
    """
    def __init__(
        self,
        in_channels: int = 384,  # From SVTRv2 last stage
        hidden_channels: int = 256,
        num_classes: int = 6625,  # Vocabulary size for Chinese characters + English
        num_residual_blocks: int = 2,
        sequence_hidden_size: int = 256,
        sequence_layers: int = 2,
        dropout: float = 0.1,
        max_text_length: int = 25
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.max_text_length = max_text_length
        
        # Input projection to reduce channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks for feature enhancement
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, dropout)
            for _ in range(num_residual_blocks)
        ])
        
        # Adaptive pooling to ensure consistent sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool height to 1, keep width
        
        # Sequence encoder (BiLSTM)
        self.sequence_encoder = SequenceEncoder(
            input_size=hidden_channels,
            hidden_size=sequence_hidden_size,
            num_layers=sequence_layers,
            dropout=dropout
        )
        
        # Output projection for CTC
        self.output_proj = nn.Linear(
            sequence_hidden_size * 2,  # Bidirectional LSTM
            num_classes
        )
        
        # Additional residual connection for the final output
        self.output_residual = nn.Linear(hidden_channels, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def _feature_to_sequence(self, x):
        """Convert 2D feature map to sequence format for CTC"""
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Apply adaptive pooling to reduce height to 1
        x = self.adaptive_pool(x)  # [B, C, 1, W]
        
        # Reshape to sequence format
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W, C]
        
        return x
    
    def forward(self, x):
        """
        Forward pass of RCTC decoder
        
        Args:
            x: Feature map from encoder [B, C, H, W]
            
        Returns:
            logits: Character predictions for CTC [B, T, num_classes]
        """
        # Input projection
        x = self.input_proj(x)  # [B, hidden_channels, H, W]
        
        # Store for residual connection
        residual_input = x
        
        # Apply residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Add residual connection
        x = x + residual_input
        
        # Convert to sequence format
        seq_features = self._feature_to_sequence(x)  # [B, W, hidden_channels]
        
        # Apply sequence encoder (BiLSTM)
        seq_output = self.sequence_encoder(seq_features)  # [B, W, 2*sequence_hidden_size]
        
        # Apply dropout
        seq_output = self.dropout(seq_output)
        
        # Main output projection
        main_logits = self.output_proj(seq_output)  # [B, W, num_classes]
        
        # Residual connection: project original sequence features to output space
        residual_logits = self.output_residual(seq_features)  # [B, W, num_classes]
        
        # Combine main and residual outputs
        logits = main_logits + residual_logits
        
        return logits
    
    def get_targets_lengths(self, targets):
        """Get target lengths for CTC loss computation"""
        if targets is None:
            return None
        # Assuming targets is a padded tensor, find actual lengths
        lengths = []
        for target in targets:
            # Find first padding token (assuming 0 is padding)
            nonzero = torch.nonzero(target).squeeze(-1)
            if len(nonzero) > 0:
                lengths.append(len(nonzero))
            else:
                lengths.append(1)  # Minimum length of 1
        return torch.tensor(lengths, device=targets.device)
    
    def get_input_lengths(self, logits):
        """Get input sequence lengths for CTC loss computation"""
        batch_size, seq_len, _ = logits.shape
        # All sequences have the same length after processing
        return torch.full((batch_size,), seq_len, device=logits.device, dtype=torch.long)


class CTCLoss(nn.Module):
    """CTC Loss wrapper for RCTC decoder"""
    def __init__(self, blank_idx: int = 0, reduction: str = 'mean', zero_infinity: bool = True):
        super().__init__()
        self.blank_idx = blank_idx
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        
    def forward(self, logits, targets, input_lengths=None, target_lengths=None):
        """
        Compute CTC loss
        
        Args:
            logits: [B, T, C] - output from decoder
            targets: [B, S] - target sequences (can be different lengths)
            input_lengths: [B] - length of each sequence in logits
            target_lengths: [B] - length of each target sequence
        """
        # Transpose for CTC: [T, B, C]
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        
        if input_lengths is None:
            batch_size, seq_len, _ = logits.shape
            input_lengths = torch.full((batch_size,), seq_len, device=logits.device, dtype=torch.long)
            
        if target_lengths is None and targets is not None:
            # Calculate target lengths (assuming 0 is padding)
            target_lengths = (targets != 0).sum(dim=1)
            
        # Flatten targets for CTC
        if targets is not None:
            targets = targets.view(-1)
            # Remove padding tokens
            targets = targets[targets != 0]
        
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.blank_idx,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity
        )
        
        return loss


if __name__ == "__main__":
    # Test the RCTC decoder
    batch_size = 2
    in_channels = 384
    height = 8
    width = 32
    num_classes = 1000
    
    # Create sample input (from encoder)
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create decoder
    decoder = RCTCDecoder(
        in_channels=in_channels,
        num_classes=num_classes
    )
    
    # Forward pass
    logits = decoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Test CTC loss
    targets = torch.randint(1, num_classes, (batch_size, 10))  # Random targets
    ctc_loss = CTCLoss()
    
    input_lengths = decoder.get_input_lengths(logits)
    target_lengths = (targets != 0).sum(dim=1)
    
    loss = ctc_loss(logits, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss.item()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"RCTC Decoder parameters: {total_params:,}")