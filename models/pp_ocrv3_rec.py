import torch
import torch.nn as nn
import torch.nn.functional as F
from .svtr_lcnet import SVTRLCNet


class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, return_feats=False, **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels, bias=True)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels, bias=True)
            self.fc2 = nn.Linear(mid_channels, out_channels, bias=True)

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            if self.return_feats:
                result = (x, predicts)
            else:
                result = predicts
        return result


class PP_OCRv3_Rec(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 backbone_config=None,
                 neck_config=None,
                 head_config=None,
                 **kwargs):
        super(PP_OCRv3_Rec, self).__init__()
        
        # Default configurations
        if backbone_config is None:
            backbone_config = {
                'dims': 64,
                'depths': 2,
                'num_heads': 8,
                'mixer': ['Global'] * 2,
                'img_size': [48, 320],
                'out_channels': 256,
                'out_char_num': 25,
            }
        
        if head_config is None:
            head_config = {
                'out_channels': 37,  # 26 letters + 10 digits + blank
                'mid_channels': None,
                'return_feats': True
            }
        
        # Build backbone (SVTR-LCNet)
        self.backbone = SVTRLCNet(
            in_channels=in_channels,
            **backbone_config
        )
        
        # Build neck (sequence modeling) - keep it simple for now
        self.neck = None
        if neck_config is not None:
            # Add sequence modeling layers if needed (LSTM, etc.)
            pass
        
        # Build head (CTC)
        backbone_out_channels = backbone_config.get('out_channels', 256)
        head_config['in_channels'] = backbone_out_channels
        
        self.head = CTCHead(**head_config)
        
        self.out_channels = head_config['out_channels']

    def forward(self, x, targets=None):
        # Extract features from backbone
        x = self.backbone(x)
        
        # x shape: [B, C, 1, W] where W is sequence length
        # Reshape for sequence processing
        B, C, H, W = x.shape
        assert H == 1, f"Height should be 1 after global average pooling, got {H}"
        
        # Reshape to [B, W, C] for CTC
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W, C]
        
        # Apply neck if exists
        if self.neck is not None:
            x = self.neck(x)
        
        # Apply head
        x = self.head(x, targets)
        
        return x


def build_pp_ocrv3_rec_model(character_dict_path=None, 
                            use_space_char=False,
                            img_size=[48, 320],
                            **kwargs):
    """
    Build PP-OCRv3 text recognition model.
    
    Args:
        character_dict_path: Path to character dictionary file
        use_space_char: Whether to use space character
        img_size: Input image size [height, width]
        **kwargs: Additional arguments
    
    Returns:
        PP_OCRv3_Rec model
    """
    
    # Default English character set (alphanumeric)
    if character_dict_path is None:
        # Create default character list: digits + uppercase + lowercase + blank
        char_list = []
        # Add digits
        for i in range(10):
            char_list.append(str(i))
        # Add uppercase letters
        for i in range(26):
            char_list.append(chr(ord('A') + i))
        # Add lowercase letters  
        for i in range(26):
            char_list.append(chr(ord('a') + i))
        
        if use_space_char:
            char_list.append(' ')
        
        # Add blank token for CTC
        char_list.append('<blank>')
        
        num_classes = len(char_list)
    else:
        # Load character dictionary from file
        with open(character_dict_path, 'r', encoding='utf-8') as f:
            char_list = [line.strip() for line in f.readlines()]
        
        if use_space_char and ' ' not in char_list:
            char_list.append(' ')
            
        # Add blank token for CTC if not present
        if '<blank>' not in char_list:
            char_list.append('<blank>')
            
        num_classes = len(char_list)
    
    # Configure model
    backbone_config = {
        'dims': 64,
        'depths': 2,
        'num_heads': 8,
        'mixer': ['Global'] * 2,
        'img_size': img_size,
        'out_channels': 256,
        'out_char_num': 25,
    }
    
    head_config = {
        'out_channels': num_classes,
        'mid_channels': None,
        'return_feats': True
    }
    
    model = PP_OCRv3_Rec(
        backbone_config=backbone_config,
        head_config=head_config,
        **kwargs
    )
    
    # Store character list for decoding
    model.character_list = char_list
    model.character_dict = {char: idx for idx, char in enumerate(char_list)}
    
    return model


def pp_ocrv3_rec_english(pretrained=False, **kwargs):
    """PP-OCRv3 English text recognition model."""
    model = build_pp_ocrv3_rec_model(
        character_dict_path=None,
        use_space_char=False,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights if available
        # model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))
        pass
    
    return model


def pp_ocrv3_rec_multilingual(character_dict_path, pretrained=False, **kwargs):
    """PP-OCRv3 multilingual text recognition model."""
    model = build_pp_ocrv3_rec_model(
        character_dict_path=character_dict_path,
        use_space_char=True,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights if available
        # model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))
        pass
    
    return model