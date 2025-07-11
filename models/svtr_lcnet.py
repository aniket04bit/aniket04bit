import torch
import torch.nn as nn
import torch.nn.functional as F
from .lcnet import LCNet
from .svtr_components import Block, DropPath


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SVTRLCNet(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 dims=64, 
                 depths=2, 
                 num_heads=8, 
                 mixer=['Global'] * 2,
                 local_mixer=[[7, 11], [7, 11]], 
                 img_size=[48, 320],
                 out_channels=256,
                 out_char_num=25,
                 mlp_ratio=4, 
                 qkv_bias=True,
                 qk_scale=None, 
                 drop_rate=0., 
                 last_drop=0.1, 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 epsilon=1e-6,
                 use_lenhead=False,
                 **kwargs):
        super(SVTRLCNet, self).__init__()
        
        self.img_size = img_size
        self.out_char_num = out_char_num
        self.out_channels = out_channels
        self.use_lenhead = use_lenhead
        
        # Use first 3 stages of LCNet as backbone
        self.backbone = LCNet(scale=0.5)
        
        # Get feature dimensions from LCNet stages
        # LCNet produces features at different scales
        # We'll use the output from stage 2 (after downsampling)
        backbone_out_channels = int(128 * 0.5)  # LCNet scale 0.5, stage 2 output
        
        # We'll determine the actual feature map size dynamically
        # so we don't need to hardcode HW here
        self.HW = None  # Will be set dynamically
        
        # Project LCNet features to transformer dimension
        self.neck = nn.Sequential(
            nn.Conv2d(backbone_out_channels, dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dims),
            Swish()
        )
        
        # Positional embedding will be created dynamically
        self.pos_embed = None
        self.dims = dims
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # SVTR Transformer blocks - HW will be set during first forward pass
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dims, 
                num_heads=num_heads, 
                mixer=mixer[i] if i < len(mixer) else 'Global',
                local_mixer=local_mixer[i] if i < len(local_mixer) else [7, 11],
                HW=[12, 80],  # Default value, will be updated dynamically
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[i], 
                norm_layer=nn.LayerNorm, 
                act_layer=nn.GELU, 
                epsilon=epsilon, 
                prenorm=True
            )
            for i in range(depths)
        ])
        
        # Final layers
        self.norm = nn.LayerNorm(dims, eps=epsilon)
        
        # Global average pooling along height dimension
        self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
        
        # Final projection
        self.last_conv = nn.Conv2d(
            in_channels=dims, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop, inplace=False)
        
        # Optional length head
        if use_lenhead:
            self.len_conv = nn.Linear(dims, out_channels)
            self.hardswish_len = nn.Hardswish() 
            self.dropout_len = nn.Dropout(p=last_drop, inplace=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        
        # Extract features from LCNet backbone (first 3 stages)
        features = self.backbone.forward_features(x)
        
        # Use output from stage 2 (after first downsampling)
        # This gives us good spatial resolution while having sufficient receptive field
        x = features[1]  # Use stage 2 features
        
        # Project to transformer dimension
        x = self.neck(x)
        
        # Reshape to sequence for transformer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C where N = H*W
        
        # Create positional embedding if not exists or if size changed
        if self.pos_embed is None or self.pos_embed.shape[1] != H * W:
            self.pos_embed = nn.Parameter(torch.zeros(1, H * W, self.dims, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
            
            # Update HW for transformer blocks
            self.HW = [H, W]
            for block in self.blocks:
                block.HW = [H, W]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Apply final norm
        x = self.norm(x)
        
        # Reshape back to feature map
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        # Global average pooling
        x = self.avg_pool(x)
        
        # Final convolution
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        
        return x

    def forward(self, x):
        return self.forward_features(x)


def svtr_lcnet_tiny(**kwargs):
    """SVTR-LCNet Tiny model for PP-OCRv3."""
    model = SVTRLCNet(
        dims=64,
        depths=2,
        num_heads=8,
        mixer=['Global'] * 2,
        **kwargs
    )
    return model


def svtr_lcnet_small(**kwargs):
    """SVTR-LCNet Small model."""
    model = SVTRLCNet(
        dims=96,
        depths=4,
        num_heads=8,
        mixer=['Local', 'Local', 'Global', 'Global'],
        **kwargs
    )
    return model


def svtr_lcnet_base(**kwargs):
    """SVTR-LCNet Base model."""
    model = SVTRLCNet(
        dims=192,
        depths=6,
        num_heads=12,
        mixer=['Local'] * 3 + ['Global'] * 3,
        **kwargs
    )
    return model