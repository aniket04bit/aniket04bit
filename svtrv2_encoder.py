import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


class PatchEmbedding(nn.Module):
    """Patch embedding layer for initial feature extraction"""
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.norm(x)
        return x, (H, W)


class ConvMixer(nn.Module):
    """Convolution-based mixer for local feature interaction"""
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Reshape to spatial format for conv
        x_spatial = x.transpose(1, 2).view(B, C, H, W)
        x_spatial = self.dwconv(x_spatial)
        x = x_spatial.flatten(2).transpose(1, 2)  # Back to [B, N, C]
        
        # Point-wise convolutions
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        return x


class GlobalMixer(nn.Module):
    """Global self-attention mixer"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FGlobalMixer(nn.Module):
    """Fast Global mixer with reduced computational complexity"""
    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_sr = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.norm(x)
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_spatial = x.transpose(1, 2).view(B, C, H, W)
            x_sr = self.sr(x_spatial).flatten(2).transpose(1, 2)
            x_sr = self.norm_sr(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class SVTRBlock(nn.Module):
    """SVTR Block with different mixer types"""
    def __init__(self, dim: int, num_heads: int, mixer_type: str = "Global", local_k: List[int] = [5, 5], sr_ratio: int = 1):
        super().__init__()
        self.mixer_type = mixer_type
        
        if mixer_type == "Conv":
            self.mixer = ConvMixer(dim, kernel_size=local_k[0])
        elif mixer_type == "FGlobal":
            self.mixer = FGlobalMixer(dim, num_heads, sr_ratio=sr_ratio)
        elif mixer_type == "Global":
            self.mixer = GlobalMixer(dim, num_heads)
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, x, H, W):
        # Mixer
        x = x + self.mixer(x, H, W)
        
        # MLP
        x = x + self.mlp(self.norm(x))
        
        return x


class DownSample(nn.Module):
    """Downsampling layer between stages"""
    def __init__(self, in_dim: int, out_dim: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class SVTRv2LNConvTwo33(nn.Module):
    """SVTRv2 Encoder with LayerNorm and Conv operations"""
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        in_channels: int = 3,
        dims: List[int] = [128, 256, 384],
        depths: List[int] = [6, 6, 6],
        num_heads: List[int] = [4, 8, 12],
        mixer: List[List[str]] = [
            ['Conv','Conv','Conv','Conv','Conv','Conv'],
            ['Conv','Conv','FGlobal','Global','Global','Global'],
            ['Global','Global','Global','Global','Global','Global']
        ],
        local_k: List[List[int]] = [[5, 5], [5, 5], [-1, -1]],
        sub_k: List[List[int]] = [[1, 1], [2, 1], [-1, -1]],
        last_stage: bool = False,
        feat2d: bool = True,
        use_pos_embed: bool = False
    ):
        super().__init__()
        self.num_stages = len(dims)
        self.feat2d = feat2d
        self.last_stage = last_stage
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, dims[0])
        
        # Positional embedding (if used)
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size[0]//4 * img_size[1]//4, dims[0]))
        else:
            self.pos_embed = None
            
        # Build stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            # Create blocks for this stage
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                mixer_type = mixer[i][j] if j < len(mixer[i]) else mixer[i][-1]
                local_kernel = local_k[i] if local_k[i] != [-1, -1] else [7, 7]
                sr_ratio = sub_k[i][0] if sub_k[i] != [-1, -1] else 1
                
                block = SVTRBlock(
                    dim=dims[i],
                    num_heads=num_heads[i],
                    mixer_type=mixer_type,
                    local_k=local_kernel,
                    sr_ratio=sr_ratio
                )
                blocks.append(block)
            
            # Downsampling (except for first stage and potentially last stage)
            if i > 0:
                downsample = DownSample(dims[i-1], dims[i])
            else:
                downsample = None
                
            stage = nn.ModuleDict({
                'downsample': downsample,
                'blocks': blocks
            })
            self.stages.append(stage)
        
        # Final norm
        self.norm = nn.LayerNorm(dims[-1])
        
        # Initialize weights
        self.apply(self._init_weights)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        # Patch embedding
        x, (H, W) = self.patch_embed(x)
        
        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        features = []
        
        # Forward through stages
        for i, stage in enumerate(self.stages):
            # Downsampling
            if stage['downsample'] is not None:
                x, (H, W) = stage['downsample'](x, H, W)
            
            # Forward through blocks
            for block in stage['blocks']:
                x = block(x, H, W)
            
            # Collect features
            if self.feat2d:
                # Reshape to 2D feature map format
                B, N, C = x.shape
                feat = x.transpose(1, 2).view(B, C, H, W)
                features.append(feat)
            else:
                features.append(x)
        
        # Final normalization
        x = self.norm(x)
        
        if not self.last_stage:
            if self.feat2d:
                B, N, C = x.shape
                x = x.transpose(1, 2).view(B, C, H, W)
            return x
        else:
            return features


if __name__ == "__main__":
    # Test the encoder
    model = SVTRv2LNConvTwo33()
    x = torch.randn(2, 3, 32, 128)  # Batch size 2, typical text recognition input size
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")