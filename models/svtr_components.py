import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim, num_heads=8, HW=[8, 25], local_k=[3, 3]):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim, dim, local_k, 1, [local_k[0]//2, local_k[1]//2], groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        x = self.local_mixer(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, mixer='Global', HW=[8, 25], local_k=[7, 11], 
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones([HW[0] * HW[1], HW[0] * HW[1]], dtype=torch.float32)
            for h in range(0, HW[0]):
                for w in range(0, HW[1]):
                    query_id = h * HW[1] + w
                    for kh in range(max(0, h - hk // 2), min(HW[0], h + hk // 2 + 1)):
                        for kw in range(max(0, w - wk // 2), min(HW[1], w + wk // 2 + 1)):
                            key_id = kh * HW[1] + kw
                            mask[query_id, key_id] = 0
            mask = mask.unsqueeze(0).unsqueeze(0)
            self.register_buffer("mask", mask)
        self.mixer = mixer

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.mixer == 'Local':
            attn += self.mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mixer='Global', local_mixer=[7, 11], HW=[8, 25], 
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, epsilon=1e-6, prenorm=True):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        self.mixer = Attention(
            dim, num_heads=num_heads, mixer=mixer, HW=HW, local_k=local_mixer,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=[32, 128], patch_size=[4, 4], in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SubSample(nn.Module):
    def __init__(self, in_channels, out_channels, types='Pool', stride=[2, 1], sub_norm='nn.LayerNorm', act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class SVTRNet(nn.Module):
    def __init__(self, 
                 img_size=[32, 128], 
                 in_chans=3, 
                 embed_dim=[64, 128, 256], 
                 depth=[3, 6, 3], 
                 num_heads=[2, 4, 8], 
                 mixer=['Local'] * 6 + ['Global'] * 6, 
                 local_mixer=[[7, 11], [7, 11], [7, 11]], 
                 patch_merging='Conv', 
                 mlp_ratio=4, 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 last_drop=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1, 
                 norm_layer='nn.LayerNorm', 
                 sub_norm='nn.LayerNorm', 
                 epsilon=1e-6, 
                 out_channels=192, 
                 out_char_num=25, 
                 block_unit='Block', 
                 act='nn.GELU', 
                 last_stage=True, 
                 sub_num=2, 
                 prenorm=True, 
                 use_lenhead=False):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_size = [4, 4]
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0])
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        Block_unit = eval(block_unit)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.blocks1 = nn.ModuleList([
            Block_unit(dim=embed_dim[0], num_heads=num_heads[0], mixer=mixer[0], 
                       HW=self.HW, local_mixer=local_mixer[0], mlp_ratio=mlp_ratio, 
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[i], norm_layer=eval(norm_layer), act_layer=eval(act), 
                       epsilon=epsilon, prenorm=prenorm)
            for i in range(depth[0])])
        
        if patch_merging is not None:
            self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[2, 1])
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([
            Block_unit(dim=embed_dim[1], num_heads=num_heads[1], mixer=mixer[depth[0] + i], 
                       HW=HW, local_mixer=local_mixer[1], mlp_ratio=mlp_ratio, 
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[depth[0] + i], norm_layer=eval(norm_layer), act_layer=eval(act), 
                       epsilon=epsilon, prenorm=prenorm)
            for i in range(depth[1])])
        
        if patch_merging is not None:
            self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], sub_norm=sub_norm, stride=[2, 1])
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        
        self.blocks3 = nn.ModuleList([
            Block_unit(dim=embed_dim[2], num_heads=num_heads[2], mixer=mixer[depth[0] + depth[1] + i], 
                       HW=HW, local_mixer=local_mixer[2], mlp_ratio=mlp_ratio, 
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[depth[0] + depth[1] + i], norm_layer=eval(norm_layer), 
                       act_layer=eval(act), epsilon=epsilon, prenorm=prenorm)
            for i in range(depth[2])])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.last_conv = nn.Conv2d(in_channels=embed_dim[2], out_channels=self.out_channels, 
                                     kernel_size=1, stride=1, padding=0, bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop, inplace=False)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(p=last_drop, inplace=False)

        # init pos_embed
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.permute([0, 2, 1]).reshape([B, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.permute([0, 2, 1]).reshape([B, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
                w = self.HW[1]
            else:
                h = self.HW[0]
                w = self.HW[1]
            x = x.permute([0, 2, 1]).reshape([B, self.embed_dim[2], h, w])
            x = self.avg_pool(x)
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x