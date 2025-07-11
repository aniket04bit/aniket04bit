import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            HardSigmoid(inplace=True)
        )

    def forward(self, x):
        return x * self.se(self.avg_pool(x))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dw_size=3, use_se=False):
        super(DepthwiseSeparable, self).__init__()
        
        self.use_se = use_se
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=dw_size,
            stride=stride, padding=dw_size//2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        if use_se:
            self.se = SEModule(in_channels)
            
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = HardSwish(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        if self.use_se:
            x = self.se(x)
            
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class LCNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dw_size=3, use_se=False):
        super(LCNetBlock, self).__init__()
        
        self.use_shortcut = stride == 1 and in_channels == out_channels
        
        self.conv = DepthwiseSeparable(
            in_channels, out_channels, stride=stride, 
            dw_size=dw_size, use_se=use_se
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_shortcut:
            out = out + x
        return out


class LCNet(nn.Module):
    def __init__(self, scale=1.0, num_classes=1000):
        super(LCNet, self).__init__()
        
        self.scale = scale
        
        # Define the architecture
        self.cfg = [
            # [out_channels, kernel_size, stride, use_se]
            [32, 3, 1, False],
            [64, 3, 2, False],
            [64, 3, 1, False],
            [64, 3, 1, False],
            [128, 3, 2, False],
            [128, 3, 1, False],
            [128, 5, 1, True],
            [128, 5, 1, True],
            [128, 5, 1, True],
            [128, 5, 1, True],
            [128, 5, 1, True],
            [256, 5, 2, True],
            [256, 5, 1, True],
            [256, 5, 1, True],
            [256, 5, 1, True],
            [256, 5, 1, True],
            [512, 5, 2, True],
            [512, 5, 1, True],
        ]
        
        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self._make_divisible(32 * scale), kernel_size=3, stride=2, 
                     padding=1, bias=False),
            nn.BatchNorm2d(self._make_divisible(32 * scale)),
            HardSwish(inplace=True)
        )
        
        # Build inverted residual blocks
        self.layers = nn.ModuleList()
        in_channels = self._make_divisible(32 * scale)
        
        for out_channels, kernel_size, stride, use_se in self.cfg:
            out_channels = self._make_divisible(out_channels * scale)
            layer = LCNetBlock(in_channels, out_channels, stride, kernel_size, use_se)
            self.layers.append(layer)
            in_channels = out_channels
        
        # Final conv layer
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels, self._make_divisible(1280 * scale), kernel_size=1, bias=False),
            nn.BatchNorm2d(self._make_divisible(1280 * scale)),
            HardSwish(inplace=True)
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self._make_divisible(1280 * scale), num_classes)
        )
        
        self._initialize_weights()

    def _make_divisible(self, v, divisor=8, min_value=None):
        """Make value divisible by divisor."""
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Collect features at different stages
            if i in [2, 6, 12, 17]:  # After stages 1, 2, 3, 4
                features.append(x)
        
        return features

    def forward_features(self, x):
        """Extract features without classification head."""
        return self.forward(x)


def lcnet_050(**kwargs):
    """LCNet with 0.5x width multiplier."""
    return LCNet(scale=0.5, **kwargs)


def lcnet_075(**kwargs):
    """LCNet with 0.75x width multiplier."""
    return LCNet(scale=0.75, **kwargs)


def lcnet_100(**kwargs):
    """LCNet with 1.0x width multiplier."""
    return LCNet(scale=1.0, **kwargs)


def lcnet_150(**kwargs):
    """LCNet with 1.5x width multiplier."""
    return LCNet(scale=1.5, **kwargs)


def lcnet_200(**kwargs):
    """LCNet with 2.0x width multiplier."""
    return LCNet(scale=2.0, **kwargs)