"""
YOLOv10 PyTorch Implementation
A complete implementation of YOLOv10 architecture using pure PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


def autopad(k, p=None, d=1):
    """Auto-padding calculation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise convolution."""
    
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class RepConv(nn.Module):
    """RepVGG-style reparameterizable convolution."""
    
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
            self.rbr_dense = Conv(c1, c2, k, s, p=p, g=g, act=False)
            self.rbr_1x1 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """C2f module with cross-stage partial connections."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class PSA(nn.Module):
    """Position-Sensitive Attention module."""
    
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b.softmax(1) * b
        return self.cv2(torch.cat((a, b), 1))


class SCDown(nn.Module):
    """Spatial-Channel Decoupling Downsampling."""
    
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class C2fCIB(nn.Module):
    """C2f module with Compact Inverted Block (CIB)."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CIB(nn.Module):
    """Compact Inverted Block."""
    
    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else Conv(2 * c_, 2 * c_, 9, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)


class Attention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class Detect(nn.Module):
    """YOLOv10 detection head with dual assignments."""
    
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        
        # Inference path
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = x[i].view(shape[0], self.no, -1)
        return self.decode(torch.cat(x, -1))

    def decode(self, x):
        """Decode bounding boxes."""
        y = x.transpose(1, 2)
        boxes = y[..., :self.reg_max * 4]
        classes = y[..., self.reg_max * 4:]
        
        # Apply DFL
        boxes = self.dfl(boxes.view(*boxes.shape[:-1], 4, self.reg_max))
        return torch.cat((boxes, classes.sigmoid()), -1)


class DFL(nn.Module):
    """Distribution Focal Loss (DFL)."""
    
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv1d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class YOLOv10(nn.Module):
    """YOLOv10 model."""
    
    def __init__(self, cfg_dict=None, ch=3, nc=80, verbose=True):
        super().__init__()
        
        # Default YOLOv10n configuration
        if cfg_dict is None:
            cfg_dict = {
                'backbone': [
                    [-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
                    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
                    [-1, 3, C2f, [128, True]],
                    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
                    [-1, 6, C2f, [256, True]],
                    [-1, 1, SCDown, [512, 3, 2]],  # 5-P4/16
                    [-1, 6, C2f, [512, True]],
                    [-1, 1, SCDown, [1024, 3, 2]],  # 7-P5/32
                    [-1, 3, C2fCIB, [1024, True, True]],
                    [-1, 1, SPPF, [1024, 5]],  # 9
                    [-1, 1, PSA, [1024]],  # 10
                ],
                'head': [
                    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
                    [[-1, 6], 1, nn.Identity, []],  # cat backbone P4
                    [-1, 3, C2f, [512]],  # 13
                    
                    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
                    [[-1, 4], 1, nn.Identity, []],  # cat backbone P3
                    [-1, 3, C2f, [256]],  # 16 (P3/8-small)
                    
                    [-1, 1, Conv, [256, 3, 2]],
                    [[-1, 13], 1, nn.Identity, []],  # cat head P4
                    [-1, 3, C2f, [512]],  # 19 (P4/16-medium)
                    
                    [-1, 1, SCDown, [512, 3, 2]],
                    [[-1, 10], 1, nn.Identity, []],  # cat backbone P5
                    [-1, 3, C2fCIB, [1024, True, True]],  # 22 (P5/32-large)
                    
                    [[16, 19, 22], 1, Detect, [nc]],  # Detect(P3, P4, P5)
                ]
            }
        
        self.model, self.save = self._parse_model(cfg_dict, ch, verbose)
        self.names = [f'class{i}' for i in range(nc)]
        self.inplace = True

    def _parse_model(self, d, ch, verbose=True):
        """Parse model configuration."""
        if verbose:
            print(f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        
        layers, save, c2 = [], [], ch[-1]
        
        # Parse backbone
        for i, (f, n, m, args) in enumerate(d['backbone']):
            m = globals()[m.__name__ if isinstance(m, type) else m]
            for j, a in enumerate(args):
                if isinstance(a, str):
                    args[j] = locals()[a] if a in locals() else a
            
            n = max(round(n), 1) if n > 1 else n
            if m in (Conv, DWConv, RepConv, Bottleneck, C2f, C2fCIB, SPPF, PSA, SCDown):
                c1, c2 = ch[f], args[0]
                c2 = c2 if c2 != -1 else c1
                args = [c1, c2, *args[1:]]
                if m in (C2f, C2fCIB):
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is nn.Identity:
                c2 = ch[f]
            
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                print(f'{i:>3}{str(f):>20}{n:>3}{sum(x.numel() for x in m_.parameters()):10.0f}  {t:<45}{str(args):<30}')
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            ch.append(c2)
        
        # Parse head
        for i, (f, n, m, args) in enumerate(d['head'], len(d['backbone'])):
            m = globals()[m.__name__ if isinstance(m, type) else m]
            for j, a in enumerate(args):
                if isinstance(a, str):
                    args[j] = locals()[a] if a in locals() else a
            
            n = max(round(n), 1) if n > 1 else n
            if m in (Conv, DWConv, RepConv, Bottleneck, C2f, C2fCIB, SPPF, PSA, SCDown):
                c1, c2 = ch[f], args[0]
                c2 = c2 if c2 != -1 else c1
                args = [c1, c2, *args[1:]]
                if m in (C2f, C2fCIB):
                    args.insert(2, n)
                    n = 1
            elif m is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):
                    args[1] = [args[1] for _ in range(len(f))]
            elif m is nn.Upsample:
                c2 = ch[f]
            elif m is nn.Identity:
                c2 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                print(f'{i:>3}{str(f):>20}{n:>3}{sum(x.numel() for x in m_.parameters()):10.0f}  {t:<45}{str(args):<30}')
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            ch.append(c2)
        
        return nn.Sequential(*layers), sorted(save)

    def forward(self, x, augment=False, profile=False):
        """Forward pass."""
        return self._forward_once(x, profile)

    def _forward_once(self, x, profile=False):
        """Single forward pass."""
        y, dt = [], []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
        
        return x

    def _profile_one_layer(self, m, x, dt):
        """Profile one layer."""
        c = m == self.model[-1] and isinstance(x, list)
        flops = sum(self._get_flops(m, xi) for xi in (x if c else [x]))
        t = sum(self._time_sync() for _ in range(10))
        dt.append((t, flops))

    @staticmethod
    def _get_flops(m, x):
        """Get FLOPs for a layer."""
        if hasattr(m, 'flops'):
            return m.flops(x)
        return 0

    @staticmethod  
    def _time_sync():
        """Get current time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))


def yolov10n(pretrained=False, **kwargs):
    """YOLOv10 nano model."""
    model = YOLOv10(**kwargs)
    if pretrained:
        # Load pretrained weights if available
        pass
    return model


def yolov10s(pretrained=False, **kwargs):
    """YOLOv10 small model."""
    cfg = {
        'backbone': [
            [-1, 1, Conv, [64, 3, 2]],
            [-1, 1, Conv, [128, 3, 2]],
            [-1, 3, C2f, [128, True]],
            [-1, 1, Conv, [256, 3, 2]],
            [-1, 6, C2f, [256, True]],
            [-1, 1, SCDown, [512, 3, 2]],
            [-1, 6, C2f, [512, True]],
            [-1, 1, SCDown, [1024, 3, 2]],
            [-1, 3, C2fCIB, [1024, True, True]],
            [-1, 1, SPPF, [1024, 5]],
            [-1, 1, PSA, [1024]],
        ],
        'head': [
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, nn.Identity, []],
            [-1, 3, C2f, [512]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 4], 1, nn.Identity, []],
            [-1, 3, C2f, [256]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, 13], 1, nn.Identity, []],
            [-1, 3, C2f, [512]],
            [-1, 1, SCDown, [512, 3, 2]],
            [[-1, 10], 1, nn.Identity, []],
            [-1, 3, C2fCIB, [1024, True, True]],
            [[16, 19, 22], 1, Detect, [kwargs.get('nc', 80)]],
        ]
    }
    model = YOLOv10(cfg, **kwargs)
    if pretrained:
        pass
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = yolov10n(nc=80)  # 80 classes for COCO
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape if isinstance(y, torch.Tensor) else [yi.shape for yi in y]}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Export to ONNX
    model.eval()
    torch.onnx.export(
        model, x, "yolov10n.onnx",
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"},
                     "output": {0: "batch", 2: "anchors"}}
    )
    print("Model exported to yolov10n.onnx")