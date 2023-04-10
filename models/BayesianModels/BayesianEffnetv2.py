"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch
import torch.nn as nn
import math

# import torchsummary
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor

from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper
import metrics

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(ModuleWrapper):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(ModuleWrapper):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            FlattenLayer(oup),
            BBB_Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            FlattenLayer(_make_divisible(inp // reduction, 8)),
            BBB_Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        self.kl += kl
        return x * y

    # def __call__(self, x):
    #     out, kl = super().__call__(x)
    #     return out, kl


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        BBB_Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        BBB_Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(ModuleWrapper):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                BBB_Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                BBB_Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                BBB_Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                BBB_Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                BBB_Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl += module.kl_loss()
        self.kl += kl
        return out

    # def __call__(self, x):
    #     out, kl = super().__call__(x)
    #     return out, kl


class ScaledSigmoid(nn.modules.Sigmoid):
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ReLU

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): any vector. typically logits from a neural network
        Returns:
            Tensor: input mapped to range (1, 101) via torch.sigmoid
        """
        return torch.sigmoid(input) * 100. + 1.  # could make args if I needed
class EffNetV2(ModuleWrapper):
    def __init__(self, cfgs, num_classes=1000, width_mult=1., layer_type='lrt', activation_type='softplus'):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs
        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            BBB_Linear(output_channel, num_classes),
            ScaledSigmoid()
        )

        self._initialize_weights()

    def forward(self, x):
        # x = self.spatial_transform(x)
        # print(x.shape)
        _kl = 0.0
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        kl = 0.0
        for module in self.modules():
            if isinstance(module, BBB_Conv2d) or isinstance(module, BBB_Linear):
                kl += module.kl_loss()
        self.kl += kl
        return x, self.kl



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BBB_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.W_mu.data.normal_(0, math.sqrt(2. / n))
                m.W_rho.data.normal_(0, math.sqrt(2. / n))
                if m.bias_mu is not None:
                    m.bias_mu.data.zero_()
                    m.bias_rho.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, BBB_Linear):
            # m.W_sigma.data.normal_(0, 0.001)
            # m.bias_sigma.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


# if __name__ == '__main__':
#     model = effnetv2_s()
#     model.eval()
#     import torchsummary
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     input = torch.randn(32, 3, 256, 256)
#     y = model(input.cuda())
#     print(y[0])
#     print("*" * 100)
#     print(y[1])
    # torchsummary.summary(model, (3, 256, 256), batch_size=256, device="cuda")
    # print(model)
    #
    #
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # model = model()  # 替换成你的自定义卷积层网络
    # print(f'The number of trainable parameters of the model is {count_parameters(model):,}')
    # # 获取所有可训练参数
    # params = list(model.parameters())
    #
    # # 遍历每个参数，输出参数形状和名称
    # for i, param in enumerate(params):
    #     print(f"Param {i}: {param.shape}, {param.name}")
