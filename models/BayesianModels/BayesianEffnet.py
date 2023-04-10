from layers import FlattenLayer, BBB_LRT_Linear, BBB_LRT_Conv2d, ModuleWrapper, BBB_Conv2d, BBB_Linear
import math
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
class BayesianEfficientNetB0(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)

        # Replace stem convolution layer with BayesianConv2d
        in_channels = 3
        out_channels = self._blocks_args[0].input_filters
        kernel_size = 3
        stride = 2
        padding = 1
        self._conv_stem = BBB_LRT_Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Replace head convolution layer with BayesianConv2d
        in_channels = self._blocks_args[-1].output_filters
        out_channels = self._blocks_args[-1].output_filters * self._global_params.width_coefficient
        kernel_size = 1
        self._conv_head = BBB_LRT_Conv2d(in_channels, out_channels, kernel_size)

        # Replace final batch normalization layer with a dummy layer
        self._bn0 = nn.Identity()

        # Replace final fully connected layer with BayesianLinear
        in_features = self._fc.in_features
        out_features = self._fc.out_features
        self._fc = BBB_LRT_Linear(in_features, out_features)

    def forward(self, x):
        kl_div = 0

        # Stem
        x = self._conv_stem(x)
        kl_div += self._conv_stem.kl_divergence()
        x = self._swish(self._bn0(x))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._conv_head(x)
        kl_div += self._conv_head.kl_divergence()
        x = self._swish(self._bn0(x))

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._global_params.include_top:
            x = self._dropout(x)
            x, kl_div_fc = self._fc(x)
            kl_div += kl_div_fc

        return x, kl_div

if __name__ == '__main__':
    model = BayesianEfficientNetB0.from_name('efficientnet-b0', num_classes=10)
    model.eval()
    import torchsummary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # input = torch.randn(32, 3, 256, 256)
    # y = model(input.cuda())
    # print(y[0])
    # print("*" * 100)
    # print(y[1])
    torchsummary.summary(model, (3, 256, 256), batch_size=256, device="cuda")
    # print(model)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = model()  # 替换成你的自定义卷积层网络
    print(f'The number of trainable parameters of the model is {count_parameters(model):,}')
    # 获取所有可训练参数
    params = list(model.parameters())

    # 遍历每个参数，输出参数形状和名称
    # for i, param in enumerate(params):
    #     print(f"Param {i}: {param.shape}, {param.name}")
