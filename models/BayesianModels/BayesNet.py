from layers import FlattenLayer, ModuleWrapper
import torch.nn as nn
from layers.BBB.BBBConv import BBBConv2d
from layers.BBB.BBBLinear import BBBLinear
from collections import OrderedDict


class SimpleCNN(ModuleWrapper):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x


# 将刚才定义的卷积神经网络中的普通卷积层和普通全连接层批量修改为贝叶斯卷积核和贝叶斯全连接层
class BBBNet(ModuleWrapper):
    def __init__(self, model: nn.Module, priors=None):
        super(BBBNet, self).__init__()
        self.model = model
        # self.new_dict = copy.deepcopy(self.model)
        self.new_dict = OrderedDict()
        for name, module in self.model.named_children():
            # 注意这里的name一定要是唯一的字符串名字，如果拉出序号则后面setattr会报错
            print(name)
            if isinstance(module, nn.Sequential):
                # 如果当前模块是 nn.Sequential，则递归调用 BBBNet，将其内部的层替换为贝叶斯层
                self.new_dict[name] = BBBNet(module, priors)
            elif isinstance(module, nn.Conv2d):
                # 获取原始卷积层的参数
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                groups = module.groups
                bias = module.bias is not None

                # 创建贝叶斯卷积层并替换原始卷积层
                bayes_conv = BBBConv2d(in_channels, out_channels,
                                       kernel_size, stride,
                                       padding, dilation,
                                       groups, bias,
                                       priors)
                bayes_conv.W_mu = nn.Parameter(bayes_conv.W_mu)
                self.register_parameter(name.replace(".", "_") + "_W_mu", bayes_conv.W_mu)

                if bayes_conv.bias is not None:
                    bayes_conv.bias_mu = nn.Parameter(bayes_conv.bias_mu)
                    self.register_parameter(name.replace(".", "_") + "_bias_mu", bayes_conv.bias_mu)

                if bayes_conv.prior is not None:
                    self.register_buffer(name.replace(".", "_") + "_prior", bayes_conv.prior)
                # 复制原始卷积层的权重和偏置的均值参数
                bayes_conv.W_mu.data.copy_(module.weight.data)
                if bias:
                    bayes_conv.bias_mu.data.copy_(module.bias.data)
                self.new_dict[name] = bayes_conv

            elif isinstance(module, nn.Linear):
                # 获取原始全连接层的参数
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias is not None

                # 创建贝叶斯全连接层并替换原始全连接层
                bayes_linear = BBBLinear(in_features, out_features,
                                         bias, priors)

                # 复制原始全连接层的权重和偏置的均值参数
                bayes_linear.W_mu.data.copy_(module.weight.data)
                if bias:
                    bayes_linear.bias_mu.data.copy_(module.bias.data)
                self.new_dict[name] = bayes_linear
            else:
                self.new_dict[name] = module
        self.model._modules = self.new_dict

if __name__ == '__main__':
    model = SimpleCNN()
    from models.NonBayesianModels.effnetv2 import *
    import torchvision.models as models
    from torchsummary import summary
    model = BBBNet(model)
    model.cuda()
    summary(model, (3, 32, 32), device='cuda', batch_size=256)