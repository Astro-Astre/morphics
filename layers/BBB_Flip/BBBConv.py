import sys

sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import utils
from metrics import calculate_kl as KL_DIV
import config_bayesian as cfg
from ..misc import ModuleWrapper


class BBBConv2d(ModuleWrapper):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None, groups: int = 1, ):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 50,
                'prior_sigma': 100,
                'posterior_mu_initial': (20, 1),
                'posterior_rho_initial': (20, 1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        # self.weight_params = nn.ParameterList([self.W_mu, self.W_rho])
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
            # self.bias_params = nn.ParameterList([self.bias_mu, self.bias_rho])
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.bias_params = []

        self.reset_parameters()

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups, self.use_bias)

    def __str__(self):
        return "BBBConv2d(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={})".format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups, self.use_bias)

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        weight_epsilon = torch.randn(self.out_channels, self.in_channels, *self.kernel_size, device=x.device)
        weight = self.W_mu + weight_epsilon * self.W_sigma
        if self.use_bias:
            bias_epsilon = torch.randn(self.out_channels, device=x.device)
            bias = self.bias_mu + bias_epsilon * self.bias_sigma
        else:
            bias = None

        out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        x_flipped = torch.cat([x[:, :, 1:], x[:, :, :-1]], dim=2)
        weight_flipped = torch.cat([weight[:, :, 1:], weight[:, :, :-1]], dim=2)
        out_flipped = F.conv2d(x_flipped, weight_flipped, None, self.stride, self.padding, self.dilation, self.groups)

        return out + out_flipped

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
