import sys
sys.path.append("") # Adds higher directory to python modules path.
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from metrics import calculate_kl as KL_DIV
from layers.misc import ModuleWrapper


class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None, groups: int = 1, ):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) # kernel_size is a tuple
        self.stride = stride # stride is a tuple
        self.padding = padding # padding is a tuple
        self.dilation = dilation # dilation is a tuple
        self.groups = groups # groups is an int
        self.use_bias = bias # bias is a boolean
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define priors
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

        # define weights and biases
        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.bias_params = []
        self.reset_parameters()

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test it by printing an object of this class.
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups, self.use_bias)

    def __str__(self):
        # (Optional)Set the string representation of this module.
        return "BBBConv2d(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={})".format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups, self.use_bias)

    def reset_parameters(self):
        # (Optional)Reset learnable parameters of the module.
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        # (Required)Defines the computation performed at every call.
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

        # Flip the kernel and convolve again
        # This is a trick to make the network invariant to horizontal and vertical flips
        x_flipped = torch.cat([x[:, :, 1:], x[:, :, :-1]], dim=2)
        weight_flipped = torch.cat([weight[:, :, 1:], weight[:, :, :-1]], dim=2)
        out_flipped = F.conv2d(x_flipped, weight_flipped, None, self.stride, self.padding, self.dilation, self.groups)

        return out + out_flipped

    def kl_loss(self):
        # (Optional)Defines the KL loss.
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
