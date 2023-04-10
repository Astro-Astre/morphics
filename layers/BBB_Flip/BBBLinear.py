import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from metrics import calculate_kl as KL_DIV
import config_bayesian as cfg
from ..misc import ModuleWrapper


class BBBLinear(ModuleWrapper):

    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, device={}'.format(
            self.in_features, self.out_features, self.use_bias, self.device
        )

    def __str__(self):
        return 'BBBLinear(in_features={}, out_features={}, bias={})'.format(
            self.in_features, self.out_features, self.use_bias
        )

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

        weight_epsilon = torch.randn(self.out_features, self.in_features, device=x.device)
        weight = self.W_mu + weight_epsilon * self.W_sigma
        if self.use_bias:
            bias_epsilon = torch.randn(self.out_features, device=x.device)
            bias = self.bias_mu + bias_epsilon * self.bias_sigma
        else:
            bias = None

        out = F.linear(x, weight, bias)

        x_flipped = torch.cat([x[:, 1:], x[:, :-1]], dim=1)
        weight_flipped = torch.cat([weight[:, 1:], weight[:, :-1]], dim=1)
        out_flipped = F.linear(x_flipped, weight_flipped, None)

        return out + out_flipped

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
