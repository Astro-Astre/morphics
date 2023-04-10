import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from training.losses import *
from shared import label_metadata, schemas
# def dirichlet_loss(output, target, alpha=1e-5):
#     output = F.softmax(output, dim=1)
#     loss = -torch.mean(torch.sum(target * torch.log(output + alpha), dim=1))
#     return loss

class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.question_answer_pairs = label_metadata.gz2_pairs  # 问题？
        self.dependencies = label_metadata.gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        # print(self.train_size)
        return calculate_multiquestion_loss(input, target, self.schema.question_index_groups) * self.train_size + beta * kl
        # return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl

# class ELBO(nn.Module):
#     def __init__(self, train_size):
#         super(ELBO, self).__init__()
#         self.train_size = train_size
#         self.question_answer_pairs = label_metadata.gz2_pairs  # 问题？
#         self.dependencies = label_metadata.gz2_and_decals_dependencies
#         self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
#
#     def forward(self, input, target, kl, beta):
#         assert not target.requires_grad
#         nll = torch.mean(calculate_multiquestion_loss(input, target, self.schema.question_index_groups)) * self.train_size
#         print(nll)
#         return nll[0] + beta * kl
#         # return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl
# def lr_linear(epoch_num, decay_start, total_epochs, start_value):
#     if epoch_num < decay_start:
#         return start_value
#     return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    """
    正则化超参数
    :param batch_idx:
    :param m:
    :param beta_type:
    :param epoch:
    :param num_epochs:
    :return:
    """
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
