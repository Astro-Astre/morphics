import numpy as np
from torch import nn
from training.losses import *
from utils import label_metadata, schemas


# 定义 ELBO 类，继承自 nn.Module
class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.question_answer_pairs = label_metadata.gz2_pairs  # 问题答案对
        self.dependencies = label_metadata.gz2_and_decals_dependencies  # 依赖关系
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)  # 构建模式

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        # 计算多问题损失
        return calculate_multiquestion_loss(input, target, self.schema.question_index_groups) * self.train_size + beta * kl


def acc(outputs, targets):
    # 计算准确度
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    # 计算 KL 散度
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    """
    计算正则化超参数 beta
    :param batch_idx: 批次索引
    :param m: 样本数
    :param beta_type: beta 类型
    :param epoch: 当前训练周期数
    :param num_epochs: 总训练周期数
    :return: 计算得到的 beta 值
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
