import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

import data
import utils
import metrics
import config_bayesian as cfg
from args import data_config
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet


import random
from torch.backends import cudnn

from pytorch_galaxy_datasets.galaxy_dataset import GalaxyDataset

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')
def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            print(net_out.shape)
            outputs[:, :, j] = net_out

        kl = kl / num_ens
        kl_list.append(kl.item())
        # log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i - 1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        # accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss / len(trainloader), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        # log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i - 1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(outputs, labels, kl, beta).item()
        # accs.append(metrics.acc(log_outputs, labels))

    return valid_loss / len(validloader)


def run(dataset, net_type):
    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    # trainset, testset, inputs, outputs = data.getDataset(dataset)
    # train_loader, valid_loader, test_loader = data.getDataloader(
    #     trainset, testset, valid_size, batch_size, num_workers)


    train_data = GalaxyDataset(annotations_file=data_config.train_file, transform=data_config.transfer)
    train_loader = DataLoader(dataset=train_data, batch_size=data_config.batch_size,
                              shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

    valid_data = GalaxyDataset(annotations_file=data_config.valid_file, transform=data_config.transfer)
    valid_loader = DataLoader(dataset=valid_data, batch_size=data_config.batch_size,
                              shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)
    net = getModel(net_type, 3, 34, priors, layer_type, activation_type).to(device)
    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(train_data)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens,
                                                      beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type,
                                               epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)

        print(
            'Epoch: {} \tTraining Loss: {:.4f}\tValidation Loss: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_loss, valid_loss, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss


if __name__ == '__main__':
    init_rand_seed(1926)
    parser = argparse.ArgumentParser(description="PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='alexnet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)