"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os.path as osp

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from morphics.train import init_rand_seed
from transfer import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.analysis import collect_feature, tsne, a_distance
import torchvision.transforms as transforms

from torch.backends import cudnn
from dataset.galaxy_dataset import *
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from utils import schemas
from training import losses, metrics
import argparse
from utils.utils import *
from torchvision.models import *
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from models.morphics import Morphics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.utils import *
from transfer.transfer_args import *


class Transfer:
    def __init__(self, model, optimizer, config):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.config.patience,
                                           verbose=True)
        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
        self.early_stopping = EarlyStopping(patience=self.config.patience, delta=0.001, verbose=True)

    def dirichlet_loss_func(self, preds, labels):
        return losses.calculate_multiquestion_loss(labels, preds, self.schema.question_index_groups)

    def train_epoch(self, train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          domain_adv: ConditionalDomainAdversarialLoss, epoch: int):
        self.model.train()
        domain_adv.train()
        for i in range(args.iters_per_epoch):
            x_s, labels_s = next(train_source_iter)[:2]
            x_t, = next(train_target_iter)[:1]
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            labels_s = labels_s.to(device)

            # compute output
            x = torch.cat((x_s, x_t), dim=0)
            y, f = self.model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            dirich_loss = self.dirichlet_loss_func(y_s, labels_s)
            transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
            s_kl = self.model.get_kl / self.config.batch_size
            loss = dirich_loss + transfer_loss * self.config.trade_off + s_kl

            losses.update(loss.item(), x_s.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

    def train(self, train_source_loader, train_target_loader, val_loader, model, domain_adv, optimizer, lr_scheduler, args):
        train_source_iter = ForeverDataIterator(train_source_loader)
        train_target_iter = ForeverDataIterator(train_target_loader)
        for epoch in range(args.epochs):
            train_epoch(train_source_iter, train_target_iter, model, domain_adv, optimizer, lr_scheduler, epoch, args)

            val_acc = self.validate(val_loader, model, args)
            self.scheduler.step(val_acc)
            self.early_stopping(val_acc, model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
def main(config):
    cudnn.benchmark = True
    train_source_dataset = GalaxyDataset(annotations_file=config.train_source, transform=config.transfer)
    train_target_dataset = GalaxyDataset(annotations_file=config.train_target, transform=config.transfer)
    val_dataset = GalaxyDataset(annotations_file=config.valid_target, transform=transforms.Compose([transforms.ToTensor()]))

    train_source_loader = DataLoader(dataset=train_source_dataset, batch_size=config.batch_size,
                                     shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    train_target_loader = DataLoader(dataset=train_target_dataset, batch_size=config.batch_size,
                                     shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.WORKERS, pin_memory=True)

    backbone = torch.load(config.model_path)
    pool_layer = nn.Identity() if config.no_pool else None
    classifier = ImageClassifier(backbone, config.num_classes, bottleneck_dim=config.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=config.finetune).to(config.device)
    classifier_feature_dim = classifier.features_dim

    if config.randomized:
        domain_discri = DomainDiscriminator(config.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * config.num_classes, hidden_size=1024).to(device)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()
    # define optimizer and lr scheduler
    optimizer = SGD(all_parameters, config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: config.lr * (1. + config.lr_gamma * float(x)) ** (-config.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=config.entropy,
        num_classes=config.num_classes, features_dim=classifier_feature_dim, randomized=config.randomized,
        randomized_dim=config.randomized_dim
    ).to(device)

    transfer = Transfer(classifier, optimizer, config)
    transfer.train(train_source_loader, train_target_loader, val_loader, classifier, domain_adv,
                   optimizer, lr_scheduler, config)


if __name__ == '__main__':
    init_rand_seed(1926)
    data_config = transfer_config()
    parser = argparse.ArgumentParser(description='CDAN for Unsupervised Domain Adaptation')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False, action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    args = parser.parse_args()
    main(args)

