"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os.path as osp

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from transfer import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.analysis import collect_feature, tsne, a_distance
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def main(args: argparse.Namespace):
    cudnn.benchmark = True

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 90)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([transforms.ToTensor()])

    # train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
    #     utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_dataset = GalaxyDataset(annotations_file=train_file, transform=train_transform)
    train_target_dataset = GalaxyDataset(annotations_file=train_file, transform=train_transform)
    val_dataset = GalaxyDataset(annotations_file=val_file, transform=val_transform)
    test_dataset = GalaxyDataset(annotations_file=test_file, transform=val_transform)
    num_classes = 34

    train_source_loader = DataLoader(dataset=train_source_dataset, batch_size=config.batch_size,
                                     shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    train_target_loader = DataLoader(dataset=train_target_dataset, batch_size=config.batch_size,
                                     shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.WORKERS, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.WORKERS, pin_memory=True)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    if randomized:
        domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()
    # define optimizer and lr scheduler
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    for epoch in range(epochs):
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args)
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: ConditionalDomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    model.train()
    domain_adv.train()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDAN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
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
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
