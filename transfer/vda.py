import random
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
from tllib.modules.entropy import entropy
# from morphics.train import init_rand_seed
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy, binary_accuracy
from torch.backends import cudnn
from dataset.galaxy_dataset import *
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from transfer import schemas
from training import losses, metrics
import argparse
from transfer.utils import *
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.utils import *
from transfer.transfer_args import *
from tllib.modules.grl import WarmStartGradientReverseLayer


const_bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Flipout",  # Flipout or Reparameterization
    "moped_enable": False,  # initialize mu/sigma from the dnn weights
    "moped_delta": 0.2,
}


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map

    Given two inputs :math:`f` and :math:`g`, the definition is

    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),

    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.

    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024

    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The Conditional Domain Adversarial Loss used in `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_

    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(T(f_j^t, g_j^t))],\\

    where :math:`T` is a :class:`MultiLinearMap`  or :class:`RandomizedMultiLinearMap` which convert two tensors to a single tensor.

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        num_classes (int, optional): Number of classes. Default: -1
        features_dim (int, optional): Dimension of input features. Default: -1
        randomized_dim (int, optional): Dimension of features after randomized. Default: 1024
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.

    Inputs:
        - g_s (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> from tllib.alignment.cdan import ConditionalDomainAdversarialLoss
        >>> import torch
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim * num_classes, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
    """

    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: Optional[bool] = False,
                 randomized: Optional[bool] = False, num_classes: Optional[int] = -1,
                 features_dim: Optional[int] = -1, randomized_dim: Optional[int] = 1024,
                 reduction: Optional[str] = 'mean', sigmoid=True):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.entropy_conditioning = entropy_conditioning
        self.sigmoid = sigmoid
        self.reduction = reduction

        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = softmax_output(g, self.schema.question_index_groups).detach()
        # g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size

        if self.sigmoid:
            d_label = torch.cat((
                torch.ones((g_s.size(0), 1)).to(g_s.device),
                torch.zeros((g_t.size(0), 1)).to(g_t.device),
            ))
            self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
            if self.entropy_conditioning:
                return F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction=self.reduction)
            else:
                return F.binary_cross_entropy(d, d_label, reduction=self.reduction)
        else:
            d_label = torch.cat((
                torch.ones((g_s.size(0),)).to(g_s.device),
                torch.zeros((g_t.size(0),)).to(g_t.device),
            )).long()
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            if self.entropy_conditioning:
                raise NotImplementedError("entropy_conditioning")
            return F.cross_entropy(d, d_label, reduction=self.reduction)


class Transfer:
    def __init__(self, model, optimizer, config):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
        self.early_stopping = EarlyStopping(patience=self.config.patience, delta=0.001, verbose=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.config.patience,
                                           verbose=True)

    def dirichlet_loss_func(self, preds, labels):
        return losses.calculate_multiquestion_loss(labels, preds, self.schema.question_index_groups)

    def train_epoch(self, train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
                    domain_adv: ConditionalDomainAdversarialLoss, epoch: int, writer: SummaryWriter):
        self.model.train()
        domain_adv.train()
        for i in range(self.config.iters_per_epoch):
            x_s, labels_s = next(train_source_iter)[:2]
            x_t, = next(train_target_iter)[:1]
            x_s, x_t, labels_s = x_s.to(self.config.device), x_t.to(self.config.device), labels_s.to(self.config.device)

            x = torch.cat((x_s, x_t), dim=0)
            y, _, f = self.model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            dirich_loss = torch.mean(self.dirichlet_loss_func(y_s, labels_s))
            transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
            kl = get_kl_loss(self.model) / self.config.batch_size
            loss = dirich_loss + transfer_loss * self.config.trade_off + kl
            losses = loss.item() / self.config.batch_size

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return dirich_loss.item(), kl.item(), transfer_loss.item()

    # def eval(self, val_iter, domain_adv):
    #     eval_loss = 0
    #     eval_kl = 0
    #     with torch.no_grad():
    #         self.model.eval()
    #         domain_adv.eval()
    #         for i in range(self.config.iters_per_epoch):
    #             x_s, labels_s = next()

    def save_checkpoint(self, epoch):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch
        }
        os.makedirs(f'{self.config.save_dir}/checkpoint', exist_ok=True)
        torch.save(checkpoint, f'{self.config.save_dir}/checkpoint/ckpt_best_{epoch}.pth')
        torch.save(self.model, f'{self.config.save_dir}/model_{epoch}.pt')

    def train(self, train_source_loader, train_target_loader, val_loader, domain_adv):
        print("你又来迁移丹辣！")
        os.makedirs(self.config.save_dir + "log/", exist_ok=True)
        writer = SummaryWriter(self.config.save_dir + "log/")
        train_source_iter = ForeverDataIterator(train_source_loader)
        train_target_iter = ForeverDataIterator(train_target_loader)
        # val_iter = ForeverDataIterator(val_loader)
        for epoch in range(self.config.epochs):
            train_loss, train_kl, transfer_loss = self.train_epoch(train_source_iter, train_target_iter, domain_adv,
                                                                   epoch, writer)
            print("Epoch: {}, train loss: {:.4f}, train kl: {:.4f}, transfer loss: {:.4f}".format(epoch, train_loss,
                                                                                                  train_kl,
                                                                                                  transfer_loss))
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_kl', train_kl, epoch)
            writer.add_scalar('transfer_loss', transfer_loss, epoch)
            self.scheduler.step(train_loss + train_kl + transfer_loss)
            self.save_checkpoint(epoch)
            self.early_stopping(train_loss + train_kl + transfer_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break


class ClassifierBase(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        num_classes (int): Number of classes

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, num_classes: int, model, training=True):
        super(ClassifierBase, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self._features_dim = 1280
        self.intermediate_output = None
        self.register_intermediate_hook()
        self.training = training

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def register_intermediate_hook(self):
        target_layer = list(list(self.model.children())[0].children())[-2]
        target_layer.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.intermediate_output = output

    def forward(self, x: torch.Tensor):
        predictions, stn = self.model(x)
        if self.training:
            return predictions, stn, torch.squeeze(self.intermediate_output)
        else:
            return predictions, stn

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.model.parameters(), "lr": 0.1 * base_lr},
        ]
        return params


def main(config):
    cudnn.benchmark = True
    subset_size = 1000
    subset_indices = list(range(subset_size))
    train_source_dataset = GalaxyDataset(annotations_file=config.train_source, transform=config.transfer)
    train_target_dataset = GalaxyDataset(annotations_file=config.train_target, transform=config.transfer)
    val_dataset = GalaxyDataset(annotations_file=config.valid_target,
                                transform=transforms.Compose([transforms.ToTensor()]))
    # train_source_dataset = Subset(train_source_dataset, subset_indices)
    # train_target_dataset = Subset(train_target_dataset, subset_indices)
    # val_dataset = Subset(val_dataset, subset_indices)
    train_source_loader = DataLoader(dataset=train_source_dataset, batch_size=config.batch_size,
                                     shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    train_target_loader = DataLoader(dataset=train_target_dataset, batch_size=config.batch_size,
                                     shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.WORKERS, pin_memory=True)

    backbone = torch.load(config.model_path, map_location=config.device)
    classifier = ClassifierBase(config.num_classes, backbone, True).to(config.device)
    classifier_feature_dim = classifier.features_dim

    if config.randomized:
        domain_discri = DomainDiscriminator(config.randomized_dim, hidden_size=1024).to(config.device)
        # dnn_to_bnn(domain_discri, const_bnn_prior_parameters)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * config.num_classes, hidden_size=1024).to(
            config.device)
        # dnn_to_bnn(domain_discri, const_bnn_prior_parameters)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()
    # define optimizer and lr scheduler
    optimizer = torch.optim.Adam(all_parameters, lr=config.lr, betas=config.betas)

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=config.entropy,
        num_classes=config.num_classes, features_dim=classifier_feature_dim, randomized=config.randomized,
        randomized_dim=config.randomized_dim, reduction='mean'
    ).to(config.device)

    transfer = Transfer(classifier, optimizer, config)
    transfer.train(train_source_loader, train_target_loader, val_loader, domain_adv)


if __name__ == '__main__':
    init_rand_seed(1926)
    data_config = transfer_config()
    parser = argparse.ArgumentParser(description='CDAN for Unsupervised Domain Adaptation')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--entropy', default=False, action='store_true', help='use entropy conditioning')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=data_config['batch_size'], type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--root', default=data_config['root'], type=str,
                        help='path to dataset')
    parser.add_argument('--save-dir', default=data_config['save_dir'], type=str,
                        help='path to save models')
    parser.add_argument('--train-source', default=data_config['train_source'], type=str,
                        help='path to train source dataset')
    parser.add_argument('--train-target', default=data_config['train_target'], type=str,
                        help='path to train target dataset')
    parser.add_argument('--valid-target', default=data_config['valid_target'], type=str,
                        help='path to valid target dataset')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--patience', default=data_config['patience'], type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--dropout_rate', default=data_config['dropout_rate'], type=float, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--WORKERS', default=data_config['WORKERS'], type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=data_config['lr'], type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--phase', default=data_config['phase'], type=str, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--sample', default=data_config['sample'], type=int, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--model_path', default=data_config['model_path'], type=str, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--randomized', default=data_config['randomized'], type=bool, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--device', default=data_config['device'], type=str, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--randomized_dim', default=data_config['randomized_dim'], type=int, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--num_classes', default=data_config['num_classes'], type=int, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--iters_per_epoch', default=data_config['iters_per_epoch'], type=int, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--trade_off', default=data_config['trade_off'], type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--transfer', default=data_config['transfer'], metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--betas', default=data_config['betas'], metavar='LR',
                        help='initial learning rate')
    args = parser.parse_args()
    main(args)
