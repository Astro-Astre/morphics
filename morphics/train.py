import random
from torch.backends import cudnn
from dataset.galaxy_dataset import *
from args import *
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import schemas
from training import losses, metrics
import argparse
from utils.utils import *
from torchvision.models import *
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from models.morphics import Morphics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.utils import *


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


def get_avg_loss(loss):
    avg_loss = np.zeros(loss.shape[1])
    for i in range(loss.shape[1]):
        avg_loss[i] += loss[:, i].sum()
    return avg_loss


class Trainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        self.config = config
        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
        self.early_stopping = EarlyStopping(patience=7, verbose=True)

    def dirichlet_loss_func(self, preds, labels):
        return losses.calculate_multiquestion_loss(labels, preds, self.schema.question_index_groups)

    def train_epoch(self, train_loader, epoch, writer):
        train_loss = 0
        train_kl = 0
        self.model.train()
        for i, (X, label) in enumerate(train_loader):
            label = torch.as_tensor(label, dtype=torch.long).cuda()
            X = X.cuda()
            output_ = []
            kl_ = []
            for mc_run in range(1):
                output = self.model(X)
                kl = get_kl_loss(self.model)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            dirichlet_loss = torch.mean(self.dirichlet_loss_func(output, label), dim=0)
            scaled_kl = kl / args.batch_size
            loss_value = torch.mean(dirichlet_loss) + scaled_kl
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            train_loss += dirichlet_loss.item()
            train_kl += kl.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_kl = train_kl / len(train_loader)
        writer.add_scalar('Training loss by steps', avg_train_loss, epoch)
        writer.add_scalar('Training kl by steps', avg_train_kl, epoch)
        return avg_train_loss, avg_train_kl

    def evaluate(self, valid_loader, epoch, writer):
        eval_loss = 0
        eval_kl = 0
        with torch.no_grad():
            self.model.eval()
            for X, label in valid_loader:
                label = torch.as_tensor(label, dtype=torch.long).cuda()
                X = X.cuda()
                output_ = []
                kl_ = []
                for mc_run in range(1):
                    output = self.model(X)
                    kl = get_kl_loss(self.model)
                    output_.append(output)
                    kl_.append(kl)
                output = torch.mean(torch.stack(output_), dim=0)
                kl = torch.mean(torch.stack(kl_), dim=0)
                dirichlet_loss = torch.mean(self.dirichlet_loss_func(output, label), dim=0)
                scaled_kl = kl / args.batch_size
                test_loss = torch.mean(dirichlet_loss) + scaled_kl
                eval_loss += dirichlet_loss.item()
                eval_kl += kl.item()
                self.scheduler.step(test_loss)
                self.early_stopping(test_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

        avg_eval_loss = eval_loss / len(valid_loader)
        avg_eval_kl = eval_kl / len(valid_loader)
        writer.add_scalar('Validating loss by steps', avg_eval_loss, epoch)
        writer.add_scalar('Validating kl by steps', avg_eval_kl, epoch)
        return avg_eval_loss, avg_eval_kl

    def save_checkpoint(self, epoch):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch
        }
        os.makedirs(f'{self.config.save_dir}/checkpoint', exist_ok=True)
        torch.save(checkpoint, f'{self.config.save_dir}/checkpoint/ckpt_best_{epoch}.pth')
        torch.save(self.model.module, f'{self.config.save_dir}/model_{epoch}.pt')

    def train(self, train_loader, valid_loader):
        print("你又来炼丹辣！")
        os.makedirs(self.config.save_dir + "log/", exist_ok=True)
        writer = SummaryWriter(self.config.save_dir + "log/")

        for epoch in range(self.config.epochs):
            train_loss, train_kl = self.train_epoch(train_loader, epoch, writer)
            print(f"epoch: {epoch}, loss: {train_loss}, kl: {train_kl}")

            eval_loss, eval_kl = self.evaluate(valid_loader, epoch, writer)
            print(f"valid_loss: {eval_loss}, valid_kl: {eval_kl}")

            self.save_checkpoint(epoch)


def main(config):
    model = efficientnet_v2_s(num_classes=34)
    model = Morphics(model)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    # torch.compile(model, mode="max-autotune", dynamic=True, fullgraph=True)
    model = model.cuda()
    # Create data loaders
    train_data = GalaxyDataset(annotations_file=config.train_file, transform=config.transfer)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.WORKERS, pin_memory=True)

    valid_data = GalaxyDataset(annotations_file=config.valid_file,
                               transform=transforms.Compose([transforms.ToTensor()]), )
    valid_loader = DataLoader(dataset=valid_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.WORKERS, pin_memory=True)

    device_ids = [0, 1]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    bayesian_loss_func = metrics.ELBO(len(train_data)).to("cuda")
    optimizer = eval(config.optimizer)(model.parameters(), **config.optimizer_parm)

    trainer = Trainer(model=model, optimizer=optimizer, config=config)
    trainer.train(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == "__main__":
    init_rand_seed(1926)
    data_config = get_data_config()
    parser = argparse.ArgumentParser(description='Galaxy Classification Training')
    parser.add_argument('--train_file', type=str, default=data_config['train_file'],
                        help='Path to the training data annotations file')
    parser.add_argument('--valid_file', type=str, default=data_config['valid_file'],
                        help='Path to the validation data annotations file')
    parser.add_argument('--save_dir', type=str, default=data_config['save_dir'],
                        help='Directory to save logs, checkpoints, and trained models')
    parser.add_argument('--epochs', type=int, default=data_config['epochs'],
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=data_config['batch_size'],
                        help='Batch size for training and validation')
    parser.add_argument('--WORKERS', type=int, default=data_config['WORKERS'],
                        help='Number of workers for data loading')
    parser.add_argument('--optimizer', type=str, default=data_config['optimizer'],
                        help='Optimizer for training')
    parser.add_argument('--optimizer_parm', type=dict, default=data_config['optimizer_parm'],
                        help='Optimizer parameters')
    parser.add_argument('--transfer', type=callable, default=data_config['transfer'],
                        help='Transforms to apply to the input data')

    args = parser.parse_args()
    main(args)
