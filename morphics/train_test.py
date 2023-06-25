import random
from torch.backends import cudnn
from dataset.galaxy_dataset import *
from args import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from utils import schemas
from training import losses, metrics
import argparse
from utils.utils import *
from torchvision.models import *
from models.morphics import Morphics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.utils import *


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Trainer:
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

    def train_epoch(self, train_loader, epoch, writer):
        train_loss = 0
        self.model.train()
        for i, (X, label) in enumerate(train_loader):
            label = torch.as_tensor(label, dtype=torch.long).to("cuda:1")
            X = X.to("cuda:1")
            output_ = []
            for mc_run in range(self.config.sample):
                output, _ = self.model(X)
                # output = losses.MultiSoftmax(output, self.schema.question_index_groups)
                output_.append(output)
            output = torch.mean(torch.stack(output_), dim=0)
            dirichlet_loss = torch.mean(self.dirichlet_loss_func(output, label), dim=0)
            loss_value = torch.mean(dirichlet_loss)
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            train_loss += torch.mean(dirichlet_loss).item()
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Training loss by steps', avg_train_loss, epoch)
        return avg_train_loss

    def evaluate(self, valid_loader, epoch, writer):
        eval_loss = 0
        with torch.no_grad():
            self.model.eval()
            for X, label in valid_loader:
                label = torch.as_tensor(label, dtype=torch.long).to("cuda:1")
                X = X.to("cuda:1")
                output_ = []
                for mc_run in range(self.config.sample):
                    output, _ = self.model(X)
                    output_.append(output)
                output = torch.mean(torch.stack(output_), dim=0)
                dirichlet_loss = torch.mean(self.dirichlet_loss_func(output, label), dim=0)
                eval_loss += torch.mean(dirichlet_loss).item()

        avg_eval_loss = eval_loss / len(valid_loader)
        writer.add_scalar('Validating loss by steps', avg_eval_loss, epoch)
        return avg_eval_loss

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
            train_loss = self.train_epoch(train_loader, epoch, writer)
            print(f"epoch: {epoch}, loss: {train_loss}")
            eval_loss = self.evaluate(valid_loader, epoch, writer)
            self.scheduler.step(eval_loss)
            print(f"valid_loss: {eval_loss}")
            self.save_checkpoint(epoch)
            self.early_stopping(eval_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input: 3 channels, Output: 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Input: 16 channels, Output: 32 channels
        self.dropout1 = nn.Dropout2d(p=0.5)  # Dropout after the last convolutional layer
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Assuming the input image size is 32x32
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout after the first fully connected layer
        self.fc2 = nn.Linear(128, 34)  # Output: 34 classes

    def forward(self, x):
        # Forward pass through the first convolutional layer, then ReLU, then max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        # Forward pass through the second convolutional layer, then ReLU, then max pooling, then dropout
        x = self.dropout1(F.max_pool2d(F.relu(self.conv2(x)), 2))

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)

        # Forward pass through the first fully connected layer, then ReLU, then dropout
        x = self.dropout2(F.relu(self.fc1(x)))

        # Forward pass through the second fully connected layer
        x = self.fc2(x)

        return x


def main(config):
    model = SimpleCNN()
    model = Morphics(model)
    model = model.to("cuda:1")
    # print(model)
    subset_size = 10000
    subset_indices = list(range(subset_size))

    train_data = GalaxyDataset(annotations_file=config.train_file, transform=config.transfer)
    # train_data = Subset(train_data, subset_indices)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.WORKERS, pin_memory=True)

    valid_data = GalaxyDataset(annotations_file=config.valid_file,
                               transform=transforms.Compose([transforms.ToTensor()]), )
    # valid_data = Subset(valid_data, subset_indices)
    valid_loader = DataLoader(dataset=valid_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    device_ids = [1]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    trainer = Trainer(model=model, optimizer=optimizer, config=config)
    trainer.train(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == "__main__":
    init_rand_seed(1926)
    data_config = get_data_config()
    parser = argparse.ArgumentParser(description='Morphics: Galaxy Classification Training')
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
    parser.add_argument('--betas', type=tuple, default=data_config['betas'],
                        help='Optimizer parameters')
    parser.add_argument('--transfer', type=callable, default=data_config['transfer'],
                        help='Transforms to apply to the input data')
    parser.add_argument('--lr', type=float, default=data_config['lr'],
                        help='Learning rate for training')
    parser.add_argument('--patience', type=int, default=data_config['patience'],
                        help='Patience for early stopping')
    parser.add_argument('--phase', type=str, default=data_config['phase'],
                        help='Phase for training')
    parser.add_argument('--sample', type=int, default=data_config['sample'],
                        help='Sample nums for training')
    parser.add_argument('--dropout_rate', type=float, default=data_config['dropout_rate'],
                        help='Dropout rate for training')
    args = parser.parse_args()
    main(args)
