import metrics
from models.BayesianModels.BayesianEffnetv2 import EffNetV2, effnetv2_m, effnetv2_l, effnetv2_xl, effnetv2_s
import random
from torch.backends import cudnn
from pytorch_galaxy_datasets.galaxy_dataset import *
from args import *
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from shared import label_metadata, schemas
from training import losses

from tqdm import tqdm

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
init_rand_seed(1926)


def get_avg_loss(loss):
    avg_loss = np.zeros(loss.shape[1])
    for i in range(loss.shape[1]):
        avg_loss[i] += loss[:, i].sum()
    return avg_loss


class trainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.question_answer_pairs = label_metadata.gz2_pairs  # 问题？
        self.dependencies = label_metadata.gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)

    def dirichlet_loss_func(self, preds, labels):  # pytorch convention is preds, labels
        return losses.calculate_multiquestion_loss(labels, preds,
                                                   self.schema.question_index_groups)  # my and sklearn convention is labels, preds

    def train(self, train_loader, valid_loader):
        print("你又来炼丹辣！")
        start = -1
        mkdir(self.config.save_dir + "log/")
        writer = torch.utils.tensorboard.SummaryWriter(self.config.save_dir + "log/")
        # writer.add_graph(model.module, torch.rand(1, 3, 256, 256).cuda())
        info = data_config()
        with open(data_config.save_dir + "info.txt", "w") as w:
            for each in info.__dir__():
                attr_name = each
                attr_value = info.__getattribute__(each)
                w.write(str(attr_name) + ':' + str(attr_value) + "\n")
        for epoch in range(start + 1, self.config.epochs):
            train_loss = 0
            self.model.train()
            for i, (X, label) in enumerate(tqdm(train_loader)):
                label = torch.as_tensor(label, dtype=torch.long)
                X = X.cuda()
                label = label.cuda()
                out, kl = self.model(X)  # 正向传播
                beta = metrics.get_beta(i - 1, len(train_loader), 0.1, epoch, data_config.epochs)
                loss_value = torch.mean(self.dirichlet_loss_func(out, label)) * len(train_loader) + beta * torch.mean(kl)  # 求损失值, out is concentration
                self.optimizer.zero_grad()  # 优化器梯度归零
                loss_value.backward()  # 反向转播，刷新梯度值
                self.optimizer.step()
                train_loss += loss_value
            losses = (train_loss / len(train_loader))
            writer.add_scalar('Training loss by steps', losses, epoch)
            print("epoch: ", epoch)
            print("loss: ", losses)
            eval_loss = 0
            with torch.no_grad():
                self.model.eval()
                for X, label in valid_loader:
                    label = torch.as_tensor(label, dtype=torch.long)
                    X = X.cuda()
                    label = label.cuda()
                    test_out, test_kl = self.model(X)
                    test_loss = torch.mean(self.dirichlet_loss_func(test_out, label)) + torch.mean(test_kl)
                    eval_loss += test_loss
            eval_losses = (eval_loss / len(valid_loader))
            writer.add_scalar('Validating loss by steps', eval_losses, epoch)
            print("valid_loss: " + str(eval_losses))
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": epoch
            }
            mkdir('%s/checkpoint' % data_config.save_dir)
            torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' % (self.config.save_dir, str(epoch)))
            torch.save(self.model.module, '%s/model_%d.pt' % (self.config.save_dir, epoch))


train_data = GalaxyDataset(annotations_file=data_config.train_file, transform=data_config.transfer)
train_loader = DataLoader(dataset=train_data, batch_size=data_config.batch_size,
                          shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

valid_data = GalaxyDataset(annotations_file=data_config.valid_file, transform=data_config.transfer)
valid_loader = DataLoader(dataset=valid_data, batch_size=data_config.batch_size,
                          shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

model = effnetv2_s(num_classes=34)
model = model.cuda()
device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids)
bayesian_loss_func = metrics.ELBO(len(train_data)).to("cuda")
optimizer = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)

Trainer = trainer(model=model, optimizer=optimizer, config=data_config)
Trainer.train(train_loader=train_loader, valid_loader=valid_loader)
