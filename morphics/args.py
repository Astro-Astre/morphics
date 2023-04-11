import torch.optim
import torchvision.transforms as transforms


def get_data_config():
    return {
        'root': "/data/renhaoye/bayes/",
        'save_dir': "/data/renhaoye/bayes/pth/model_Adam_1/",
        'catalog_loc': "/data/renhaoye/mw_catalog.csv",
        'train_file': "/data/renhaoye/MorCG_DECaLS/dataset/mw_train.txt",
        'valid_file': "/data/renhaoye/MorCG_DECaLS/dataset/mw_valid.txt",
        'model_architecture': "Eff",
        'epochs': 100,
        'batch_size': 256+64,
        'accelerator': "gpu",
        'gpus': 2,
        'nodes': 1,
        'patience': 8,
        'always_augment': False,
        'dropout_rate': 0.2,
        'mixed_precision': False,
        'WORKERS': 20,
        'transfer': transforms.Compose([transforms.ToTensor()]),
        'lr': 0.0001,
        'optimizer': "torch.optim.Adam",
        'optimizer_parm': {'lr': 0.0001, 'betas': (0.9, 0.999)}
    }


class transfer_config:
    root = "/data/renhaoye/zoobot_astre_log/"
    seedd = 1935
    save_dir = "/data/renhaoye/MorCG_DECaLS/pth/model_transfer_%d/" % seedd
    train_file = "/data/renhaoye/MorCG_DECaLS/dataset/overlap_%d_train.txt" % seedd
    valid_file = "/data/renhaoye/MorCG_DECaLS/dataset/overlap_1926_valid.txt"
    # train_file = "/data/renhaoye/MorCG_DECaLS/dataset/mw_overlap_train.txt"
    # valid_file = "/data/renhaoye/MorCG_DECaLS/dataset/mw_overlap_valid.txt"
    num_workers = 20
    model_architecture = "Eff"
    epochs = 30
    batch_size = 256
    accelerator = "gpu"
    gpus = 2
    nodes = 1
    patience = 8
    always_augment = False
    dropout_rate = 0.2
    mixed_precision = False
    WORKERS = 12
    transfer = transforms.Compose([transforms.ToTensor()])
    lr = 1e-6  # 学习率
    # optimizer = "torch.optim.AdamW"  # 优化器方法名称，eval()调用
    # optimizer_parm = {'lr': lr, 'weight_decay': 0.01}  # 优化器参数
    optimizer = "torch.optim.Adam"  # 优化器方法名称，eval()调用
    optimizer_parm = {'lr': lr, 'betas': (0.9, 0.999), 'weight_decay': 0.2}  # 优化器参数
    # torch.optim.Adam()
