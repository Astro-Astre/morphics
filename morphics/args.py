import torchvision.transforms as transforms


def get_data_config():
    return {
        'root': "/data/public/renhaoye/morphics/",
        'save_dir': "/data/public/renhaoye/morphics/pth/",
        'catalog_loc': "/data/renhaoye/mw_catalog.csv",
        'train_file': "/data/public/renhaoye/morphics/dataset/train.txt",
        'valid_file': "/data/public/renhaoye/morphics/dataset/valid.txt",
        'model_architecture': "efficientnetv2_s",
        'epochs': 100,
        'batch_size': 256,
        'accelerator': "gpu",
        'gpus': 2,
        'nodes': 1,
        'patience': 8,
        'always_augment': False,
        'dropout_rate': 0.2,
        'mixed_precision': False,
        'WORKERS': 40,
        'transfer': transforms.Compose([
            transforms.RandomRotation(degrees=(0, 90)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
            transforms.ToTensor(),
        ]),
        'lr': 0.0001,
        'optimizer': "torch.optim.Adam",
        'optimizer_parm': {'lr': 0.0001, 'betas': (0.9, 0.999)}
    }


def get_transfer_config(seed=1926):
    return {
        'root': "/data/renhaoye/zoobot_astre_log/",
        'seed': seed,
        'save_dir': "/data/renhaoye/MorCG_DECaLS/pth/model_transfer_%d/" % seed,
        'train_file': "/data/renhaoye/MorCG_DECaLS/dataset/overlap_%d_train.txt" % seed,
        'valid_file': "/data/renhaoye/MorCG_DECaLS/dataset/overlap_1926_valid.txt",
        'num_workers': 20,
        'model_architecture': "Eff",
        'epochs': 30,
        'batch_size': 256,
        'accelerator': "gpu",
        'gpus': 2,
        'nodes': 1,
        'patience': 8,
        'always_augment': False,
        'dropout_rate': 0.2,
        'mixed_precision': False,
        'WORKERS': 12,
        'transfer': transforms.Compose([transforms.ToTensor()]),
        'lr': 1e-6,
        'optimizer': "torch.optim.Adam",
        'optimizer_parm': {'lr': 1e-6, 'betas': (0.9, 0.999), 'weight_decay': 0.2}
    }