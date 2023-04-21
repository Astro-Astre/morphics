import torchvision.transforms as transforms


def transfer_config():
    return {
        'root': "/data/public/renhaoye/morphics/",
        'save_dir': "/data/public/renhaoye/morphics/pth_cdan/",
        'train_source': "/data/public/renhaoye/morphics/dataset/train_sdss_raw.txt",
        'train_target': "/data/public/renhaoye/morphics/dataset/train_des_raw.txt",
        'valid_target': "/data/public/renhaoye/morphics/dataset/valid_des_raw.txt",
        'epochs': 1000,
        'batch_size': 128,
        'patience': 8,
        'dropout_rate': 0.2,
        'WORKERS': 128,
        'transfer': transforms.Compose([
            transforms.RandomRotation(degrees=(0, 90)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
            transforms.ToTensor(),
        ]),
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'phase': "train",
        'sample': 1,
        'model_path': "/data/public/renhaoye/morphics/pth_stn/model_19.pt",
        'randomized': True,
        'bottleneck_dim': 64,
        'device': 'cuda:1',
        'finetune': True,
        'no_pool': True,
        'randomized_dim': 8,
        'num_classes': 34,
    }
