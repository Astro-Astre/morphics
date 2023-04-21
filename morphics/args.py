import torchvision.transforms as transforms


def get_data_config():
    return {
        'root': "/data/public/renhaoye/morphics/",
        'save_dir': "/data/public/renhaoye/morphics/pth/",
        'catalog_loc': "/data/renhaoye/mw_catalog.csv",
        'train_file': "/data/public/renhaoye/morphics/dataset/train.txt",
        'valid_file': "/data/public/renhaoye/morphics/dataset/valid.txt",
        'model_architecture': "efficientnetv2_s",
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
    }
