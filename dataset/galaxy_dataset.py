import os
import time

import numpy as np
from torch.utils.data import Dataset
import pickle
from astropy.io import fits
from prefetch_generator import BackgroundGenerator


def load_img(file):
    """
    加载图像，dat和fits均支持，不过仅支持CxHxW
    :param filename: 传入文件名，应当为CHW
    :return: 返回CHW的ndarray
    """
    if ".fits" in file:
        with fits.open(file) as hdul:
            return hdul[0].data.astype(np.float32)
    elif ".dat" in file:
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        raise TypeError


class GalaxyDataset(Dataset):
    def __init__(self, annotations_file, transform):
        with open(annotations_file, "r") as file:
            imgs = []
            for line in file:
                start = time.time()
                line = line.strip("\n")
                line = line.rstrip("\n")
                words = line.split()
                label = str(line)[:-1].split("label:")
                hidden = list(label[-1][:].split(" "))
                hidden = hidden[:-1]
                votes = []
                for vote in hidden:
                    votes.append(float(vote))
                imgs.append((words[0], votes))
                end = time.time()-start
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = load_img(path)
        return np.nan_to_num(img), np.array(label)

    def __len__(self):
        return len(self.imgs)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TestDataset(Dataset):
    def __init__(self, dir, transform):
        files = os.listdir(dir)
        if "agmtn" in dir:
            temp = []
            for i in range(len(files)):
                if "_flipped" not in files[i] and "_rotated" not in files[i] and "_shifted" not in files[i]:
                    temp.append(files[i])
            files = temp.copy()
        imgs = []
        for i in range(len(files)):
            ra, dec = files[i].split(".fits")[0].split("_")
            imgs.append((dir+files[i] ,str(ra), str(dec)))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, ra, dec = self.imgs[index]
        img = load_img(path)
        return np.nan_to_num(img), str(ra), str(dec)

    def __len__(self):
        return len(self.imgs)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())