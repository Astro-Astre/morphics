import math
import random
from functools import partial
import numpy as np
import multiprocessing

import pandas as pd

random.seed(1926)
# SAVE_PATH = "/data/renhaoye/MorCG/dataset/"  # the head of the directory to save
import os
import pickle
from astropy.io import fits


def chw2hwc(img):
    ch1, ch2, ch3 = img[0], img[1], img[2]
    h, w = ch1.shape
    return np.concatenate((ch1.reshape(h, w, 1), ch2.reshape(h, w, 1), ch3.reshape(h, w, 1)), axis=2)


def hwc2chw(img):
    ch1, ch2, ch3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return np.array((ch1, ch2, ch3))


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


def save_fits(data: np.ndarray, filename: str):
    """
    将ndarray保存成fits文件
    :param data: 待保存数据
    :param filename: 保存文件名
    :return:
    """
    if len(data.shape) == 2:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    elif data.shape[-1] == 3:
        g, r, z = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        data = np.array((g, r, z))
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    elif data.shape[0] == 3:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    else:
        raise RuntimeError


class Img:
    def __init__(self, image, rows, cols, center=None):
        self.g_dst = None
        self.r_dst = None
        self.z_dst = None
        if center is None:
            center = [0, 0]
        self.dst = None
        self.g_src = image[0]
        self.r_src = image[1]
        self.z_src = image[2]
        self.transform = None
        self.rows = rows
        self.cols = cols
        self.center = center  # rotate center

    def Shift(self, delta_x, delta_y):  # 平移
        # delta_x>0 shift left  delta_y>0 shift top
        self.transform = np.array([[1, 0, delta_x],
                                   [0, 1, delta_y],
                                   [0, 0, 1]])

    def Flip(self):  # vertically flip
        self.transform = np.array([[-1, 0, self.rows - 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])

    def Rotate(self, beta):  # rotate
        # beta<0 rotate clockwise
        self.transform = np.array([[math.cos(beta), -math.sin(beta), 0],
                                   [math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])

    def Process(self):
        self.g_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.r_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.z_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([i - self.center[0], j - self.center[1], 1])
                [x, y, z] = np.dot(self.transform, src_pos)
                x = int(x) + self.center[0]
                y = int(y) + self.center[1]
                if x >= self.rows or y >= self.cols:
                    self.g_dst[i][j] = 0.
                    self.r_dst[i][j] = 0.
                    self.z_dst[i][j] = 0.
                else:
                    self.g_dst[i][j] = self.g_src[int(x)][int(y)]
                    self.r_dst[i][j] = self.r_src[int(x)][int(y)]
                    self.z_dst[i][j] = self.z_src[int(x)][int(y)]
        self.dst = np.array((self.g_dst, self.r_dst, self.z_dst))


def flip(img, save_dir):
    height, width = img.shape[1:3]
    output = Img(img, height, width, [0, 0])
    output.Flip()
    output.Process()
    save_fits(output.dst, save_dir + "_flipped.fits")


def rotate(img, save_dir):
    seed = random.randint(0, 90)
    height, width = img.shape[1:3]
    output = Img(img, height, width, [height / 2, width / 2])
    output.Rotate(seed)
    output.Process()
    save_fits(output.dst, save_dir + "_rotated.fits")


def shift(img, save_dir, pixel):
    height, width = img.shape[1:3]
    output = Img(img, height, width, [0, 0])
    output.Shift(pixel, 0)
    output.Process()
    save_fits(output.dst, save_dir + "_shifted.fits")


def augmentation(i, files):
    dst_dir = "/data/renhaoye/MorCG/dataset/out_decals/overlap_agmtn/"  # 目标存放文件夹
    src_dir = "/data/renhaoye/MorCG/dataset/out_decals/scaled/"  # 原始路径
    src_file = src_dir + files[i]  # 原始图片绝对路径
    ra_dec = files[i].split(".fits")[0]
    dst_file = dst_dir + ra_dec  # 保存绝对路径 不带扩展名
    scaled = load_img(src_file)
    save_fits(scaled, dst_dir + files[i])
    flip(scaled, dst_file)
    shift(scaled, dst_file, random.randint(1, 10))
    rotate(scaled, dst_file)


import astropy.units as u
from astropy.coordinates import SkyCoord


def match(df_1, df_2, pixel, df1_name):
    """
    match two catalog
    :param df_1:
    :param df_2:
    :return:
    """
    sdss = SkyCoord(ra=df_1.ra * u.degree, dec=df_1.dec * u.degree)
    decals = SkyCoord(ra=df_2.ra * u.degree, dec=df_2.dec * u.degree)
    idx, d2d, d3d = sdss.match_to_catalog_sky(decals)
    max_sep = pixel * 0.262 * u.arcsec
    distance_idx = d2d < max_sep

    sdss_matches = df_1.iloc[distance_idx]
    matches = idx[distance_idx]
    decal_matches = df_2.iloc[matches]
    test = sdss_matches.loc[:].rename(columns={"ra": "%s" % df1_name[0], "dec": "%s" % df1_name[1]})
    test.insert(0, 'ID', range(len(test)))
    decal_matches.insert(0, 'ID', range(len(decal_matches)))
    new_df = pd.merge(test, decal_matches, how="inner", on=["ID"])
    return new_df.drop("ID", axis=1)


if __name__ == "__main__":
    overlap = pd.read_csv("/data/renhaoye/MorCG_DECaLS/dataset/overlap.csv")
    loc_files = os.listdir("/data/renhaoye/MorCG/dataset/out_decals/scaled/")
    loc = []
    for i in range(len(loc_files)):
        loc.append([float(loc_files[i].split("_")[0]), float(loc_files[i].split("_")[1].split(".fits")[0])])
    loc = pd.DataFrame(loc, columns=["ra", "dec"])
    # print(loc.head())
    overlap_true = match(loc, overlap, 5, ["MGS_ra", "MGS_dec"]).drop(columns=["ra", "dec"])
    overlap_true = overlap_true.rename(columns={"MGS_ra": "ra", "MGS_dec": "dec"})
    # print(overlap_true.columns)
    # print(overlap_true.iloc[0, :])
    # overlap_true = match(loc, overlap, 2, ["real_ra", "real_dec"]).drop(columns=["ra", "dec"]).rename(columns={"real_ra": "ra", "real_dec": "dec"})
    src = []
    for i in range(len(overlap)):
        src.append("%s_%s.fits" % (overlap_true.loc[i, "ra"], overlap_true.loc[i, "dec"]))
    index = []
    for i in range(len(src)):
        index.append(i)
    p = multiprocessing.Pool(15)
    p.map(partial(augmentation, files=src), index)
    p.close()
    p.join()
