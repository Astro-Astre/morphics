"""
author: renhaoye
function: 从sdss文件cutout出来
"""
import io
import multiprocessing
import os
from functools import partial
import numpy as np
from astropy.nddata import Cutout2D
import pandas as pd
from astropy.io import fits
import re
from astropy.wcs import FITSFixedWarning, WCS
import bz2
import warnings
from numpy import VisibleDeprecationWarning
from warnings import simplefilter

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
simplefilter(action='ignore', category=FutureWarning)


def get_image(i, files):
    ra, dec = files.loc[i, "ra"], files.loc[i, "dec"]
    # 定义文件名模板
    filename_template = "/data/public/renhaoye/morphics/dataset/sdss/raw_fits/{ra}_{dec}_{filter}.fits.bz2"
    for idx, filter_name in enumerate(['g', 'r', 'z'], start=1):
        fits_bz2_filename = filename_template.format(ra=ra, dec=dec, filter=filter_name, idx=idx)
        if not os.path.isfile(fits_bz2_filename):
            continue
        if not os.path.exists(f"/data/public/renhaoye/morphics/dataset/sdss/cutout/{ra}_{dec}_{filter_name}.fits"):
            try:
                with bz2.open(fits_bz2_filename, "rb") as f:
                    decompressed_data = f.read()
                with fits.open(io.BytesIO(decompressed_data)) as hdul:
                    hdu = hdul[0]
                    wcs = WCS(hdu.header)
                    x, y = wcs.all_world2pix([[float(ra), float(dec)]], 0)[0]
                    position = (x, y)
                    if 0 <= x < hdu.data.shape[1] and 0 <= y < hdu.data.shape[0]:
                        cutout_size = np.array([256, 256]) * 0.396
                        cutout = Cutout2D(hdu.data, position, cutout_size, wcs=wcs)
                        fits.writeto(
                            f'/data/public/renhaoye/morphics/dataset/sdss/cutout/{ra}_{dec}_{filter_name}.fits',
                            cutout.data, cutout.wcs.to_header(), overwrite=True)
            except:
                print(fits_bz2_filename)


def main():
    raw_fits_path = "/data/public/renhaoye/morphics/dataset/sdss/raw_fits"
    cutout_path = "/data/public/renhaoye/morphics/dataset/sdss/cutout"

    def extract_ra_dec(filename):
        match = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+)_\w+\.fits(\.bz2)?', filename)
        if match:
            ra, dec = match.group(1), match.group(2)
            return (ra, dec)  # 返回一个字符串元组
        return None

    # 获取已经在cutout文件夹中的(ra, dec)坐标
    cutout_files = os.listdir(cutout_path)
    cutout_ra_dec_set = set(
        extract_ra_dec(filename) for filename in cutout_files if extract_ra_dec(filename) is not None)

    # 获取原始数据中的(ra, dec)坐标并排除已在cutout文件夹中的坐标
    filenames = os.listdir(raw_fits_path)
    ra_dec_list = [extract_ra_dec(filename) for filename in filenames if
                   extract_ra_dec(filename) is not None and extract_ra_dec(filename) not in cutout_ra_dec_set]

    ra_dec_df = pd.DataFrame(ra_dec_list, columns=['ra', 'dec'])
    return ra_dec_df


if __name__ == '__main__':
    df = main()
    index = []
    for i in range(len(df)):
        index.append(i)
    p = multiprocessing.Pool(200)
    p.map(partial(get_image, files=df), index)
    p.close()
    p.join()
