"""
author: renhaoye
function: 从sdss文件cutout出来
"""
import concurrent.futures
import io
import multiprocessing
import os
from functools import partial
import numpy as np
from astropy.nddata import Cutout2D
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u
import pandas as pd
from tqdm import tqdm
import warnings
from numpy import VisibleDeprecationWarning
from warnings import simplefilter
from astropy.io import fits
import re
from astropy.wcs import FITSFixedWarning, WCS
import bz2

path = "/data/public/renhaoye/74.28893482011404_-0.8270303005988735_g.fits.bz2"


# def get_image(i, files):
#     ra, dec = files.loc[i,"ra"], files.loc[i,"dec"]
#     pos = coords.SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
#     xid = SDSS.query_region(pos, spectro=False, radius=1 * u.arcsec)
#
#     im = SDSS.get_images(coordinates=pos, band='grz')
#     for idx, image in enumerate(im, start=1):
#         hdu = image[0]
#         wcs = WCS(hdu.header)
#         x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
#         position = (x, y)
#
#         if 0 <= x < hdu.data.shape[1] and 0 <= y < hdu.data.shape[0]:
#             # cutout_size = np.array([256, 256]) * 0.262
#             cutout_size = np.array([256, 256]) * 0.396
#             cutout = Cutout2D(hdu.data, position, cutout_size, wcs=wcs)
#             fits.writeto(f'/data/public/renhaoye/morphics/dataset/sdss/raw_fits/{ra}_{dec}_{image[0].header["FILTER"]}_{idx}.fits', cutout.data, cutout.wcs.to_header(),overwrite=True)

#
# def get_image(i, files):
#     ra, dec = files.loc[i, "ra"], files.loc[i, "dec"]
#     # 定义文件名模板
#     filename_template = "/data/public/renhaoye/morphics/dataset/sdss/raw_fits/{ra}_{dec}_{filter}_{idx}.fits.bz2"
#     for idx, filter_name in enumerate(['g', 'r', 'z'], start=1):
#         fits_bz2_filename = filename_template.format(ra=ra, dec=dec, filter=filter_name, idx=idx)
#         if not os.path.isfile(fits_bz2_filename):
#             continue
#         with bz2.open(fits_bz2_filename, "rb") as f:
#             decompressed_data = f.read()
#         with fits.open(io.BytesIO(decompressed_data)) as hdul:
#             hdu = hdul[0]
#             wcs = WCS(hdu.header)
#             x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
#             position = (x, y)
#             if 0 <= x < hdu.data.shape[1] and 0 <= y < hdu.data.shape[0]:
#                 cutout_size = np.array([256, 256]) * 0.396
#                 cutout = Cutout2D(hdu.data, position, cutout_size, wcs=wcs)
#                 fits.writeto(
#                     f'/data/public/renhaoye/morphics/dataset/sdss/raw_fits/{ra}_{dec}_{filter_name}_{idx}.fits',
#                     cutout.data, cutout.wcs.to_header(), overwrite=True)


def get_image(i, files):
    ra = 74.28893482011404
    dec = -0.8270303005988735
    with bz2.open(path, "rb") as f:
        decompressed_data = f.read()
    with fits.open(io.BytesIO(decompressed_data)) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
        position = (x, y)
        if 0 <= x < hdu.data.shape[1] and 0 <= y < hdu.data.shape[0]:
            cutout_size = np.array([256, 256]) * 0.396
            cutout = Cutout2D(hdu.data, position, cutout_size, wcs=wcs)
            fits.writeto(
                f'/data/public/renhaoye/{ra}_{dec}_g.fits',
                cutout.data, cutout.wcs.to_header(), overwrite=True)


def main():
    filenames = os.listdir("/data/public/renhaoye/dataset/sdss/raw_fits")
    def extract_ra_dec(filename):
        match = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+)_\w+\.fits\.bz2', filename)
        if match:
            ra, dec = match.group(1), match.group(2)
            return ra, dec
        return None
    ra_dec_list = [extract_ra_dec(filename) for filename in filenames if extract_ra_dec(filename) is not None]
    ra_dec_df = pd.DataFrame(ra_dec_list, columns=['ra', 'dec'])

if __name__ == '__main__':
    # main()
    get_image(0, 0)
