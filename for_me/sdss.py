import multiprocessing
from functools import partial

import numpy as np
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u
import pandas as pd
from tqdm import tqdm
import warnings
from numpy import VisibleDeprecationWarning
from astropy.wcs import WCS

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from warnings import simplefilter
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)

simplefilter(action='ignore', category=FutureWarning)


# def get_image(i, files):
#     ra, dec = files.loc[i,"ra"], files.loc[i,"dec"]
#     pos = coords.SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
#     xid = SDSS.query_region(pos, spectro=False, radius=1 * u.arcsec)
#     if xid is None:
#         return "nan"
#     elif len(xid) == 1:
#         im = SDSS.get_images(matches=xid, band='grz')
#         for idx, image in enumerate(im, start=1):
#             fits.writeto(f'/data/public/renhaoye/morphics/dataset/sdss/raw_fits/{ra}_{dec}_{image[0].header["FILTER"]}_{idx}.fits',
#                          image[0].data, header=image[0].header, overwrite=True)
#     elif len(xid) > 1:
#         im = SDSS.get_images(coordinates=pos, band='grz')
#         for idx, image in enumerate(im, start=1):
#             hdu = image[0]
#             wcs = WCS(hdu.header)
#             x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
#             position = (x, y)
#
#             if 0 <= x < hdu.data.shape[1] and 0 <= y < hdu.data.shape[0]:
#                 # cutout_size = np.array([256, 256]) * 0.262
#                 cutout_size = np.array([256, 256]) * 0.396
#                 cutout = Cutout2D(hdu.data, position, cutout_size, wcs=wcs)
#                 fits.writeto(f'/data/public/renhaoye/morphics/dataset/sdss/raw_fits/{ra}_{dec}_{image[0].header["FILTER"]}_{idx}.fits', cutout.data, cutout.wcs.to_header(),overwrite=True)
def get_image(i, files):
    ra, dec = files.loc[i,"ra"], files.loc[i,"dec"]
    pos = coords.SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    im = SDSS.get_images(coordinates=pos, band='grz')
    for idx, image in enumerate(im, start=1):
        hdu = image[0]
        wcs = WCS(hdu.header)
        x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
        position = (x, y)
        if 0 <= x < hdu.data.shape[1] and 0 <= y < hdu.data.shape[0]:
            # cutout_size = np.array([256, 256]) * 0.262
            cutout_size = np.array([256, 256]) * 0.396
            cutout = Cutout2D(hdu.data, position, cutout_size, wcs=wcs)
            fits.writeto(f'/data/public/renhaoye/morphics/dataset/sdss/raw_fits/{ra}_{dec}_{image[0].header["FILTER"]}_{idx}.fits', cutout.data, cutout.wcs.to_header(),overwrite=True)

# cord = pd.DataFrame({'ra': df['ra'], 'dec': df['dec']})
# get = pd.DataFrame(
#     columns=['ra', 'dec', 'objid', 'run', 'rerun', 'camcol', 'field', 'z', 'plate', 'mjd', 'fiberID', 'specobjid',
#              'run2d'])

# num = 0
# for index, row in tqdm(cord.iterrows()):
#     ra = row["ra"]
#     dec = row["dec"]
#     o = get_xid(ra, dec)
#     if type(o) == str:
#         num += 1
#     elif len(o) == 1:
#         get = get.append(o, ignore_index=True)
#     elif len(o) > 1:
#         get = get.append(o.iloc[0], ignore_index=True)
# get.to_csv("/data/public/renhaoye/test.csv", index=False)
# print(num)

if __name__ == '__main__':
    with fits.open("/data/public/renhaoye/morphics/dataset/VAGC_MGS-m14_1777--20180116.fits") as hdul:
        table_data = hdul[1].data
        df = pd.DataFrame({'ra': table_data['RA'], 'dec': table_data['Dec']})
    # df = pd.read_csv("/data/public/renhaoye/morphics/dataset/astre_catalog.csv", )
    # loc = []
    # for i in range(len(df)):
    #     loc.append([df.loc[i, "ra"], df.loc[i, "dec"]])
    # loc = pd.DataFrame(loc, columns=["ra", "dec"])
    index = []
    for i in range(len(df)):
        index.append(i)
    # print(loc.loc[1, "ra"])
    # print(loc.iloc[1,:])
    # get_image(11, df)
    # for i in tqdm(range(100)):
    #     get_image(i, df)
    p = multiprocessing.Pool(128)
    p.map(partial(get_image, files=df), index)
    p.close()
    p.join()
