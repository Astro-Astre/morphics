"""
author: renhaoye
function: 从sdss上查询信息
"""
import concurrent.futures
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
from warnings import simplefilter
from astropy.io import fits
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
simplefilter(action='ignore', category=FutureWarning)


def get_image(i, files):
    ra, dec = files.loc[i, "ra"], files.loc[i, "dec"]
    pos = coords.SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    xid = SDSS.query_region(pos, spectro=False, radius=3 * u.arcsec, timeout=180)
    # 将xid写入文件
    # with open("/data/public/renhaoye/xid.txt", "a") as f:
    #     table_str = xid.pformat(max_lines=-1, show_name=False)
    #     for e in range(len(table_str)):
    #         f.write(str(table_str[e]))
    #         f.write("\n")


def main():
    with fits.open("/data/public/renhaoye/morphics/dataset/VAGC_MGS-m14_1777--20180116.fits") as hdul:
        table_data = hdul[1].data
        df = pd.DataFrame({'ra': table_data['RA'], 'dec': table_data['Dec']})
    index = list(range(len(df)))

    get_image(1, df)
    # p = multiprocessing.Pool(128)
    # p.map(partial(get_image, files=df), index)
    # p.close()
    # p.join()
    # # 使用 ThreadPoolExecutor 实现并发
    # with concurrent.futures.ThreadPoolExecutor(max_workers=128 + 64) as executor:
    #     futures = [executor.submit(get_image, i, df) for i in index]
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    #         pass


if __name__ == '__main__':
    main()
