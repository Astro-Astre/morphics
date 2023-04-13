import concurrent.futures
import pandas as pd
from tqdm import tqdm
import warnings
from numpy import VisibleDeprecationWarning
from warnings import simplefilter
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
simplefilter(action='ignore', category=FutureWarning)
IMAGING_URL_SUFFIX = ('{base}/dr{dr}/{instrument}/photoObj/frames/'
                      '{rerun}/{run}/{camcol}/'
                      'frame-{band}-{run:06d}-{camcol}-'
                      '{field:04d}.fits.bz2')


def get_url(i, files, f):
    for b in ["g", "r", "z"]:
        linkstr = IMAGING_URL_SUFFIX
        instrument = 'eboss'
        link = linkstr.format(base='https://data.sdss.org/sas', run=files.loc[i, 'run'],
                              dr=17, instrument=instrument,
                              rerun=files.loc[i, 'rerun'], camcol=files.loc[i, 'camcol'],
                              field=files.loc[i, 'field'], band=b)
        f.write(f"{link} /data/public/renhaoye/morphics/dataset/sdss/raw_fits/{files.loc[i, 'ra']}_{files.loc[i, 'dec']}_{b}.fits.bz2")
        f.write("\n")


def main():
    df = pd.read_csv("/data/public/renhaoye/morphics/dataset/sdss_mgs.csv")
    index = list(range(len(df)))
    # 使用 ThreadPoolExecutor 实现并发
    # with concurrent.futures.ThreadPoolExecutor(max_workers=128 + 64) as executor:
    #     futures = [executor.submit(get_url, i, df) for i in index]
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    #         pass

    with open("/data/public/renhaoye/urls.txt", "a") as f:
        for idx in tqdm(range(len(df))):
            get_url(idx, df, f)


if __name__ == '__main__':
    main()
