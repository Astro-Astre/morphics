import multiprocessing
import os
from functools import partial
import numpy as np
import pandas as pd
from astropy.io import fits
import re
from astropy.wcs import WCS


def main(input_folder):
    # 获取文件夹中的文件名
    filenames = os.listdir(input_folder)

    # 提取(ra, dec, filter)信息
    def extract_info(filename):
        match = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+)_(\w+)\.fits', filename)
        if match:
            ra, dec, filter_name = match.group(1), match.group(2), match.group(3)
            return (ra, dec, filter_name)
        return None

    # 创建一个字典，其中键是(ra, dec)，值是滤波器列表
    coord_filter_dict = {}
    for filename in filenames:
        info = extract_info(filename)
        if info is not None:
            ra_dec = (info[0], info[1])
            filter_name = info[2]
            if ra_dec in coord_filter_dict:
                coord_filter_dict[ra_dec].add(filter_name)
            else:
                coord_filter_dict[ra_dec] = {filter_name}

    # 选择具有g、r、z滤波器的(ra, dec)坐标
    valid_ra_dec = [ra_dec for ra_dec, filters in coord_filter_dict.items() if {'g', 'r', 'z'} == filters]

    # 创建Pandas DataFrame
    ra_dec_df = pd.DataFrame(valid_ra_dec, columns=['ra', 'dec'])
    ra_dec_df.to_csv("/data/public/renhaoye/morphics/dataset/sdss_integration.csv")
    return ra_dec_df


def integration(i, df):
    channels = []
    wcs_list = []
    ra, dec = float(df.loc[i, "ra"]), float(df.loc[i, "dec"])

    for filter_name in ['g', 'r', 'z']:
        filename = f"{ra}_{dec}_{filter_name}.fits"
        filepath = os.path.join("/data/public/renhaoye/morphics/dataset/sdss/cutout/", filename)
        with fits.open(filepath) as hdul:
            channel_data = hdul[0].data
            channels.append(channel_data)
            wcs_list.append(WCS(hdul[0].header))

    # 计算(ra, dec)在每个通道的像素坐标
    pix_coords = [wcs.wcs_world2pix([(ra, dec)], 0)[0] for wcs in wcs_list]
    x_coords, y_coords = zip(*pix_coords)

    # 计算输出图像的尺寸
    max_width = int(max([channel_data.shape[1] for channel_data in channels]) + max(x_coords) - min(x_coords))
    max_height = int(max([channel_data.shape[0] for channel_data in channels]) + max(y_coords) - min(y_coords))

    # 创建输出图像
    output_shape = (3, max_height, max_width)
    output_data = np.zeros(output_shape)

    # 将每个通道的数据复制到输出图像上的正确位置
    for i, channel_data in enumerate(channels):
        x_min = int(x_coords[i] - min(x_coords))
        x_max = x_min + channel_data.shape[1]
        y_min = int(y_coords[i] - min(y_coords))
        y_max = y_min + channel_data.shape[0]
        output_data[i, y_min:y_max, x_min:x_max] = channel_data

    # 保存叠加后的FITS文件并写入WCS信息
    output_filename = f"{ra}_{dec}.fits"
    output_filepath = os.path.join("/data/public/renhaoye/morphics/dataset/sdss/integration/", output_filename)
    hdu = fits.PrimaryHDU(output_data)
    hdu.header.update(wcs_list[0].to_header())
    hdu.writeto(output_filepath, overwrite=True)


if __name__ == '__main__':
    # 示例用法
    input_folder = "/data/public/renhaoye/morphics/dataset/sdss/cutout/"
    output_folder = "/data/public/renhaoye/morphics/dataset/sdss/integration/"
    df = main(input_folder)
    # integration(1, df)
    index = []
    for i in range(len(df)):
        index.append(i)
    p = multiprocessing.Pool(255)
    p.map(partial(integration, df=df), index)
    p.close()
    p.join()
