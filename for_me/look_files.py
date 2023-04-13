import os
import re
# 指定文件夹路径
folder_path = '/data/public/renhaoye/morphics/dataset/sdss/raw_fits/'

# 初始化一个空的集合，用于存放所有唯一的 RA_DEC
unique_ra_dec = set()

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    # 判断是否符合 RA_DEC_i.fits 的命名规范
    match = re.match(r'(\d+\.\d+)_(\d+\.\d+)_(g|r|z)_(\d+)\.fits', filename)
    if match:
        # 提取 RA_DEC
        ra_dec = f"{match.group(1)}_{match.group(2)}"
        unique_ra_dec.add(ra_dec)

# 统计唯一 RA_DEC 的数量
ra_dec_count = len(unique_ra_dec)

print(f"文件夹 {folder_path} 中共有 {ra_dec_count} 个不同的 RA_DEC。")