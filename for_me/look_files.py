# import os
# import time
#
#
# def count_files_in_directory(directory):
#     return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
#
#
# def format_time(remaining_time):
#     days, seconds = divmod(int(remaining_time), 86400)
#     hours, seconds = divmod(seconds, 3600)
#     minutes, seconds = divmod(seconds, 60)
#     return f"{days:02d}Days,{hours:02d}Hours,{minutes:02d}Min,{seconds:02d}Seconds"
#
#
# def monitor_directory(directory, interval=10):
#     prev_count = count_files_in_directory(directory)
#     total_length = count_lines("/data/public/renhaoye/urls.txt")
#     start_time = time.time()
#
#     while True:
#         time.sleep(interval)
#         current_count = count_files_in_directory(directory)
#
#         if current_count != prev_count:
#             diff = current_count - prev_count
#             elapsed_time = time.time() - start_time
#             speed = diff / elapsed_time
#             remaining_files = total_length - current_count
#             estimated_remaining_time = remaining_files / speed
#             formatted_remaining_time = format_time(estimated_remaining_time)
#             print(
#                 f"文件数量变化：{diff}，Finish：{current_count / total_length * 100:.6f}% 预计剩余时间：{formatted_remaining_time}")
#
#             prev_count = current_count
#             start_time = time.time()
#
#
# def count_lines(filename):
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         return len(lines)
#
#
# if __name__ == "__main__":
#     directory_to_monitor = "/data/public/renhaoye/morphics/dataset/sdss/raw_fits"  # 修改为你要监控的文件夹路径
#     monitor_directory(directory_to_monitor)


# import os
#
#
# def get_files_with_extension(directory, extension):
#     return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
#
#
# def get_size(file_path):
#     return os.path.getsize(file_path)
#
#
# def remove_smallest_files(directory, extension, percentage):
#     files = get_files_with_extension(directory, extension)
#     if not files:
#         print(f"No {extension} files found in {directory}")
#         return
#
#     files_with_sizes = [(f, get_size(f)) for f in files]
#     files_with_sizes.sort(key=lambda x: x[1])
#
#     num_files_to_remove = int(len(files_with_sizes) * percentage)
#     removed_files = files_with_sizes[:num_files_to_remove]
#
#     for file_path, size in removed_files:
#         if size == 0:
#             os.remove(file_path)
#             print(f"Removed {file_path} with size {size} bytes")
#
#
# if __name__ == "__main__":
#     directory = "/data/public/renhaoye/morphics/dataset/sdss/raw_fits/"  # 替换为你的文件夹路径
#     extension = ".bz2"
#     percentage = 0.001  # 更改为您希望删除的文件比例，这里是 10%
#
#     remove_smallest_files(directory, extension, percentage)

import os


def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines


def write_txt_file(lines):
    with open("/data/public/renhaoye/urlss.txt", 'w') as f:
        f.writelines(lines)


def get_existing_filenames(directory):
    return set(os.listdir(directory))


def filter_urls_file(txt_file_path, directory):
    existing_files = get_existing_filenames(directory)

    lines = read_txt_file(txt_file_path)
    filtered_lines = []

    for line in lines:
        file_name = line.strip().split('/')[-1]
        if file_name not in existing_files:
            filtered_lines.append(line)

    write_txt_file(filtered_lines)


if __name__ == "__main__":
    txt_file_path = "/data/public/renhaoye/urls.txt"  # 替换为你的txt文件路径
    directory = "/data/public/renhaoye/morphics/dataset/sdss/raw_fits/"  # 替换为你的文件夹路径

    filter_urls_file(txt_file_path, directory)
