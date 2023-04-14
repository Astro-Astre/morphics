import os
import time


def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def format_time(remaining_time):
    days, seconds = divmod(int(remaining_time), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days:02d}Days,{hours:02d}Hours,{minutes:02d}Min,{seconds:02d}Seconds"


def monitor_directory(directory, interval=10):
    prev_count = count_files_in_directory(directory)
    total_length = count_lines("/data/public/renhaoye/urls.txt")
    start_time = time.time()

    while True:
        time.sleep(interval)
        current_count = count_files_in_directory(directory)

        if current_count != prev_count:
            diff = current_count - prev_count
            elapsed_time = time.time() - start_time
            speed = diff / elapsed_time
            remaining_files = total_length - current_count
            estimated_remaining_time = remaining_files / speed
            formatted_remaining_time = format_time(estimated_remaining_time)
            print(
                f"文件数量变化：{diff}，Finish：{current_count / total_length * 100:.6f}% 预计剩余时间：{formatted_remaining_time}")

            prev_count = current_count
            start_time = time.time()


def count_lines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return len(lines)


if __name__ == "__main__":
    directory_to_monitor = "/data/public/renhaoye/morphics/dataset/sdss/raw_fits"  # 修改为你要监控的文件夹路径
    monitor_directory(directory_to_monitor)
