# logger_utils.py
import logging
import os
import datetime

def setup_loggers(output_dir):
    """
    创建日志记录器，并将日志文件保存在以时间命名的文件夹中
    """
    # 创建一个以当前时间命名的文件夹
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outpath = os.path.join(output_dir, timestamp)
    os.makedirs(outpath, exist_ok=True)

    # 创建 logger1：记录原始数据（激活和权重的原始值）
    logger1 = logging.getLogger('logger1')
    logger1.setLevel(logging.INFO)
    log_file1 = os.path.join(outpath, 'original_data.log')
    fh1 = logging.FileHandler(log_file1)
    fh1.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger1.addHandler(fh1)

    # 创建 logger2：记录量化后的数据（激活和权重的量化值）
    logger2 = logging.getLogger('logger2')
    logger2.setLevel(logging.INFO)
    log_file2 = os.path.join(outpath, 'quantized_data.log')
    fh2 = logging.FileHandler(log_file2)
    fh2.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger2.addHandler(fh2)

    # 创建 logger3：记录缩放因子和零点
    logger3 = logging.getLogger('logger3')
    logger3.setLevel(logging.INFO)
    log_file3 = os.path.join(outpath, 'input_and_output.log')
    fh3 = logging.FileHandler(log_file3)
    fh3.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger3.addHandler(fh3)

    # 创建 logger4：记录整体数据
    logger4 = logging.getLogger('logger4')
    logger4.setLevel(logging.INFO)
    log_file4 = os.path.join(outpath, 'main.log')
    fh4 = logging.FileHandler(log_file4)
    fh4.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger4.addHandler(fh4)

    # 创建 logger5：记录量化误差数据
    logger5 = logging.getLogger('logger5')
    logger5.setLevel(logging.INFO)
    log_file5 = os.path.join(outpath, 'difference.log')
    fh5 = logging.FileHandler(log_file5)
    fh5.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger5.addHandler(fh5)

    return logger1, logger2, logger3, logger4, logger5, outpath

# 设置保存路径，日志文件将保存在以时间命名的文件夹中
output_dir = "./output_dir"

# 创建日志记录器
logger1, logger2, logger3, logger4, logger5, outpath = setup_loggers(output_dir)

