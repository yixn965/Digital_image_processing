import torch
import argparse
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter
from copy import deepcopy
from utils.logger_utils import logger1, logger2, logger3, logger4, logger5, outpath
from .quantizer import UniformQuantizer_ar,UniformQuantizer_diff,UniformQuantizer_group,UniformQuantizer_group_diff,UniformQuantizer_group_scaling,UniformQuantizer_scale_channels

import os
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import datetime
# import matplotlib.pyplot as plt
import os
# import pandas as pd
import csv
from sklearn.cluster import KMeans
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import random
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import copy
from functools import partial
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
import os
from datetime import datetime
import numpy as np

# 输出目录
# output_dir1 = "/opt/data/private/GaoJing/deeplearnng/mar/plot/fp_16_random_images/activations"
# output_dir2 = "/opt/data/private/GaoJing/deeplearnng/mar/plot/quant_16_random_images/before/activations"
# output_dir3 = "/opt/data/private/GaoJing/deeplearnng/mar/plot/quant_16_random_images/after/activations"
# output_dir4 = "/opt/data/private/GaoJing/deeplearnng/mar/plot/fp_16_random_images/weights"
# output_dir5 = "/opt/data/private/GaoJing/deeplearnng/mar/plot/quant_16_random_images/before/weights"
# output_dir6 = "/opt/data/private/GaoJing/deeplearnng/mar/plot/quant_16_random_images/after/weights"
output_dir1 = output_dir2 = output_dir3 = output_dir4 = output_dir5 = output_dir6 =outpath
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)
os.makedirs(output_dir3, exist_ok=True)
os.makedirs(output_dir4, exist_ok=True)
os.makedirs(output_dir5, exist_ok=True)
os.makedirs(output_dir6, exist_ok=True)
# 定义保存路径
OUTPUT_DIR = "/opt/data/private/GaoJing/deeplearnng/mar/plot"

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_statistics(data, title, filename, output_dir):
    """
    将数据保存为排序的 CSV 文件
    """
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().detach().numpy().flatten()  # PyTorch 张量处理
    elif isinstance(data, np.ndarray):
        data_np = data.flatten()  # NumPy 数组处理
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    data_sorted = np.sort(data_np)  # 对数据进行排序
    
    # 保存为 CSV 文件
    save_path = os.path.join(output_dir, filename)  # 拼接输出目录和文件名
    df = pd.DataFrame(data_sorted, columns=[title])  # 创建 DataFrame
    df.to_csv(save_path, index=False)  # 保存为 CSV 文件
    # logger4.info(f"Saved CSV: {save_path}")

def save_and_plot(data, title, filename, output_dir):
    """
    保存统计信息并绘制分布图到指定路径
    """
    # 保存统计信息（CSV）
    save_statistics(data, title, filename.replace(".png", ".csv"), output_dir)
    
    # # 确保 data 是 NumPy 数组
    # if isinstance(data, torch.Tensor):
    #     data_np = data.cpu().detach().numpy().flatten()
    # elif isinstance(data, np.ndarray):
    #     data_np = data.flatten()
    # else:
    #     raise TypeError(f"Unsupported data type: {type(data)}")
    
    # # 绘制分布并保存为图像
    # plt.figure(figsize=(8, 6))  # 设置图像大小
    # plt.hist(data_np, bins=50, alpha=0.75)  # 绘制直方图
    # plt.title(title)  # 设置标题
    # plt.xlabel('Value')  # X 轴标签
    # plt.ylabel('Frequency')  # Y 轴标签
    # plt.grid(True)  # 显示网格

    # # 保存图像到指定路径
    # save_path = os.path.join(output_dir, filename)  # 拼接保存路径
    # plt.savefig(save_path)  # 保存图像
    # plt.close()  # 关闭当前图像
    # # logger4.info(f"Saved plot: {save_path}")


def save_quantization_statistics0(x, w, x_quant, w_quant, step, i, layer_name):
    """
    保存量化前后逐通道的最值、均值以及量化前后逐通道差值的最值。
    """
    if x.ndim == 3:
        x = x.view(-1, x.size(2))  # 合并前两个维度，例如 [64, 64, 3072] -> [4096, 3072]
    
    dim = 1  # 只针对权重，1是按照输出通道来做（主流），0是按照输入通道来做
    
    # 量化前逐通道最值和均值
    x_max_pre = x.max(dim=0).values.cpu().tolist()
    x_min_pre = x.min(dim=0).values.cpu().tolist()
    x_mean_pre = x.mean(dim=0).cpu().tolist()
    w_max_pre = w.max(dim=dim).values.cpu().tolist()
    w_min_pre = w.min(dim=dim).values.cpu().tolist()
    w_mean_pre = w.mean(dim=dim).cpu().tolist()

    # 量化后逐通道最值和均值
    x_max_post = x_quant.max(dim=0).values.cpu().tolist()
    x_min_post = x_quant.min(dim=0).values.cpu().tolist()
    x_mean_post = x_quant.mean(dim=0).cpu().tolist()
    w_max_post = w_quant.max(dim=dim).values.cpu().tolist()
    w_min_post = w_quant.min(dim=dim).values.cpu().tolist()
    w_mean_post = w_quant.mean(dim=dim).cpu().tolist()

    # 量化前后差值逐通道最值（绝对差后求最值）
    diff_x = (x - x_quant).abs()
    diff_w = (w - w_quant).abs()
    diff_x_max = diff_x.max(dim=0).values.cpu().tolist()
    diff_x_min = diff_x.min(dim=0).values.cpu().tolist()
    diff_x_mean = diff_x.mean(dim=0).cpu().tolist()
    diff_w_max = diff_w.max(dim=dim).values.cpu().tolist()
    diff_w_min = diff_w.min(dim=dim).values.cpu().tolist()
    diff_w_mean = diff_w.mean(dim=dim).cpu().tolist()

    # 定义新的文件夹路径
    base_paths = {
        "x": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/all_channels_activations/x",
        "w": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/all_channels_weights/w",
        "x_quant": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/all_channels_activations/x_quant",
        "w_quant": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/all_channels_weights/w_quant",
        "x_diff": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/all_channels_activations/x_diff",
        "w_diff": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/all_channels_weights/w_diff"
    }

    # 为每个基础路径创建以 step 为命名的子目录
    for key, base_dir in base_paths.items():
        step_dir = os.path.join(base_dir, f"step_{step}")
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)  # 如果目录不存在，则创建

    # 保存数据到对应的目录
    pd.DataFrame({
        "Channel": list(range(len(x_max_pre))), 
        "Max": x_max_pre, 
        "Min": x_min_pre, 
        "Mean": x_mean_pre
    }).to_csv(
        os.path.join(base_paths["x"], f"step_{step}/x_step{step}_timestep{i}.csv"), index=False
    )
    
    pd.DataFrame({
        "Channel": list(range(len(w_max_pre))), 
        "Max": w_max_pre, 
        "Min": w_min_pre, 
        "Mean": w_mean_pre
    }).to_csv(
        os.path.join(base_paths["w"], f"step_{step}/w_step{step}_timestep{i}.csv"), index=False
    )
    
    pd.DataFrame({
        "Channel": list(range(len(x_max_post))), 
        "Max": x_max_post, 
        "Min": x_min_post, 
        "Mean": x_mean_post
    }).to_csv(
        os.path.join(base_paths["x_quant"], f"step_{step}/x_quant_step{step}_timestep{i}.csv"), index=False
    )
    
    pd.DataFrame({
        "Channel": list(range(len(w_max_post))), 
        "Max": w_max_post, 
        "Min": w_min_post, 
        "Mean": w_mean_post
    }).to_csv(
        os.path.join(base_paths["w_quant"], f"step_{step}/w_quant_step{step}_timestep{i}.csv"), index=False
    )
    
    pd.DataFrame({
        "Channel": list(range(len(diff_x_max))), 
        "Max": diff_x_max, 
        "Min": diff_x_min, 
        "Mean": diff_x_mean
    }).to_csv(
        os.path.join(base_paths["x_diff"], f"step_{step}/diff_x_step{step}_timestep{i}.csv"), index=False
    )
    
    pd.DataFrame({
        "Channel": list(range(len(diff_w_max))), 
        "Max": diff_w_max, 
        "Min": diff_w_min, 
        "Mean": diff_w_mean
    }).to_csv(
        os.path.join(base_paths["w_diff"], f"step_{step}/diff_w_step{step}_timestep{i}.csv"), index=False
    )

def save_distribution_plots0(step, i, w, w_quant, x, x_quant, layer_name):
    """
    保存并绘制每个通道的分布图，包括量化前后的权重和激活数据。
    """
    # 预设的文件夹路径
    folder_paths = {
        "weights": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/each_channels_weights",
        "weights_quant": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/each_channels_weights_quant",
        "activations": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/each_channels_activations",
        "activations_quant": "/opt/data/private/GaoJing/deeplearnng/mar/plot/quantlinear/final_layer_adaLN_channel1/each_channels_activations_quant"
    }
    # 对每个文件夹路径创建以 step 和 i 组合的子目录
    for key, base_dir in folder_paths.items():
        step_i_dir = os.path.join(base_dir, f"step_{step}_timestep_{i}")
        if not os.path.exists(step_i_dir):
            os.makedirs(step_i_dir)  # 如果目录不存在则创建

    # 获取输出通道数量
    channel_count = w.shape[0]

    # 保存并绘制每个通道的权重（原始权重）
    for channel_weight_idx in range(channel_count):
        channel_weights = w[channel_weight_idx, :].cpu().detach().numpy()  # 获取每个通道的权重
        save_and_plot(
            channel_weights,
            f"Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
            f"{layer_name}_weights_channel{channel_weight_idx}_step{step}_timestep{i}.png",
            os.path.join(folder_paths["weights"], f"step_{step}_timestep_{i}")
        )

    # 保存并绘制每个通道的权重（量化后的权重）
    for channel_weight_idx in range(channel_count):
        channel_weights = w_quant[channel_weight_idx, :].cpu().detach().numpy()
        save_and_plot(
            channel_weights,
            f"Quantized Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
            f"{layer_name}_weights_quant_channel{channel_weight_idx}_step{step}_timestep{i}.png",
            os.path.join(folder_paths["weights_quant"], f"step_{step}_timestep_{i}")
        )

    # 保存并绘制每个通道的激活值（原始激活）
    for channel_activation_idx in range(channel_count):
        channel_activations = x[channel_activation_idx, :].cpu().detach().numpy()
        save_and_plot(
            channel_activations,
            f"Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
            f"{layer_name}_activations_channel{channel_activation_idx}_step{step}_timestep{i}.png",
            os.path.join(folder_paths["activations"], f"step_{step}_timestep_{i}")
        )

    # 保存并绘制每个通道的激活值（量化后的激活）
    for channel_activation_idx in range(channel_count):
        channel_activations = x_quant[channel_activation_idx, :].cpu().detach().numpy()
        save_and_plot(
            channel_activations,
            f"Quantized Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
            f"{layer_name}_activations_quant_channel{channel_activation_idx}_step{step}_timestep{i}.png",
            os.path.join(folder_paths["activations_quant"], f"step_{step}_timestep_{i}")
        )

# smoothquant里面使用的保存数据的函数
def save_quantization_stats(smoothed_x,smoothed_weight, step, i, layer_name,dim_ac = None,dim = None):
    """
    保存量化前后逐通道的最值、均值以及量化前后逐通道差值的最值。
    """
    # if smoothed_x.ndim == 3:
    #     smoothed_x = smoothed_x.view(-1, smoothed_x.size(2))
    #     smoothed_x_quant = smoothed_x_quant.view(-1, smoothed_x_quant.size(2))
    if smoothed_x.ndim == 4:
        if dim_ac == 0:
            # 合并第二、三、四个维度，保留 batch_size 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(0), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(0), -1)
            # 结果形状：[batch_size, patches * channels * features]

        elif dim_ac == 1:
            # 合并第一、三、四个维度，保留 patches 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(1), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(1), -1)
            # 结果形状：[patches, batch_size * channels * features]

        elif dim_ac == 2:
            # 合并第一、二、四个维度，保留 channels 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(2), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(2), -1)
            # 结果形状：[channels, batch_size * patches * features]

        elif dim_ac == 3:
            # 合并第一、二、三维，保留 features 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(3), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(3), -1)
            # 结果形状：[features, batch_size * patches * channels]

        else:
            raise ValueError("Invalid value for dim_ac. It must be 0, 1, 2, or 3.")


    elif smoothed_x.ndim == 3:
        if dim_ac == 0:
            # 按照第一个维度（batch_size）来调整
            smoothed_x = smoothed_x.view(smoothed_x.size(0), -1)  # 合并第二和第三个维度
            # smoothed_x_quant = smoothed_x_quant.view(smoothed_x_quant.size(0), -1)  # 同样调整量化张量
            # 结果形状：[64, 256 * 768]

        elif dim_ac == 1:
            # 按照第二个维度（patches）来调整
            smoothed_x = smoothed_x.view(smoothed_x.size(1), -1)  # 合并第一个和第三个维度
            # smoothed_x_quant = smoothed_x_quant.view(smoothed_x_quant.size(1), -1)  # 同样调整量化张量
            # 结果形状：[256, 64 * 768]

        elif dim_ac == 2:
            # 按照第三个维度（features）来调整
            smoothed_x = smoothed_x.view(smoothed_x.size(2), -1)  # 合并第一个和第二个维度
            # smoothed_x_quant = smoothed_x_quant.view(smoothed_x_quant.size(2), -1)  # 同样调整量化张量
            # 结果形状：[768, 64 * 256]

        else:
            raise ValueError("Invalid value for dim_ac. It must be 0, 1, or 2.")
    
    elif smoothed_x.ndim == 2:
        smoothed_x = smoothed_x.transpose(0, 1)
    
    # if smoothed_x.ndimension() == 2:
    #     smoothed_x = smoothed_x.transpose(0, 1)
    #     smoothed_x_quant = smoothed_x_quant.transpose(0, 1)

    # dim = 1  # 只针对权重，1是按照输出通道来做（主流），0是按照输入通道来做
    # dim_ac = 1  # 这个是几表示把哪些数据放在一起处理，最基础的方法是为0
    
    # 量化前逐通道最值和均值
    x_max_pre = smoothed_x.max(dim=1).values.cpu().tolist()
    x_min_pre = smoothed_x.min(dim=1).values.cpu().tolist()
    x_mean_pre = smoothed_x.mean(dim=1).cpu().tolist()
    weight_max_pre = smoothed_weight.max(dim=dim).values.cpu().tolist()
    weight_min_pre = smoothed_weight.min(dim=dim).values.cpu().tolist()
    weight_mean_pre = smoothed_weight.mean(dim=dim).cpu().tolist()

    # 量化后逐通道最值和均值
    # x_max_post = smoothed_x_quant.max(dim=1).values.cpu().tolist()
    # x_min_post = smoothed_x_quant.min(dim=1).values.cpu().tolist()
    # x_mean_post = smoothed_x_quant.mean(dim=1).cpu().tolist()
    # weight_max_post = smoothed_weight_quant.max(dim=dim).values.cpu().tolist()
    # weight_min_post = smoothed_weight_quant.min(dim=dim).values.cpu().tolist()
    # weight_mean_post = smoothed_weight_quant.mean(dim=dim).cpu().tolist()

    # # 量化前后差值逐通道最值（绝对差后求最值）
    # diff_x = (smoothed_x - smoothed_x_quant).abs()
    # # diff_weight = (smoothed_weight - smoothed_weight_quant).abs()
    # diff_x_max = diff_x.max(dim=1).values.cpu().tolist()
    # diff_x_min = diff_x.min(dim=1).values.cpu().tolist()
    # diff_x_mean = diff_x.mean(dim=1).cpu().tolist()
    # # diff_weight_max = diff_weight.max(dim=dim).values.cpu().tolist()
    # # diff_weight_min = diff_weight.min(dim=dim).values.cpu().tolist()
    # # diff_weight_mean = diff_weight.mean(dim=dim).cpu().tolist()

    # 定义基础文件夹路径

    base_paths = {
        "smoothed_x": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/layer/{layer_name}/all_channels_activations_a{dim_ac}/smoothed_x",
        "smoothed_weight": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/layer/{layer_name}/all_channels_weights_w{dim}/smoothed_w",
        # "smoothed_x_quant": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/{layer_name}/all_channels_activations_a{dim_ac}/smoothed_x_quant",
        # "smoothed_weight_quant": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/{layer_name}/all_channels_weights_w{dim}/smoothed_w_quant",
        # "smoothed_x_diff": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/{layer_name}/all_channels_activations_a{dim_ac}/smoothed_x_diff",
        # "smoothed_weight_diff": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/{layer_name}/all_channels_weights_w{dim}/smoothed_w_diff"
    }

    # 为每个基础路径创建以 step 为命名的子目录
    for key, base_dir in base_paths.items():
        step_dir = os.path.join(base_dir, f"step_{step}")
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)  # 如果目录不存在，则创建

    # 保存数据到对应的目录
    pd.DataFrame({
        "Channel": list(range(len(x_max_pre))), 
        "Max": x_max_pre, 
        "Min": x_min_pre, 
        "Mean": x_mean_pre
    }).to_csv(
        os.path.join(base_paths["smoothed_x"], f"step_{step}/smoothed_x_step{step}_timestep{i}.csv"), index=False
    )
    
    pd.DataFrame({
        "Channel": list(range(len(weight_max_pre))), 
        "Max": weight_max_pre, 
        "Min": weight_min_pre, 
        "Mean": weight_mean_pre
    }).to_csv(
        os.path.join(base_paths["smoothed_weight"], f"step_{step}/smoothed_weight_step{step}_timestep{i}.csv"), index=False
    )
    
    # pd.DataFrame({
    #     "Channel": list(range(len(x_max_post))), 
    #     "Max": x_max_post, 
    #     "Min": x_min_post, 
    #     "Mean": x_mean_post
    # }).to_csv(
    #     os.path.join(base_paths["smoothed_x_quant"], f"step_{step}/smoothed_x_quant_step{step}_timestep{i}.csv"), index=False
    # )
    
    # pd.DataFrame({
    #     "Channel": list(range(len(weight_max_post))), 
    #     "Max": weight_max_post, 
    #     "Min": weight_min_post, 
    #     "Mean": weight_mean_post
    # }).to_csv(
    #     os.path.join(base_paths["smoothed_weight_quant"], f"step_{step}/smoothed_weight_quant_step{step}_timestep{i}.csv"), index=False
    # )
    
    # pd.DataFrame({
    #     "Channel": list(range(len(diff_x_max))), 
    #     "Max": diff_x_max, 
    #     "Min": diff_x_min, 
    #     "Mean": diff_x_mean
    # }).to_csv(
    #     os.path.join(base_paths["smoothed_x_diff"], f"step_{step}/diff_x_step{step}_timestep{i}.csv"), index=False
    # )
    
    # pd.DataFrame({
    #     "Channel": list(range(len(diff_weight_max))), 
    #     "Max": diff_weight_max, 
    #     "Min": diff_weight_min, 
    #     "Mean": diff_weight_mean
    # }).to_csv(
    #     os.path.join(base_paths["smoothed_weight_diff"], f"step_{step}/diff_weight_step{step}_timestep{i}.csv"), index=False
    # )

def save_plots(step, i,smoothed_x,smoothed_weight, layer_name, dim_ac = None, dim = None):
    
    # if smoothed_x.ndim == 3:
    #     smoothed_x = smoothed_x.view(-1, smoothed_x.size(2))
    #     smoothed_x_quant = smoothed_x_quant.view(-1, smoothed_x_quant.size(2))
    if smoothed_x.ndim == 4:
        if dim_ac == 0:
            # 合并第二、三、四个维度，保留 batch_size 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(0), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(0), -1)
            # 结果形状：[batch_size, patches * channels * features]

        elif dim_ac == 1:
            # 合并第一、三、四个维度，保留 patches 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(1), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(1), -1)
            # 结果形状：[patches, batch_size * channels * features]

        elif dim_ac == 2:
            # 合并第一、二、四个维度，保留 channels 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(2), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(2), -1)
            # 结果形状：[channels, batch_size * patches * features]

        elif dim_ac == 3:
            # 合并第一、二、三维，保留 features 维度
            smoothed_x = smoothed_x.reshape(smoothed_x.size(3), -1)
            # smoothed_x_quant = smoothed_x_quant.reshape(smoothed_x_quant.size(3), -1)
            # 结果形状：[features, batch_size * patches * channels]

        else:
            raise ValueError("Invalid value for dim_ac. It must be 0, 1, 2, or 3.")

    if smoothed_x.ndim == 3:
        if dim_ac == 0:
            # 按照第一个维度（batch_size）来调整
            smoothed_x = smoothed_x.view(smoothed_x.size(0), -1)  # 合并第二和第三个维度
            # smoothed_x_quant = smoothed_x_quant.view(smoothed_x_quant.size(0), -1)  # 同样调整量化张量
            # 结果形状：[64, 256 * 768]

        elif dim_ac == 1:
            # 按照第二个维度（patches）来调整
            smoothed_x = smoothed_x.view(smoothed_x.size(1), -1)  # 合并第一个和第三个维度
            # smoothed_x_quant = smoothed_x_quant.view(smoothed_x_quant.size(1), -1)  # 同样调整量化张量
            # 结果形状：[256, 64 * 768]

        elif dim_ac == 2:
            # 按照第三个维度（features）来调整
            smoothed_x = smoothed_x.view(smoothed_x.size(2), -1)  # 合并第一个和第二个维度
            # smoothed_x_quant = smoothed_x_quant.view(smoothed_x_quant.size(2), -1)  # 同样调整量化张量
            # 结果形状：[768, 64 * 256]

        else:
            raise ValueError("Invalid value for dim_ac. It must be 0, 1, or 2.")
    
    if smoothed_x.ndimension() == 2:
        smoothed_x = smoothed_x.transpose(0, 1)
        # smoothed_x_quant = smoothed_x_quant.transpose(0, 1)

    # 预设的6个独立文件夹路径
    folder_paths = {
        "weights": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/layer/{layer_name}/each_channels_weights_w{dim}",
        # "weights_quant": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/{layer_name}/each_channels_weights_quant_w{dim}",
        "activations": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/layer/{layer_name}/each_channels_activations_a{dim_ac}",
        # "activations_quant": f"/opt/data/private/GaoJing/deeplearnng/mar/plot/{layer_name}/each_channels_activations_quant_a{dim_ac}"
    }

    # 对每个文件夹路径创建以 step 和 i 组合的子目录
    for key, base_dir in folder_paths.items():
        # 创建以 step 和 i 为名的子目录
        step_i_dir = os.path.join(base_dir, f"step_{step}_timestep_{i}")
        if not os.path.exists(step_i_dir):
            os.makedirs(step_i_dir)  # 如果目录不存在则创建

    # 获取输出通道数量
    channel_count = smoothed_weight.shape[0]

    # 打印每个通道的权重（原始权重）
    for channel_weight_idx in range(channel_count):
        channel_weights = smoothed_weight[channel_weight_idx, :].cpu().detach().numpy()  # 获取每个通道的权重
        save_and_plot(
            channel_weights,
            f"Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
            f"{layer_name}_weights_channel{channel_weight_idx}_step{step}_timestep{i}.png",
            os.path.join(folder_paths["weights"], f"step_{step}_timestep_{i}")  # 将文件保存在对应的目录
        )

    # # 打印每个通道的权重（量化后的权重）
    # for channel_weight_idx in range(channel_count):
    #     channel_weights = smoothed_weight_quant[channel_weight_idx, :].cpu().detach().numpy()  # 获取每个通道的权重
    #     save_and_plot(
    #         channel_weights,
    #         f"Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
    #         f"{layer_name}_weights_channel{channel_weight_idx}_step{step}_timestep{i}.png",
    #         os.path.join(folder_paths["weights_quant"], f"step_{step}_timestep_{i}")  # 将文件保存在对应的目录
    #     )

    # 获取激活的通道数量
    activation_count = smoothed_x.shape[0]

    # 打印每个通道的激活（原始激活）
    for channel_activation_idx in range(activation_count):
        channel_activations = smoothed_x[channel_activation_idx,:].cpu().detach().numpy()  # 获取每个通道的激活
        save_and_plot(
            channel_activations,
            f"Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
            f"{layer_name}_activations_channel{channel_activation_idx}_step{step}_timestep{i}.png",
            os.path.join(folder_paths["activations"], f"step_{step}_timestep_{i}")  # 将文件保存在对应的目录
        )

    # # 打印每个通道的激活（量化后的激活）
    # for channel_activation_idx in range(activation_count):
    #     channel_activations = smoothed_x_quant[channel_activation_idx,:].cpu().detach().numpy()  # 获取每个通道的激活
    #     save_and_plot(
    #         channel_activations,
    #         f"Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
    #         f"{layer_name}_activations_channel{channel_activation_idx}_step{step}_timestep{i}.png",
    #         os.path.join(folder_paths["activations_quant"], f"step_{step}_timestep_{i}")  # 将文件保存在对应的目录
    #     )

def save_layer_statistics_to_csv(x, layer_name, i, step, num_bsz, save_folder):
    """
    将每个step的x的最大值、最小值、均值保存到以layer_name命名的CSV文件中。
    
    Args:
    - x (Tensor): 输入数据，可以是任意形状的Tensor。
    - layer_name (str): 层的名称，将用于命名CSV文件。
    - i (int): 传入的参数i，保存到CSV中。
    - step (int): 当前的step，用于记录到CSV文件中。
    - num_bsz (int): 当前的批次大小，用于记录到CSV文件中。
    - save_folder (str): 保存CSV文件的文件夹路径。
    """
    # 创建文件夹（如果不存在）
    os.makedirs(save_folder, exist_ok=True)
    
    # CSV 文件路径
    csv_file_path = os.path.join(save_folder, f"{layer_name}.csv")

    # 如果文件不存在，先写入表头
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ "num_bsz","step", "i", "max_value", "min_value", "mean_value"])

    # 计算x的最大值、最小值和均值
    max_value = torch.max(x).item()
    min_value = torch.min(x).item()
    mean_value = torch.mean(x).item()

    # 将结果写入CSV文件
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([ f"num_bsz_{num_bsz}", f"step{step}", f"{i}",
                         f"max_value: {max_value}", f"min_value: {min_value}", f"mean_value: {mean_value}"])

    # print(f"Step {step}, i = {i}, num_bsz = {num_bsz}, max_value = {max_value}, min_value = {min_value}, mean_value = {mean_value}")

import numpy as np

# 归一化函数：将数据归一化到[0, 1]之间
def normalize(data, epsilon=1e-8):
    min_val = data.min()
    max_val = data.max()
    range_val = max_val - min_val
    
    # 如果range为零，返回原数据，或者可以设置一个小的常数值避免除零
    if range_val <= epsilon:
        return np.full_like(data, 0)
    
    return (data - min_val) / range_val

# 计算方差
def compute_variance(data):
    return np.var(data)

def compute_uniformity(data):
    center = 0.5  # 中心值
    differences = data - center
    squared_diff = differences ** 2
    uniformity = np.mean(squared_diff)
    return uniformity

# 用二分法寻找最优阈值
def find_optimal_threshold(data, by_channel = False,is_weight = False,limit = False):
    if by_channel:
        data_abs = data.abs()
        # data_abs = data
        data_array = data_abs.amax(dim=tuple(range(data.ndim - 1)), keepdim=False)
        data_sorted = np.sort(data_array.cpu().detach().numpy())
        n = len(data_sorted)
    elif is_weight:
        data_abs = data.abs()
        data_array = data_abs.reshape(-1).cpu().detach().numpy()
        data_sorted = np.sort(data_array)  # 对数据进行排序
        n = len(data_sorted)
    else:
        data_abs = data
        data_array = data_abs.reshape(-1).cpu().detach().numpy()
        data_sorted = np.sort(data_array)  # 对数据进行排序
        n = len(data_sorted)
    
    best_threshold = data_sorted[n-1]
    # best_variance = float('inf')  # 初始设定为一个非常大的方差

    first_iteration = True  # 标志变量，用来判断是否是第一次迭代
    min_diff = float('inf')  # 初始化最小差值为一个非常大的值
    iteration_count = 0  # 记录迭代次数

    # 当前阈值的前后边界
    left = 0
    right = n - 1

    while left < right:
        # 根据是否第一次迭代来设置阈值
        if first_iteration:
            threshold = data_sorted[int(0.99 * n)]  # 初始化阈值为数据的99.9%位置
            # first_iteration = False
        else:
            threshold = data_sorted[(left + right) // 2]  # 计算新的阈值
        
        # 将数据分为两组
        group1 = data_sorted[data_sorted <= threshold]
        group2 = data_sorted[data_sorted > threshold]

        # 检查是否有空组，如果有空组则跳过当前阈值
        if len(group1) == 0 or len(group2) == 0:
            # 如果分组为空，跳过当前阈值
            iteration_count += 1  # 迭代次数加1
            if iteration_count >= 20:
                break
            continue  # 跳过当前循环，继续新的迭代
        
        # 分别对两组进行归一化
        group1_normalized = normalize(group1)
        group2_normalized = normalize(group2)
        
         # 计算两组的均匀性度量（与0.5的差值平方和的均值）
        variance1 = compute_uniformity(group1_normalized) 
        variance2 = compute_uniformity(group2_normalized)
        # current_diff = abs(variance1 - variance2)
        current_diff = variance1 + variance2 + abs(variance1 - variance2)

        if limit:
            variance1 = compute_uniformity(group1_normalized) * 4  # 越小越好，（0，1）
            variance2 = len(group2) / (n*0.002) # 衡量异常值多少 # 越小越好 （0，1）
            # 计算当前方差差值
            # current_diff = abs(variance1 - variance2* 1.5)
            # current_diff = abs(variance1 * 0.9 - variance2 * 0.1)
            current_diff = variance1 + variance2
        
        
        # 如果当前的差值小于之前的最小差值，则更新最小差值
        if current_diff < min_diff:
            min_diff = current_diff
            best_threshold = threshold
        else:
            if iteration_count >= 20:
                break
        
        # 根据方差更大的组来进行二分
        if variance1 > variance2:
            if first_iteration:
                right = int(0.99 * n)
                first_iteration = False
            else:
                # right = (left + right) // 2 - 1
                # right = (left + right) // 2
                if (left+right)%2==0:
                    right = (left + right) // 2
                else:
                    right = (left + right) // 2 - 1
        else:
            if first_iteration:
                left = int(0.99 * n)
                first_iteration = False
            else:
                # left = (left + right) // 2 + 1
                # left = (left + right) // 2
                if (left+right)%2==0:
                    left = (left + right) // 2
                else:
                    left = (left + right) // 2 + 1
        
        iteration_count += 1  # 迭代次数加1
    
    if data_sorted[-1] < 1:
        best_threshold = 1e10
    # else:
    #     # 2. 计算整个数据的中位数
    #     max = data_sorted[-1]
    #     # eps = 1e-6  # 防止除0
    #     # 3. 计算比值：二分法得到的best_threshold与中位数的比值
    #     ratio = max / best_threshold
    #     # 4. 如果比值小于等于3，则说明数据分布较均匀，不适合分组量化，直接使用最大值作为阈值
    #     if ratio <= 3:
    #         best_threshold = 1e10
    group1 = data_sorted[data_sorted <= best_threshold]
    group2 = data_sorted[data_sorted > best_threshold]
    length = len(group2)
    greater_than_threshold_percentage = len(group2) / n * 100
    # logger3.info(f"Threshold: {best_threshold}, Percentage of values > threshold: {greater_than_threshold_percentage:.5f}% , nums：{len(group2)}")

    # 如果 group2 中的数据占比大于 0.1%，且不是 by_channel 和 is_weight，则再次调用该函数
    if (greater_than_threshold_percentage > 0.1 and length > 50000) and not by_channel:
        best_threshold = data_sorted[int(n*0.999)]
        return best_threshold

    if by_channel:
        channel_indices = [i for i in range(n) if data_array[i] > best_threshold]
        return best_threshold,channel_indices
    else:
        return best_threshold

def compute_threshold_statistics(x, threshold = 0, layer_name = None):
        """
        计算输入激活中的正常值和异常值的比例。
        参数:
            x: 输入激活张量，形状为 (batch_size, channels, height, width)
            step: 当前的步骤，用于选择相应的阈值
        返回:
            None: 结果会直接通过 logger3 输出
        """
        
        # 将输入张量展平为二维 (N, C)，方便计算
        x_flat = x.reshape(-1)
        

        # 计算每两个相邻位置的阈值
        pairs = []
        for i in range(0, len(x_flat)-1, 2):  # 紧邻两个位置
            pair = (x_flat[i], x_flat[i+1])
            pair_threshold = threshold
            pairs.append(pair)

        # 将值与阈值进行比较，统计结果
        normal_normal = 0
        normal_abnormal = 0
        abnormal_normal = 0
        abnormal_abnormal = 0
        
        # 遍历每一组
        for pair in pairs:
            # 判断每个元素是否为正常值 (小于阈值) 或异常值 (大于阈值)
            first_value = pair[0]
            second_value = pair[1]
            
            # 判断每个位置的值
            first_is_normal = first_value < threshold
            second_is_normal = second_value < threshold
            
            # 根据正常值和异常值的组合，进行统计
            if first_is_normal and second_is_normal:
                normal_normal += 1
            elif first_is_normal and not second_is_normal:
                normal_abnormal += 1
            elif not first_is_normal and second_is_normal:
                abnormal_normal += 1
            else:
                abnormal_abnormal += 1

        # 计算比例
        total_pairs = len(pairs)
        normal_normal_ratio = normal_normal / total_pairs
        normal_abnormal_ratio = normal_abnormal / total_pairs
        abnormal_normal_ratio = abnormal_normal / total_pairs
        abnormal_abnormal_ratio = abnormal_abnormal / total_pairs

        # 打印结果到 logger3
        logger3.info(f"layer {layer_name}:")
        logger3.info(f"Normal-Normal Ratio: {normal_normal_ratio:.4f}")
        logger3.info(f"Normal-Abnormal Ratio: {normal_abnormal_ratio:.4f}")
        logger3.info(f"Abnormal-Normal Ratio: {abnormal_normal_ratio:.4f}")
        logger3.info(f"Abnormal-Abnormal Ratio: {abnormal_abnormal_ratio:.4f}")


def genetic_kmeans(data, k, pop_size=50, ngen=20, cxpb=0.5, mutpb=0.2):
    """
    使用遗传算法优化的K-Means聚类
    
    参数:
        data: 要聚类的数据，形状为(n_samples, n_features)
        k: 聚类数量
        pop_size: 遗传算法种群大小
        ngen: 遗传算法迭代次数
        cxpb: 交叉概率
        mutpb: 变异概率
    
    返回:
        best_centers: 最优的聚类中心
        best_labels: 最优的聚类标签
    """
    n_samples, n_features = data.shape
    
    # 创建遗传算法所需的类型
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # 初始化工具箱
    toolbox = base.Toolbox()
    
    # 定义个体生成函数 - 每个个体是k个聚类中心的集合
    def create_individual():
        # 从数据中随机选择k个点作为初始中心
        indices = np.random.choice(range(n_samples), size=k, replace=False)
        centers = data[indices].flatten().tolist()
        return centers
    
    # 注册遗传算法操作
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 评估函数 - 计算K-Means的目标函数(簇内平方和)
    def evaluate(individual):
        # 将个体转换为聚类中心矩阵
        centers = np.array(individual).reshape(k, n_features)
        
        # 计算每个点到最近中心的距离平方和
        distances = np.array([np.sum((data - center)**2, axis=1) for center in centers])
        min_distances = np.min(distances, axis=0)
        total_distance = np.sum(min_distances)
        
        return (total_distance,)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)  # 高斯变异
    toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择
    
    # 创建初始种群
    population = toolbox.population(n=pop_size)
    
    # 运行遗传算法
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    
    # 获取最优个体
    best_individual = tools.selBest(population, k=1)[0]
    best_centers = np.array(best_individual).reshape(k, n_features)
    
    # 分配标签
    distances = np.array([np.sum((data - center)**2, axis=1) for center in best_centers])
    best_labels = np.argmin(distances, axis=0)
    
    return best_centers, best_labels

def custom_quantization_error(labels, tensor, data):
    """
    计算基于聚类的量化误差
    
    参数:
        labels: 聚类标签数组，形状为(n_channels,)
        tensor: 输入张量，形状为(..., n_channels)
        data: 原始数据（用于计算MSE）
    
    返回:
        error: 量化误差（MSE）
    """
    # 确保输入是NumPy数组
    labels = np.asarray(labels)
    tensor = tensor.cpu().numpy()
    
    # 获取通道维度
    n_channels = tensor.shape[-1]
    
    # 1. 确定主簇（最大值最小的簇）
    unique_labels = np.unique(labels)
    cluster_max = {}
    
    # 计算每个簇的最大值
    for label in unique_labels:
        channel_indices = np.where(labels == label)[0]
        cluster_data = tensor[..., channel_indices]
        cluster_max[label] = np.max(cluster_data)
    
    # 找到最大值最小的簇作为主簇
    main_cluster = min(cluster_max.keys(), key=lambda x: cluster_max[x])
    
    # 2. 分离正常通道和异常通道
    normal_channels = np.where(labels == main_cluster)[0]
    abnormal_channels = np.where(labels != main_cluster)[0]
    
    # 计算正常通道的最大值
    normal_max = np.max(tensor[..., normal_channels])
    
    # 3. 缩放异常通道
    scaled_tensor = tensor.copy()
    scale_factors = {}
    
    for ch in abnormal_channels:
        channel_max = np.max(tensor[..., ch])
        scale_factor = normal_max / channel_max
        scale_factors[ch] = scale_factor
        scaled_tensor[..., ch] *= scale_factor
    
    # 4. 8位量化/反量化过程
    def quantize_8bit(tensor):
        # 将数据缩放到0-255范围
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        scale = 255.0 / (max_val - min_val) if max_val != min_val else 1.0
        
        # 量化
        quantized = np.round((tensor - min_val) * scale).astype(np.uint8)
        
        # 反量化
        dequantized = quantized.astype(np.float32) / scale + min_val
        return dequantized
    
    quantized_tensor = quantize_8bit(scaled_tensor)
    
    # 5. 恢复原始范围
    restored_tensor = quantized_tensor.copy()
    for ch in abnormal_channels:
        restored_tensor[..., ch] /= scale_factors[ch]
    
    # 6. 计算MSE误差
    error = np.mean((restored_tensor - tensor) ** 2)

    # # 仅对异常通道计算MSE
    # error = np.mean((restored_tensor[..., abnormal_channels] - tensor[..., abnormal_channels]) ​** 2)
    
    return error

def find_optimal_k(data, tensor,data_norm, k_range=range(2, 11)):
    """
    寻找最优的K值
    
    参数:
        data: 要聚类的数据
        k_range: 要尝试的K值范围
    
    返回:
        best_k: 最优的K值
        results: 包含所有K值结果的字典
    """
    results = {}
    
    for k in tqdm(k_range, desc="Testing K values"):
        # 使用遗传算法优化的K-Means
        centers, labels = genetic_kmeans(data_norm, k)
        
        # 计算自定义量化误差
        quant_error = custom_quantization_error(labels, tensor, data)
        
        # 计算轮廓系数(可选)
        silhouette = silhouette_score(data, labels) if k > 1 else 0
        
        # 存储结果
        results[k] = {
            'centers': centers,
            'labels': labels,
            'quant_error': quant_error,
            'silhouette': silhouette
        }
    
    # 找到量化误差最小的K值
    best_k = min(results.keys(), key=lambda x: results[x]['quant_error'])
    
    return best_k, results

def visualize_results(data, layer_name, results, base_dir=os.path.join(outpath, "visualizations")):
    """
    改进的可视化结果保存函数
    
    参数:
        results: find_optimal_k返回的结果字典
        base_dir: 基础保存路径
    """
    # 创建以当前时间为名的子文件夹
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, current_time + layer_name)
    os.makedirs(save_dir, exist_ok=True)
    
    ks = sorted(results.keys())
    quant_errors = [results[k]['quant_error'] for k in ks]
    silhouettes = [results[k]['silhouette'] for k in ks]
    
    # 1. 创建主评估图
    plt.figure(figsize=(10, 6))
    plt.plot(ks, quant_errors, 'o-', color='tab:red', label='Quantization Error')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Quantization Error', color='tab:red')
    plt.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = plt.twinx()
    ax2.plot(ks, silhouettes, 's-', color='tab:blue', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Cluster Evaluation Metrics')
    plt.savefig(os.path.join(save_dir, "cluster_evaluation_metrics.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 为每个K值创建单独的聚类可视化
    for k in ks:
        centers = results[k]['centers']
        labels = results[k]['labels']
        
        plt.figure(figsize=(12, 6))
        
        # 原始数据分布
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(data)), data, c=labels, cmap='viridis', s=50, alpha=0.6)
        plt.scatter([0]*len(centers), centers, c='red', marker='X', s=200, label='Centers')
        plt.title(f'Original Data (K={k})')
        plt.xlabel('Channel Index')
        plt.ylabel('Max Activation Value')
        plt.legend()
        
        # 添加文本信息
        plt.subplot(1, 2, 2)
        plt.axis('off')
        info_text = (
            f"K = {k}\n"
            f"Quant Error = {results[k]['quant_error']:.8f}\n"
            f"Silhouette = {results[k]['silhouette']:.4f}\n"
            f"Centers:\n{np.array2string(centers.flatten(), precision=4)}"
        )
        plt.text(0.1, 0.5, info_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5))
        
        plt.suptitle(f'Cluster Analysis (K={k})')
        plt.savefig(os.path.join(save_dir, f"cluster_analysis_k_{k}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logger4.info(f"All visualizations saved to: {save_dir}")

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import os

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

class GeneticKMeans:
    def __init__(self, visualize=False):
        """
        基于遗传算法优化的K-Means聚类缩放线性层
        
        参数:
            visualize: 是否生成可视化结果 (默认False)
        """
        self.abnormal_channels = []
        self.visualize = visualize
        self.results = {}  # 存储聚类评估结果
        self.is_quantization_acceptable = True  # 默认不可接受
        self.acceptable_error_threshold = 10   # 可配置的误差阈值
        # self.k_range = k_range
        # self.w = w
        
    def optimize(self, activation_tensor, layer_name="",args = None):
        """
        主优化方法：检测异常通道并计算最优缩放方案
        
        参数:
            activation_tensor: 输入张量 (..., num_channels)
            layer_name: 当前层名称 (用于可视化保存路径)
            
        返回:
            abnormal_channels: 异常通道索引列表
        """
        # 1. 提取通道最大值
        # if isinstance(activation_tensor, torch.Tensor):
        #     activation_tensor = activation_tensor.detach().cpu()
        # max_values = np.max(np.abs(activation_tensor), 
        #                    axis=tuple(range(activation_tensor.ndim-1)))
        # max_values_np = max_values.reshape(-1, 1)

        tensor = activation_tensor.clone()
        reshaped = tensor.view(-1, tensor.shape[-1])  # [N, 1536]
        max_values, _ = reshaped.max(dim=0)  # [1536]
        max_values_np = max_values.cpu().numpy().reshape(-1, 1)
        
        logger3.info(f"Processing layer {layer_name} with {max_values_np.shape[0]} channels")
        
        # 2. 寻找最优聚类方案
        best_k, best_labels, best_abnormal = self._find_optimal_clustering(
            max_values_np, reshaped,k_range=range(2, 20),w=0.001,args=args
        )
        
        # 3. 存储结果
        self.abnormal_channels = best_abnormal
        logger4.info(f"Layer {layer_name} - Optimal k={best_k}, "
                    f"found {len(best_abnormal)} abnormal channels")
        
        # 4. 可视化（如果启用）
        if self.visualize:
            self.visualize_results(max_values_np, layer_name)
            
        return self.abnormal_channels
    
    def _find_optimal_clustering(self, max_values, activation_tensor, k_range=range(2, 20),w=0.001,args=None):
        """寻找最优聚类方案"""

        # if args is not None:
        #     logger4.info(f"Genetic K-Means params: pop_size={args.pop_size}, ngen={args.ngen}, "
        #                 f"cxpb={args.cxpb}, mutpb={args.mutpb}")
                     
        for k in tqdm(k_range, desc="Testing clustering options"):
            tensor = activation_tensor.clone()
            # 遗传K-Means聚类
            centers, labels = self._genetic_kmeans(max_values, k)
            
            # 计算量化误差和异常通道
            quant_error, abnormal,final_labels = self._compute_quantization_error(
                labels, tensor, max_values,w=w
            )
            abnormal_channels = np.array(abnormal)
            
            # 存储结果
            self.results[k] = {
                'centers': centers,
                'labels': labels,
                'quant_error': quant_error,
                'silhouette': silhouette_score(max_values, labels) if k > 1 else 0,
                'abnormal_channels': abnormal_channels.tolist(),
                'final_labels':final_labels
            }
            
            logger3.info(f"k={k} - Quant error: {quant_error:.6f}, "
                         f"silhouette: {self.results[k]['silhouette']:.4f}")
        
        # 选择量化误差最小的方案
        best_k = min(self.results.keys(), key=lambda x: self.results[x]['quant_error'])

        # 检查最佳k对应的量化误差是否超过阈值
        if hasattr(self, 'acceptable_error_threshold'):
            self.is_quantization_acceptable = (
                self.results[best_k]['quant_error'] <= self.acceptable_error_threshold
            )
        else:
            # 如果没有设置阈值，默认认为可接受
            self.is_quantization_acceptable = True

        return best_k, self.results[best_k]['labels'], self.results[best_k]['abnormal_channels']
    
    def _genetic_kmeans(self, data, k, pop_size=50, ngen=20, cxpb=0.5, mutpb=0.2,random_state = 42):
        """
        遗传算法优化的K-Means
        
        参数:
            data: 要聚类的数据
            k: 聚类数量
            pop_size: 种群大小
            ngen: 迭代次数
            cxpb: 交叉概率
            mutpb: 变异概率
        
        返回:
            centers: 聚类中心
            labels: 聚类标签
        """
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        n_samples, n_features = data.shape
        
        # 创建遗传算法类型
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # 初始化工具箱
        toolbox = base.Toolbox()
        # 定义个体生成函数 - 每个个体是k个聚类中心的集合
        def create_individual():
            # 从数据中随机选择k个点作为初始中心
            indices = np.random.choice(range(n_samples), size=k, replace=False)
            centers = data[indices].flatten().tolist()
            return centers
        
        # 注册遗传算法操作
        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # # 定义个体生成函数 - 每个个体是k个聚类中心的集合
        # def create_individual():
        #     # 从数据中随机选择k个点作为初始中心
        #     indices = np.random.choice(range(n_samples), size=k, replace=False)
        #     centers = data[indices].flatten().tolist()
        #     return centers
    
        # # 注册遗传算法操作
        # toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._kmeans_fitness, data=data)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # 运行遗传算法
        population = toolbox.population(n=pop_size)
        algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
        
        # 获取最佳结果
        best_individual = tools.selBest(population, k=1)[0]
        best_centers = np.array(best_individual).reshape(k, n_features)
        
        # 分配标签
        distances = np.array([np.sum((data - center)**2, axis=1) for center in best_centers])
        best_labels = np.argmin(distances, axis=0)
        
        return best_centers, best_labels
    
    def _kmeans_fitness(self, individual, data):
        """K-Means的适应度函数（簇内平方和）"""
        k = len(individual) // data.shape[1]
        centers = np.array(individual).reshape(k, data.shape[1])
        
        distances = np.array([np.sum((data - center)**2, axis=1) for center in centers])
        min_distances = np.min(distances, axis=0)
        return (np.sum(min_distances),)
    
    def _compute_quantization_error(self, labels, activation_tensor, max_values,w=0.001):
        """
        计算量化误差
        
        参数:
            labels: 聚类标签
            activation_tensor: 原始激活张量
            max_values: 各通道最大值
        
        返回:
            mse_error: 量化后的MSE误差
        """
        labels = np.asarray(labels)
        tensor_np = activation_tensor.cpu().numpy()
        unique_labels = np.unique(labels)
        k = len(unique_labels)
        total_channels = activation_tensor.shape[-1]
        
        if k == 2:
            # 直接处理k=2的情况
            cluster_max = {label: np.max(max_values[labels == label]) for label in unique_labels}
            final_labels = labels
            main_cluster = min(cluster_max.keys(), key=lambda x: cluster_max[x])
            normal_channels = np.where(labels == main_cluster)[0]
            abnormal_channels = np.where(labels != main_cluster)[0]
        else:
            # 提取每簇最大值
            first_labels = labels
            # cluster_maxima = [np.max(max_values[first_labels == i]) for i in range(k)]
            cluster_maxima = []
            for i in range(k):
                cluster_data = max_values[first_labels == i]
                if len(cluster_data) == 0:
                    # 处理空簇：赋予全局最小值或特定值
                    cluster_maxima.append(np.min(max_values))  # 或 np.nan / 0.0
                else:
                    cluster_maxima.append(np.max(cluster_data))
            cluster_maxima = np.array(cluster_maxima).reshape(-1, 1)
      
            # 强制k=2的二次聚类
            kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10).fit(cluster_maxima)
            second_labels = kmeans.labels_
            final_centers = kmeans.cluster_centers_.flatten()  # 二次聚类中心
                        
            # 确定主簇（最大值较小的簇）
            if np.mean(cluster_maxima[second_labels == 0]) < np.mean(cluster_maxima[second_labels == 1]):
                main_cluster_group = 0
            else:
                main_cluster_group = 1
                        
            # 映射回原始标签
            main_clusters = [i for i in range(k) if second_labels[i] == main_cluster_group]
            final_labels = np.zeros_like(first_labels)
                        
            for new_label, orig_label in enumerate(range(k)):
                mask = (first_labels == orig_label)
                if orig_label in main_clusters:
                    final_labels[mask] = 0  # 主簇
                else:
                    final_labels[mask] = 1  # 异常簇
                        
            main_cluster = 0  # 主簇标签已重新映射为0
            abnormal_channels = np.where(final_labels != main_cluster)[0].tolist()
            normal_channels = np.where(final_labels == main_cluster)[0].tolist()


        # 计算正常通道的最大值
        normal_max = np.max(tensor_np[..., normal_channels])
        
        # 缩放异常通道
        scaled_tensor = tensor_np.copy()
        scale_factors = {}
        
        for ch in abnormal_channels:
            channel_max = np.max(tensor_np[..., ch])
            scale_factor = normal_max / channel_max
            scale_factors[ch] = scale_factor
            scaled_tensor[..., ch] *= scale_factor
        
        # 8位量化/反量化
        def quantize_8bit(tensor):
            min_val = np.min(tensor)
            max_val = np.max(tensor)
            scale = 255.0 / (max_val - min_val) if max_val != min_val else 1.0
            quantized = np.round((tensor - min_val) * scale).astype(np.uint8)
            return quantized.astype(np.float32) / scale + min_val
            
        # 获取异常通道数据（确保输出是NumPy数组）
        abnormal_tensor = np.asarray(scaled_tensor[..., abnormal_channels])
        symbol_matrix = np.sign(abnormal_tensor)

        # 执行量化（假设quantize_8bit返回NumPy数组）
        quantized_tensor = np.asarray(quantize_8bit(scaled_tensor))

        # 处理异常通道
        abnormal_quantized = np.asarray(quantized_tensor[..., abnormal_channels])
        current_symbol = np.sign(abnormal_quantized)
        symbol_mask = (current_symbol != symbol_matrix)

        # 调整符号
        adjusted_quantized = abnormal_quantized.copy()  # NumPy用copy()代替clone()
        adjusted_quantized[symbol_mask] *= -1

        # 回写结果
        quantized_tensor[..., abnormal_channels] = adjusted_quantized

        # 统计计算
        adjusted_count = np.sum(symbol_mask)
        total_elements = abnormal_quantized.size  # NumPy的size属性等价于numel()

        # print(f"符号调整: 调整了{adjusted_count}/{total_elements}个元素")
                            
        # logger3.info(f"符号调整: 调整了{adjusted_count}/{total_elements}个元素 (设备: {abnormal_quantized})")
        
        # 恢复异常通道原始范围
        for ch in abnormal_channels:
            quantized_tensor[..., ch] /= scale_factors[ch]

        logger4.info(f"异常通道数量: {len(abnormal_channels)}, 权重w: {w}")
        error = np.sum((quantized_tensor[..., abnormal_channels] - tensor_np[..., abnormal_channels]) ** 2) + w*(np.sum((quantized_tensor[..., normal_channels] - tensor_np[..., normal_channels]) ** 2))


        return error,abnormal_channels,final_labels

    # def plot_cluster_with_broken_y(self, sorted_data, sorted_labels, title, annotation=None, save_path=None,color = False):
    #     """
    #     三段式断轴绘图：0-7 (50%)，7-50 (25%)，50+ (25%)，使用掩码避免重叠，并用渐变唯一颜色。
    #     """
    #     fig = plt.figure(figsize=(7, 6))
    #     gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.05)

    #     # 掩码划分
    #     lower_mask = sorted_data < 7
    #     mid_mask = (sorted_data >= 6.5) & (sorted_data < 50)
    #     upper_mask = sorted_data >= 50

    #     # 渐变颜色分配（每个label一个颜色，按大小排序）
    #     unique_labels = np.unique(sorted_labels)
    #     num_labels = len(unique_labels)
    #     # 柔和分组色盘（Dark2），颜色分明且不刺眼
    #     # colors = plt.cm.get_cmap('Dark2')(np.linspace(0, 1, num_labels))
    #     if color and num_labels == 2:
    #         # 将 RGB 转为 Matplotlib 可识别的 [0,1] 范围
    #         small_rgb = np.array([28, 134, 167]) / 255.0
    #         large_rgb = np.array([174, 46, 46]) / 255.0

    #         # 判断哪个label是“较小簇”，哪个是“较大簇”
    #         label_sizes = {label: np.sum(sorted_labels == label) for label in unique_labels}
    #         sorted_by_size = sorted(label_sizes.items(), key=lambda x: x[1])  # 从小到大

    #         # 分配颜色：小簇用蓝绿，大簇用深红
    #         label_to_color = {
    #             sorted_by_size[1][0]: small_rgb,
    #             sorted_by_size[0][0]: large_rgb
    #         }
    #     else:
    #         # 多簇情况，使用默认色盘
    #         colors = plt.cm.get_cmap('Paired')(np.linspace(0, 1, num_labels))
    #         label_to_color = {label: colors[i] for i, label in enumerate(sorted(unique_labels))}


    #     def get_colors(mask):
    #         return [label_to_color[label] for label in sorted_labels[mask]]

    #     # 上段
        
    #     ax0 = plt.subplot(gs[0])
    #     ax0.set_title("Stage 3", fontsize=25)
    #     colors_upper = get_colors(upper_mask)
    #     sizes_upper = [150 if np.allclose(c, large_rgb) else 50 for c in colors_upper]
    #     ax0.scatter(np.arange(len(sorted_data))[upper_mask],
    #                 sorted_data[upper_mask],
    #                 c=colors_upper,
    #                 s=sizes_upper,
    #                 alpha=0.7)
    #     ax0.tick_params(axis='both', labelsize=20) 
    #     ax0.set_ylim(20, max(sorted_data) + 100)
    #     ax0.spines['bottom'].set_visible(False)
    #     # ax0.set_ylabel('Max Activation Value', fontsize=18, labelpad=14)
    #     ax0.tick_params(bottom=False, labelbottom=False)

    #     # 中段
    #     ax1 = plt.subplot(gs[1], sharex=ax0)
    #     colors_mid = get_colors(mid_mask)
    #     sizes_mid = [150 if np.allclose(c, large_rgb) else 50 for c in colors_mid]
    #     ax1.scatter(np.arange(len(sorted_data))[mid_mask],
    #                 sorted_data[mid_mask],
    #                 c=colors_mid,
    #                 s=sizes_mid,
    #                 alpha=0.7)
    #     ax1.tick_params(axis='both', labelsize=20) 
    #     ax1.set_ylim(6, 50)
    #     ax1.spines['top'].set_visible(False)
    #     ax1.spines['bottom'].set_visible(False)
    #     ax1.tick_params(bottom=False, labelbottom=False)

    #     # 下段
    #     ax2 = plt.subplot(gs[2], sharex=ax0)
    #     colors_lower = get_colors(lower_mask)
    #     sizes_lower = [150 if np.allclose(c, large_rgb) else 50 for c in colors_lower]
    #     ax2.scatter(np.arange(len(sorted_data))[lower_mask],
    #                 sorted_data[lower_mask],
    #                 c=colors_lower,
    #                 s=sizes_lower,
    #                 alpha=0.7)
    #     ax2.tick_params(axis='both', labelsize=20)
    #     ax2.set_ylim(0, 6.5)
    #     ax2.set_xlabel('Channel Index', fontsize=24, labelpad=10)
    #     ax2.spines['top'].set_visible(False)

    #     # 添加断轴标记
    #     d = .015
    #     kwargs = dict(color='k', clip_on=False, linewidth=3)

    #     # ax0 & ax1
    #     ax0.plot((-d, +d), (-d, +d), transform=ax0.transAxes, **kwargs)
    #     ax0.plot((1 - d, 1 + d), (-d, +d), transform=ax0.transAxes, **kwargs)
    #     ax1.plot((-d, +d), (1 - d, 1 + d), transform=ax1.transAxes, **kwargs)
    #     ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax1.transAxes, **kwargs)

    #     # ax1 & ax2
    #     ax1.plot((-d, +d), (-d, +d), transform=ax1.transAxes, **kwargs)
    #     ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, **kwargs)
    #     ax2.plot((-d, +d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
    #     ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)

    #     # # 注释信息（中间段）+ 图例
    #     # if annotation:
    #     #     ax1.text(0.5, 0.82, annotation, transform=ax1.transAxes,
    #     #             fontsize=10, verticalalignment='top', horizontalalignment='center',
    #     #             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.6'))

    #     from matplotlib.patches import Patch

    #     if color and num_labels == 2:
    #         legend_elements = [
    #             Patch(facecolor=large_rgb, label='Outlier Channels'),
    #             Patch(facecolor=small_rgb, label='Normal Channels')
    #         ]

    #         # 添加图例到整张图中，而非某个子图
    #         fig.legend(
    #             handles=legend_elements,
    #             loc='upper center',
    #             bbox_to_anchor=(0.51, 0.65),  # 整体图像中间上方，越小越往下
    #             ncol=1,
    #             frameon=True,
    #             framealpha=0.4,
    #             fontsize=24
    #         )


    #     fig.suptitle(title)
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()

    # def visualize_results(self, data, layer_name, base_dir=os.path.join(outpath, "visualizations")):
    #     """
    #     聚类可视化结果：绘制聚类分布图（断轴）与量化误差曲线图。
    #     """
    #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_dir = os.path.join(base_dir, f"{current_time}_{layer_name}")
    #     os.makedirs(save_dir, exist_ok=True)

    #     ks = sorted(self.results.keys())
    #     quant_errors = [self.results[k]['quant_error'] for k in ks]
    #     best_k_idx = np.argmin(quant_errors)

    #     # # 1. 绘制量化误差趋势图
    #     # plt.figure(figsize=(12, 6))
    #     # plt.plot(ks, quant_errors, 'o-', color='tab:blue', label='Quantization Error')
    #     # plt.scatter(ks[best_k_idx], quant_errors[best_k_idx], marker='*', color='red', s=200, zorder=5, label='Best K')
    #     # plt.xlabel('Number of Clusters (K)')
    #     # plt.ylabel('Quantization Error')
    #     # plt.title(f'Quantization Error Curve - {layer_name}')
    #     # plt.legend()
    #     # plt.grid(True)
    #     # plt.savefig(os.path.join(save_dir, "quantization_error_curve.png"), dpi=300, bbox_inches='tight')
    #     # plt.close()

    #     # 2. 每个K的聚类可视化（左：labels，右：final_labels）
    #     for k in ks:
    #         result = self.results[k]
    #         labels = np.asarray(result['labels'])
    #         final_labels = np.asarray(result['final_labels'])
    #         quant_error = result['quant_error']
    #         abnormal_count = len(result['abnormal_channels'])

    #         sorted_data = np.asarray(data).flatten()
    #         sorted_labels = labels
    #         sorted_final_labels = final_labels

    #         annotation = f"K = {k}\nError = {quant_error:.4f}\nOutlier Channels = {abnormal_count}"

    #         # # 左图：原始labels
    #         # self.plot_cluster_with_broken_y(
    #         #     sorted_data=sorted_data,
    #         #     sorted_labels=sorted_labels,
    #         #     title=f"K={k} - Original Labels",
    #         #     save_path=os.path.join(save_dir, f"k_{k}_original_labels_broken.png")
    #         # )

    #         # 右图：二次聚类后的final_labels
    #         self.plot_cluster_with_broken_y(
    #             sorted_data=sorted_data,
    #             sorted_labels=sorted_final_labels,
    #             title=f"Stage 3",
    #             annotation=annotation,
    #             save_path=os.path.join(save_dir, f"k_{k}_final_labels_broken.png"),
    #             color = True 
    #         )

    #     logger4.info(f"All visualizations saved to: {save_dir}")


    def plot_combined_cluster(self, sorted_data, sorted_labels_1, sorted_labels_2, title, save_path=None):
        """
        绘制聚类结果对比图：左为第一次聚类（断轴 + 中部连接三角形标记），右为每簇最大值聚类（断轴），颜色来源为二次聚类 final_labels。
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import RegularPolygon
        import numpy as np

        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(3, 2, width_ratios=[2, 2], height_ratios=[1, 1, 2], hspace=0.05, wspace=0.3)

        # 获取 mask
        data = sorted_data
        lower_mask = data < 7
        mid_mask = (data >= 6.5) & (data < 50)
        upper_mask = data >= 50

        # 二次聚类颜色
        small_rgb = np.array([28, 134, 167]) / 255.0
        large_rgb = np.array([174, 46, 46]) / 255.0

        # label 配色
        unique_labels_1 = np.unique(sorted_labels_1)
        num_labels_1 = len(unique_labels_1)
        base_colors = np.array([
            (0.62, 0.855, 0.898),  # 蓝色
            (0.173, 0.627, 0.173),  # 绿色
            (0.58, 0.404, 0.741),   # 紫色
            (0.89, 0.467, 0.761),   # 粉紫色
            (0.737, 0.741, 0.133),  # 芥末黄
            (1.0, 0.733, 0.471)     # 明亮橙色
        ])

        if num_labels_1 <= len(base_colors):
            colors_1 = base_colors[np.random.choice(len(base_colors), num_labels_1, replace=False)]
        else:
            # 使用 tab20 补充其余颜色
            extra_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))[:, :3]
            extra_pool = [tuple(c) for c in extra_colors if tuple(c) not in base_colors.tolist()]
            needed = num_labels_1 - len(base_colors)
            random.shuffle(extra_pool)
            colors_1 = np.vstack([base_colors, extra_pool[:needed]])
        
        label_to_color_1 = {label: colors_1[i] for i, label in enumerate(sorted(unique_labels_1))}

        # 打印并记录每个 label 对应的 RGB 值
        for label in sorted(unique_labels_1):
            rgb = tuple(np.round(label_to_color_1[label][:3], 3))  # 保留3位小数更简洁
            logger3.info(f"Label {label}: RGB = {rgb}")


        cluster_max_values = [np.max(data[sorted_labels_1 == label]) for label in sorted(unique_labels_1)]
        label_to_maxval = {label: val for label, val in zip(sorted(unique_labels_1), cluster_max_values)}
        final_labels = [0 if val <= np.median(cluster_max_values) else 1 for val in cluster_max_values]
        label_to_final_label = {label: fl for label, fl in zip(sorted(unique_labels_1), final_labels)}

        def get_colors(labels, mapping):
            return [mapping[lbl] for lbl in labels]

        # ------- 左图三段式断轴 ------- #
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[1, 0], sharex=ax0)
        ax2 = plt.subplot(gs[2, 0], sharex=ax0)

        for ax, mask, ylim in [(ax0, upper_mask, (50, max(data) + 100)),
                            (ax1, mid_mask, (6, 50)),
                            (ax2, lower_mask, (0, 6.5))]:
            ax.scatter(np.arange(len(data))[mask], data[mask],
                    c=get_colors(sorted_labels_1[mask], label_to_color_1), s=50, alpha=0.7)
            ax.set_ylim(ylim)
            # ax.spines['top'].set_visible(False)

        ax0.spines['bottom'].set_visible(False)
        ax0.tick_params(bottom=False, labelbottom=False)
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(bottom=False, labelbottom=False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_xlabel('Channel Index', fontsize=24, labelpad=10)
        ax1.set_ylabel('Max Activation Value', fontsize=24, labelpad=10)
        ax0.tick_params(axis='both', labelsize=20) 
        ax1.tick_params(axis='both', labelsize=20) 
        ax2.tick_params(axis='both', labelsize=20) 

        # 添加断轴标记
        d = .015
        kwargs = dict(color='k', clip_on=False, linewidth=3)
        ax0.plot((-d, +d), (-d, +d), transform=ax0.transAxes, **kwargs)
        ax0.plot((1 - d, 1 + d), (-d, +d), transform=ax0.transAxes, **kwargs)
        ax1.plot((-d, +d), (1 - d, 1 + d), transform=ax1.transAxes, **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax1.transAxes, **kwargs)
        ax1.plot((-d, +d), (-d, +d), transform=ax1.transAxes, **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, **kwargs)
        ax2.plot((-d, +d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)

        # 添加三角形标记（中心竖直对齐）
        # 根据左图每个 label 的中位数激活值，从高到低排序
        label_medians = {
            label: np.median(data[sorted_labels_1 == label])
            for label in unique_labels_1
        }
        # sorted_labels_by_height = sorted(label_medians.items(), key=lambda x: -x[1])  # 从高到低
        

        # -------- 替换 center_ax 绘制三角形部分，使其和右图断轴对齐 -------- #  
        center_ax = fig.add_axes([0.48, 0.1, 0.04, 0.8])
        center_ax.axis('off')

        cluster_max_values = [
            np.max(data[sorted_labels_1 == label])
            for label in sorted(unique_labels_1)
        ]

        y_breaks = [0, 6.5, 50, max(cluster_max_values) + 100]  # 每段上界
        y_heights = [2, 1, 1]  # 下 -> 中 -> 上（要和 y_breaks 顺序一致）
        total = sum(y_heights)
        height_ratios = [h / total for h in y_heights]
        y_starts = [0]
        for h in height_ratios[:-1]:
            y_starts.append(y_starts[-1] + h)

        for i, val in enumerate(cluster_max_values):
            if val < y_breaks[1]:
                frac = (val - y_breaks[0]) / (y_breaks[1] - y_breaks[0])
                y = y_starts[0] + frac * height_ratios[0]
            elif val < y_breaks[2]:
                frac = (val - y_breaks[1]) / (y_breaks[2] - y_breaks[1])
                y = y_starts[1] + frac * height_ratios[1]
            else:
                frac = (val - y_breaks[2]) / (y_breaks[3] - y_breaks[2])
                y = y_starts[2] + frac * height_ratios[2]

            center_ax.scatter(0.5, y, marker='o', s=300, color=label_to_color_1[unique_labels_1[i]])



        # # 画布中央三角标记
        # center_ax = fig.add_axes([0.48, 0.1, 0.04, 0.8])
        # center_ax.axis('off')
        # label_medians = {label: np.median(data[sorted_labels_1 == label]) for label in unique_labels_1}
        # sorted_labels_by_height = sorted(label_medians.items(), key=lambda x: -x[1])
        # for idx, (label, _) in enumerate(sorted_labels_by_height):
        #     center_ax.scatter(0.5, 0.85 - idx * 0.065, marker='^', s=500, color=label_to_color_1[label])


        ax0.set_title("Stage 1", fontsize=25)

        small_rgb = np.array([28, 134, 167]) / 255.0
        large_rgb = np.array([174, 46, 46]) / 255.0

        axr0 = plt.subplot(gs[0, 1])
        axr1 = plt.subplot(gs[1, 1], sharex=axr0)
        axr2 = plt.subplot(gs[2, 1], sharex=axr0)

        cluster_max_values = [np.max(data[sorted_labels_1 == label]) for label in sorted(unique_labels_1)]
        sorted_label_indices = [np.where(sorted_labels_1 == label)[0][0] for label in sorted(unique_labels_1)]
        final_labels_ref = sorted_labels_2[sorted_label_indices]
        color_map = [small_rgb, large_rgb]

        zipped = list(zip(cluster_max_values, sorted(unique_labels_1), final_labels_ref))
        zipped.sort(key=lambda x: x[0])  # 按最大值升序
        k = len(unique_labels_1) 
        if k >= 3:
            idx_swap_1 = 0
            idx_swap_2 = k // 2  if (k // 2 ) < k else k - 1  # 防止越界
            zipped[idx_swap_1], zipped[idx_swap_2] = zipped[idx_swap_2], zipped[idx_swap_1]

        cluster_max_values_sorted, labels_sorted, final_labels_sorted = zip(*zipped)

        for i, val in enumerate(cluster_max_values_sorted):
            label = labels_sorted[i]
            cluster_final_label = final_labels_sorted[i]
            color = color_map[cluster_final_label]
            if val < 6.5:
                axr2.scatter(i, val, color=color_map[cluster_final_label], s=700, marker='^', edgecolors='none')
            elif val < 50:
                axr1.scatter(i, val, color=color_map[cluster_final_label], s=700, marker='^', edgecolors='none')
            else:
                axr0.scatter(i, val, color=color_map[cluster_final_label], s=700, marker='^', edgecolors='none')

        axr0.set_ylim(50, max(cluster_max_values) + 100)
        axr1.set_ylim(6.5, 50)
        axr2.set_ylim(0, 6.5)

        axr0.spines['bottom'].set_visible(False)
        axr1.spines['top'].set_visible(False)
        axr2.spines['top'].set_visible(False)
        axr0.tick_params(bottom=False, labelbottom=False)
        axr1.spines['bottom'].set_visible(False)
        axr1.tick_params(bottom=False, labelbottom=False)
        # axr2.set_ylabel('Max Activation Value')
        axr2.set_xlabel('Cluster Index', fontsize=24, labelpad=10)
        axr0.tick_params(axis='both', labelsize=20) 
        axr1.tick_params(axis='both', labelsize=20) 
        axr2.tick_params(axis='both', labelsize=20) 


        axr0.plot((-d, +d), (-d, +d), transform=axr0.transAxes, **kwargs)
        axr0.plot((1 - d, 1 + d), (-d, +d), transform=axr0.transAxes, **kwargs)
        axr1.plot((-d, +d), (1 - d, 1 + d), transform=axr1.transAxes, **kwargs)
        axr1.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axr1.transAxes, **kwargs)
        axr1.plot((-d, +d), (-d, +d), transform=axr1.transAxes, **kwargs)
        axr1.plot((1 - d, 1 + d), (-d, +d), transform=axr1.transAxes, **kwargs)
        axr2.plot((-d, +d), (1 - d, 1 + d), transform=axr2.transAxes, **kwargs)
        axr2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axr2.transAxes, **kwargs)

        axr0.set_title("Stage 2", fontsize=25)

        fig.suptitle(title)

        from matplotlib.patches import Patch

        # 定义图例元素：两种颜色代表不同类型簇
        legend_elements = [
            Patch(facecolor=large_rgb, label='Non-main Clusters'),
            Patch(facecolor=small_rgb, label='Main Clusters')
        ]

        # 添加图例到整个图上（而不是某个子图）
        fig.legend(
            handles=legend_elements,
            loc='center right',             # 图像右侧居中
            bbox_to_anchor=(0.90, 0.535),     # 0.98 表示靠近右边界，0.5 为垂直居中
            ncol=1,
            frameon=True,
            framealpha=0.9,
            fontsize=24
        )


        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_results(self, data, layer_name, base_dir=os.path.join(outpath, "visualizations")):
        """
        聚类可视化结果：聚合左右图对比，增加异常通道放大效果。
        """
        import os
        import numpy as np
        from datetime import datetime

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, f"{current_time}_{layer_name}")
        os.makedirs(save_dir, exist_ok=True)

        ks = sorted(self.results.keys())
        quant_errors = [self.results[k]['quant_error'] for k in ks]
        best_k_idx = np.argmin(quant_errors)

        orange_rgb = (242/255, 170/255, 69/255)
        deep_orange_rgb = (247/255, 181/255, 46/255)
        test_orange_rgb = (234/255, 189/255, 93/255)
        light_blue_rgb = (118/255, 188/255, 193/255)
        deep_blue_rgb = (5/255, 174/255, 186/255)

        light_red_rgb = (208/255, 153/255, 153/255)
        deep_red_rgb = (189/255, 117/255, 135/255)

        green_rgb = (153/255, 204/255, 153/255)
        deep_green_rgb = (1/255, 113/255, 0/255)

        line_rgb = (0/255, 157/255, 158/255)
        star_rgb = (242/255, 112/255, 34/255)

        # 绘制误差曲线
        plt.figure(figsize=(7, 6))
        plt.plot(ks, quant_errors, 'o-', color=line_rgb, label='Quantization Error', linewidth=5, markersize=12)
        plt.scatter(ks[best_k_idx], quant_errors[best_k_idx], marker='*', color=star_rgb, s=1200, zorder=5, label='Best K')
        plt.xlabel('Number of Clusters (K)', fontsize=24, labelpad=10)
        plt.ylabel('Quantization Error', fontsize=24, labelpad=10)
        plt.title(' ', fontsize=24)
        plt.tick_params(axis='both', labelsize=20)  # 坐标轴数值字体大小
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color=line_rgb, label='Quantization Error',
                linestyle='-', markersize=12, linewidth=3),
            Line2D([0], [0], marker='*', color=star_rgb, label='Best K',
                linestyle='None', markersize=20)  # 控制图例中五角星的大小
        ]
        
        # 放置图例到右下角（图内）
        plt.legend(handles=legend_elements,loc='lower right', fontsize=16,bbox_to_anchor=(0.97, 0.2), frameon=True)
        # plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "quantization_error_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # for k in ks:
        #     result = self.results[k]
        #     labels = np.asarray(result['labels'])
        #     final_labels = np.asarray(result['final_labels'])
        #     quant_error = result['quant_error']
        #     abnormal_count = len(result['abnormal_channels'])

        #     sorted_data = np.asarray(data).flatten()
        #     sorted_labels = labels
        #     sorted_final_labels = final_labels

        #     # 生成并列图（左：原始聚类标签，右：二次聚类）
        #     self.plot_combined_cluster(
        #         sorted_data=sorted_data,
        #         sorted_labels_1=sorted_labels,
        #         sorted_labels_2=sorted_final_labels,
        #         title=f"Clustering Comparison - K={k} | Error={quant_error:.4f} | Outliers={abnormal_count}",
        #         save_path=os.path.join(save_dir, f"k_{k}_cluster_comparison.png")
        #     )

        logger4.info(f"All visualizations saved to: {save_dir}")


    def _quantization_fitness(self, individual,cluster_groups, data,k, max_values, activation_tensor):
        """量化优化的适应度函数"""
        centers = np.array(individual).reshape(k, 1)
        
        # 分配标签给原始簇
        distances = np.array([np.abs(data - center) for center in centers])
        cluster_labels = np.argmin(distances, axis=0).flatten()
        
        # 将簇标签映射回原始通道
        channel_labels = np.zeros(activation_tensor.shape[-1])
        for group_idx, (channel_indices, _) in enumerate(cluster_groups):
            channel_labels[channel_indices] = cluster_labels[group_idx]
        error,_ = self._compute_quantization_error(channel_labels, activation_tensor, max_values)
        
        # 计算量化误差
        return  (float(error),)

    def _init_individual(self, n_samples,data, k):
        """初始化个体"""
        indices = np.random.choice(range(n_samples), size=k, replace=False)
        
        return data[indices].flatten().tolist()
    
    def _genetic_kmeans_for_quantization(self, cluster_groups, activation_tensor, k,max_values, pop_size=50, ngen=20, cxpb=0.5, mutpb=0.2):
            """
            专门用于量化优化的遗传算法K-Means
            
            参数:
                cluster_groups: 第一次聚类的结果，每个元素是(channel_indices, channel_max_values)元组
                activation_tensor: 原始激活张量
                k: 聚类数量
                pop_size: 种群大小
                ngen: 迭代次数
                cxpb: 交叉概率
                mutpb: 变异概率
            
            返回:
                centers: 聚类中心
                labels: 聚类标签
            """
            # 准备二次聚类数据（每个原始簇的最大值）
            cluster_maxima = np.array([np.max(channel_max_values) for _, channel_max_values in cluster_groups])
            data = cluster_maxima.reshape(-1, 1)
            n_samples, n_features = data.shape
            cluster_groups = copy.deepcopy(cluster_groups)
            
            # 创建遗传算法类型
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            # 初始化工具箱
            toolbox = base.Toolbox()
            # 定义个体生成函数 - 每个个体是k个聚类中心的集合
            def create_individual():
                # 从数据中随机选择k个点作为初始中心
                indices = np.random.choice(range(n_samples), size=k, replace=False)
                centers = data[indices].flatten().tolist()
                return centers
            
            # 注册遗传算法操作
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            evaluate_func = partial(
                self._quantization_fitness,
                cluster_groups=cluster_groups,
                data=data,
                k=k,
                max_values=max_values,
                activation_tensor=activation_tensor
            )
            toolbox.register("evaluate", evaluate_func)
            # toolbox.register("evaluate", self._quantization_fitness, 
            #                 cluster_groups = cluster_groups, data=data,k=k,max_values = max_values, activation_tensor=activation_tensor)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # 运行遗传算法
            population = toolbox.population(n=pop_size)
            algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
            
            # 获取最佳结果
            best_individual = tools.selBest(population, k=1)[0]
            best_centers = np.array(best_individual).reshape(k, n_features)
            
            # 分配标签
            distances = np.array([np.sum((data - center)**2, axis=1) for center in best_centers])
            best_labels = np.argmin(distances, axis=0)
            
            return best_labels
    
    def _kmeans_with_quant_fitness(self, cluster_groups, activation_tensor, k, max_values, max_iter=20):
        """
        使用量化误差作为标准的K-means聚类，确保将k个簇合并为2个有效簇
        
        参数:
            cluster_groups: [(channel_indices, channel_max_values), ...]
            activation_tensor: 原始激活张量
            k: 第一次聚类得到的簇数量
            max_values: 各通道最大值
            max_iter: 最大迭代次数
            
        返回:
            best_labels: 形状为(k,)，表示每个原始簇属于哪个新簇(0或1)
        """
        # 1. 准备数据（使用第一次聚类各簇的最大值）
        cluster_maxima = np.array([np.max(vals) for _, vals in cluster_groups])
        data = cluster_maxima.reshape(-1, 1)
        n_samples = data.shape[0]
        
        # 强制设置二次聚类的目标簇数为2
        target_k = 2
        
        # 2. 初始化中心（确保选择两个不同的中心点）
        while True:
            centers = data[np.random.choice(n_samples, target_k, replace=False)].flatten()
            if len(np.unique(centers)) == target_k:  # 确保中心点不同
                break
        
        for _ in range(max_iter):
            # 3. 分配标签：使用量化误差作为标准
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                min_error = float('inf')
                best_j = 0
                
                for j in range(target_k):
                    # 临时分配标签
                    temp_labels = np.zeros(n_samples)
                    temp_labels[i] = j
                    
                    # 转换为个体格式
                    individual = centers.copy()
                    individual[j] = data[i]
                    
                    # 计算量化误差
                    error = self._quantization_fitness(
                        individual=individual,
                        cluster_groups=cluster_groups,
                        data=data,
                        k=target_k,
                        max_values=max_values,
                        activation_tensor=activation_tensor
                    )[0]
                    
                    if error < min_error:
                        min_error = error
                        best_j = j
                
                labels[i] = best_j
            
            # 4. 检查是否所有点都分配到了同一个簇
            if len(np.unique(labels)) < target_k:
                # 随机选择一个点分配到另一个簇
                labels[np.random.choice(n_samples)] = 1 - labels[0]
                continue
            
            # 5. 更新中心点
            new_centers = np.array([np.mean(data[labels == j]) for j in range(target_k)])
            
            # 6. 收敛检查
            if np.allclose(centers, new_centers, rtol=1e-5):
                break
            centers = new_centers
        
        # 参考之前的算法进行修改
        # 提取每簇最大值
        first_labels = cluster_groups[k]['labels']
        cluster_maxima = [np.max(max_values[first_labels == i]) for i in range(k)]
        cluster_maxima = np.array(cluster_maxima).reshape(-1, 1)

                    
        # 强制k=2的二次聚类
        kmeans = KMeans(n_clusters=2, random_state=42).fit(cluster_maxima)
        second_labels = kmeans.labels_
        final_centers = kmeans.cluster_centers_.flatten()  # 二次聚类中心
                    
        # 确定主簇（最大值较小的簇）
        if np.mean(cluster_maxima[second_labels == 0]) < np.mean(cluster_maxima[second_labels == 1]):
            main_cluster_group = 0
        else:
            main_cluster_group = 1
                    
        # 映射回原始标签
        main_clusters = [i for i in range(k) if second_labels[i] == main_cluster_group]
        final_labels = np.zeros_like(first_labels)
                    
        for new_label, orig_label in enumerate(range(k)):
            mask = (first_labels == orig_label)
            if orig_label in main_clusters:
                final_labels[mask] = 0  # 主簇
            else:
                final_labels[mask] = 1  # 异常簇
                    
        main_cluster = 0  # 主簇标签已重新映射为0
        abnormal_channels = np.where(final_labels != main_cluster)[0].tolist()
        return labels

    def _exhaustive_quant_clustering(self, cluster_groups, activation_tensor, max_values,k):
        """
        穷举所有二分类方案，找到最小量化误差的划分
        
        参数:
            cluster_groups: [(channel_indices, channel_max_values), ...]
            activation_tensor: 原始张量
            max_values: 各通道最大值
            
        返回:
            best_labels: 最优划分标签 (形状[n_clusters], 值为0或1)
            min_error: 最小量化误差
        """
        n_clusters = len(cluster_groups)
        cluster_maxima = np.array([np.max(vals) for _, vals in cluster_groups])
        
        # 生成所有可能的二分类方案（避免对称重复）
        all_combinations = []
        for i in range(1, 2**n_clusters - 1):
            # 将整数转换为二进制标签数组
            labels = [(i >> j) & 1 for j in range(n_clusters)]
            # 确保方案中同时存在0和1（即有异常通道）
            # if 0 in labels and 1 in labels:
            all_combinations.append(np.array(labels))
        
        # 评估每种划分方案
        min_error = float('inf')
        best_labels = np.zeros(n_clusters, dtype=int)
        
        for labels in all_combinations:
            # 将簇标签映射到通道
            channel_labels = np.zeros(activation_tensor.shape[-1], dtype=int)
            for group_idx, (ch_indices, _) in enumerate(cluster_groups):
                channel_labels[ch_indices] = labels[group_idx]
            
            # 计算量化误差
            error, _ = self._compute_quantization_error(
                channel_labels, activation_tensor, max_values
            )
            
            # 更新最优解
            if error < min_error:
                min_error = error
                best_labels = labels
        
        return best_labels


class QuantLinear_ar(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 i = None):
        super(QuantLinear_ar, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_ar(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.i = -1
        
    def __repr__(self):
        s = super(QuantLinear_ar, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1,args=None):
        """
        Perform forward pass using quantized or non-quantized weights, and save statistics/images to separate directories.
        """
        # if params:
        #     # 动态更新量化参数
        #     self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)
            # self.input_quantizer.channel_wise = params.get('channel_wise', self.input_quantizer.channel_wise)

        # logger3.info(f"Input Of QuantLinear_ar Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")

        if i ==None:
            i=-1
        else:
            i = i
        
        if self.use_input_quant:
            x_quant = self.input_quantizer(x, i=i, step=1, calib5=calib5,adjustment = adjustment)
                    
        if self.use_weight_quant:
            # 执行权重量化
            w = self.weight_quantizer(self.weight, i=i, step=1, calib5=calib5, is_weight=True)
        else:
            # 不进行量化，直接使用原始权重
            w = self.weight
                
        if self.use_input_quant:
            out = F.linear(x_quant, weight=w, bias=self.bias)
        else:
            out = F.linear(x, weight=w, bias=self.bias)
        return out

import torch.nn as nn
import torch.nn.functional as F

class QuantConv2d_ar(nn.Conv2d):
    """
    Class to quantize weights of given Conv2d layer
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 input_quant_params={},
                 weight_quant_params={},
                 i=None):
        super(QuantConv2d_ar, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )

        self.input_quantizer = UniformQuantizer_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.i = -1

    def __repr__(self):
        s = super(QuantConv2d_ar, self).__repr__()
        s = "(" + s + f", input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False,
                layer_name=None, params=None, save_stats=False,
                adjustment=False, num=None, a=False, num_bsz=-1, args=None):
        
        # logger3.info(f"Input Of QuantConv2d_ar Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")
        if i ==None:
            i=-1
        else:
            i = i

        if self.use_input_quant:
            x_quant = self.input_quantizer(x, i=i, step=1, calib5=calib5, adjustment=adjustment)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight, i=i, step=1, calib5=calib5, is_weight=True)
        else:
            w = self.weight

        if self.use_input_quant:
            out = F.conv2d(x_quant, weight=w, bias=self.bias,
                           stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups)
        else:
            out = F.conv2d(x, weight=w, bias=self.bias,
                           stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups)
        return out


class QuantLinear_diff(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 i = None):
        super(QuantLinear_diff, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)
        # self.input_quantizer = UniformQuantizer_group_diff(**input_quant_params)
        # self.weight_quantizer = UniformQuantizer_group_diff(**weight_quant_params)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_input_quant = False
        self.use_weight_quant = False

        self.initial_zs = 128
        # self.initial_ss = torch.zeros(6400, device=self.device)
        self.initial_ss = torch.full((6400,), .04347, device=self.device)
        # 0.02353     --3
        # 0.03137 127 --4
        # 0.03922     --5
        # 0.04706     --6
        # 0.05490     --7
        # 0.04347,128    --5.545
        self.adjustment_factor = torch.zeros(6400, device=self.device)

        self.i = None
        

    def __repr__(self):
        s = super(QuantLinear_diff, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1):
        """
        Perform forward pass using quantized or non-quantized weights, and save statistics/images to separate directories.
        """
        if i == None:
            i=-1
        else:
            i = i

        if self.use_input_quant:
            x_quant = self.input_quantizer(x, i=i, step=1, calib5=calib5,adjustment = adjustment)
        
        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight, i=i, step=1, calib5=calib5, is_weight=True)
        else:
            w = self.weight

        if self.use_input_quant:
            out = F.linear(x_quant, weight=w, bias=self.bias)

        else:
            out = F.linear(x, weight=w, bias=self.bias)
        return out


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul, self).__init__()

        input_quant_params_matmul = deepcopy(input_quant_params)
       
        self.quantizer_A = UniformQuantizer_ar(**input_quant_params_matmul)
        self.quantizer_B = UniformQuantizer_ar(**input_quant_params_matmul)
        # self.quantizer_A = UniformQuantizer_ar(**input_quant_params_matmul)
        # self.quantizer_B = UniformQuantizer_group(**input_quant_params_matmul)
        

        self.use_input_quant = False
        self.i = None

    def __repr__(self):
        s = super(QuantMatMul, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant


    def forward(self, A, B,i =  None,calib5 = False, step = None,d =False, layer_name=None, draw = False, save_stats=False):
        """
        Matrix multiplication with optional input quantization
        # """
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}A: min={A.min()}, max={A.max()}, mean={A.mean()}")
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}B: min={B.min()}, max={B.max()}, mean={B.mean()}")
        self.i = i
        
        # layer_name = self.__class__.__name__ if layer_name is None else layer_name

        # 判断是否启用输入量化
        if self.use_input_quant:
            if draw:
                # 保存量化前后的数据到量化路径
                save_and_plot(A, f"Original Activation of Step{step} Timestep{i} {layer_name} A", f"original_step{step}_time{i}_{layer_name}_A.png", output_dir2)
                save_and_plot(B, f"Original Activation of Step{step} Timestep{i} {layer_name} B", f"original_step{step}_time{i}_{layer_name}_B.png", output_dir2)
           
            # 记录原始激活值
            # logger1.info(f"Original A: min={A.min()}, max={A.max()}, mean={A.mean()}")
            # logger1.info(f"Original B: min={B.min()}, max={B.max()}, mean={B.mean()}")

            # 执行输入量化
            A = self.quantizer_A(A, i=i, calib5=calib5, step=step)
            B = self.quantizer_B(B, i=i, calib5=calib5, step=step)
            
            if draw:
                # 保存量化前后的数据到量化路径
                save_and_plot(A, f"Quantized Activation of Step{step} Timestep{i} {layer_name} A", f"quantized_step{step}_time{i}_{layer_name}_A.png", output_dir3)
                save_and_plot(B, f"Quantized Activation of Step{step} Timestep{i} {layer_name} B", f"quantized_step{step}_time{i}_{layer_name}_B.png", output_dir3)
            
            # 记录量化后的激活值
            # logger2.info(f"Quantized A: min={A.min()}, max={A.max()}, mean={A.mean()}")
            # logger2.info(f"Quantized B: min={B.min()}, max={B.max()}, mean={B.mean()}")
        else:
            if draw:
                # 如果未启用输入量化，保存原始数据到原始路径
                save_and_plot(A, f"Original Activation of Step{step} Timestep{i} {layer_name} A", f"original_step{step}_time{i}_{layer_name}_A.png", output_dir1)
                save_and_plot(B, f"Original Activation of Step{step} Timestep{i} {layer_name} B", f"original_step{step}_time{i}_{layer_name}_B.png", output_dir1)
            
            # 记录原始激活值
            # logger3.info(f"Original A: min={A.min()}, max={A.max()}, mean={A.mean()}")
            # logger3.info(f"Original B: min={B.min()}, max={B.max()}, mean={B.mean()}")
                  
        # 执行矩阵乘法
        out = A @ B
        # logger3.info(f"Output Of Step{step} Timestep{i} {layer_name}: min={out.min()}, max={out.max()}, mean={out.mean()}")
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from quant.quantizer import UniformQuantizer_scale_channels, UniformQuantizer_diff
import logging

# logger4 = logging.getLogger("QuantConv2d_scale_channels")

class QuantConv2d_scale_channels(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 input_quant_params={},
                 weight_quant_params={},
                 i=None,
                 args=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.input_quantizer = UniformQuantizer_scale_channels(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False
        self.abnormal_channels = None
        self.threshold = 5.545
        self.i = i
        self.count = 0
        self.args = args

        self.save_dir = "/opt/data/private/GaoJing/mar_1/visualizations/conv_boxplot_output"
        os.makedirs(self.save_dir, exist_ok=True)

    def __repr__(self):
        s = super().__repr__()
        s = "(" + s + f", input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False,
                layer_name=None, params=None, save_stats=False,
                adjustment=False, num=None, a=False, num_bsz=-1,
                sign_scaling=False, scale_quant=None, shift_quant=None, args=None):
        
        self.i = i
        if args is not None:
            self.threshold = 3

        if self.use_input_quant:
            if x.abs().max() >= self.threshold:
                self.count += 1
                
                logger4.info(f"{layer_name}, threshold={self.threshold}, SR triggered {self.count} times.")

                b, c, h, w = x.shape
                x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)  # shape: [B*H*W, C]
                channel_max = x_flat.abs().max(dim=0)[0]
                self.abnormal_channels = torch.where(channel_max >= self.threshold)[0].tolist()

                abnormal_max = channel_max.max().item()
                scale_ratio = self.threshold / abnormal_max
                self.abnormal_channels = torch.tensor(self.abnormal_channels, device=x.device)

                # scale down abnormal channels
                x_flat[..., self.abnormal_channels] *= scale_ratio
                x_scaled = x_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
                x_quant = self.input_quantizer(x_scaled, i=i, step=step, calib5=calib5, adjustment=adjustment)

                # scale back
                x_quant_flat = x_quant.permute(0, 2, 3, 1).reshape(-1, c)
                x_quant_flat[..., self.abnormal_channels] /= scale_ratio
                x_quant = x_quant_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            else:
                x_quant = self.input_quantizer(x, i=i, step=step, calib5=calib5, adjustment=adjustment)
        else:
            x_quant = x

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight, i=i, step=step, calib5=calib5, is_weight=True)
        else:
            w = self.weight

        return F.conv2d(x_quant, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear_ar_scale(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}):
        super(QuantLinear_ar_scale, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_ar(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False
        self.abnormal_channels = []  # 仅计算一次异常通道
        self.scaling_factors_dict = { }  # 每次迭代都计算缩放因子
        self.is_quantization_acceptable = True
        self.symbol_matrix = None
    
    def __repr__(self):
        s = super(QuantLinear_ar_scale, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def calculate_channel_stats(self, tensor):
        """
        计算张量在指定通道上的统计指标（最大值、最小值、均值、中位数、标准差）
        
        参数:
            tensor (torch.Tensor): 输入张量（假设通道在最后一维）
            target_channels (list): 需要计算统计指标的通道索引列表
            
        返回:
            dict: 包含统计结果的字典
            str: 格式化的日志内容字符串
        """
        # 目标通道列表
        target_channels = [351, 1657, 1747, 3472, 3616, 3701]

        # 提取指定通道的数据
        channel_data = tensor[..., target_channels]  # 形状为 [..., len(target_channels)]
        
        # 计算统计指标
        stats = {
            "max": torch.max(channel_data, dim=-1).values,  # 沿最后一个维度取最大值
            "min": torch.min(channel_data, dim=-1).values,  # 沿最后一个维度取最小值
            "mean": torch.mean(channel_data, dim=-1),       # 沿最后一个维度取均值
            "median": torch.median(channel_data, dim=-1).values,  # 沿最后一个维度取中位数
            "std": torch.std(channel_data, dim=-1),         # 沿最后一个维度取标准差
        }
        
        # 获取统计结果的形状（除通道维度外的所有维度）
        stat_shape = stats["max"].shape
        
        # 格式化输出
        log_content = [
            "\n" + "=" * 80,
            "张量通道统计报告",
            "-" * 80,
            f"目标通道: {target_channels}",
            "-" * 80,
            "通道ID   |    最大值    |    最小值    |    均值     |    中位数   |    标准差   ",
            "-" * 80
        ]
        
        for j in range(len(target_channels)):
            log_line = (
                f"{target_channels[j]:<9} | "
                f"{stats['max'][j]:>11.6f} | "
                f"{stats['min'][j]:>11.6f} | "
                f"{stats['mean'][j]:>11.6f} | "
                f"{stats['median'][j]:>11.6f} | "
                f"{stats['std'][j]:>11.6f}"
            )
            log_content.append(log_line)
        
        log_content.extend([
            "-" * 80,
            f"统计维度: 除通道维度外的所有维度 (原始张量形状: {tensor.shape})",
            f"统计结果形状: {stat_shape}",
            "=" * 80 + "\n"
        ])
        
        return stats, "\n".join(log_content)

    def adjust_and_quantize(self, tensor, quantizer, is_weight=False, i=None, step=None, calib5=False,layer_name = None, adjustment=False):
        """
        使用 K-Means 进行通道划分，并调整特定通道的值
        """
        if is_weight: # 权重
            tensor_clone = tensor.clone() 
            return quantizer(tensor_clone, i=i, step=step, calib5=calib5, is_weight=is_weight, adjustment=adjustment)
        else: # 激活
            if calib5:
                original_shape = tensor.shape
                reshaped = tensor.view(-1, tensor.shape[-1])  # [..., N]
                max_values, _ = reshaped.max(dim=0)  # [N]
                activation_tensor = reshaped.clone()
                max_original = activation_tensor.max()
                min_original = activation_tensor.min()
                
                # _, log_content = self.calculate_channel_stats(reshaped) # 测试思路是否正确，通道是否能对应的上，#此通道是针对L64_encoder0fc2
                # print(log_content)

                if isinstance(self.abnormal_channels, (list, np.ndarray)) and len(self.abnormal_channels) == 0: # 为空 表示未初始化
                        optimizer = GeneticKMeans(visualize=True)
                        self.abnormal_channels = optimizer.optimize(
                            activation_tensor=activation_tensor,
                            layer_name=layer_name
                        )
                        max_after_GeneticKMeans = activation_tensor.max()
                        min_after_GeneticKMeans = activation_tensor.min()
                # 计算正常通道的最大值
                normal_mask = torch.ones_like(max_values, dtype=torch.bool)
                normal_mask[self.abnormal_channels] = False
                cluster_max = max_values[normal_mask].max().item()
                            
                # 初始化缩放因子存储结构（字典：通道索引 -> 缩放因子列表）
                if not hasattr(self, 'scaling_factors_dict'):
                            self.scaling_factors_dict = {}
                            
                # 为每个异常通道单独计算并存储缩放因子
                for ch in self.abnormal_channels:
                    scaling_factor = cluster_max / max_values[ch].item() if max_values[ch].item() != 0 else 1.0
                    if scaling_factor > 1.0:
                        scaling_factor = 1.0
                    if ch not in self.scaling_factors_dict:
                        self.scaling_factors_dict[ch] = []
                    self.scaling_factors_dict[ch].append(scaling_factor)
                    reshaped[:, ch] *= scaling_factor
                max = reshaped.max()
                min = reshaped.min()
                            
                quantized_tensor = quantizer(reshaped, i=i, step=step, calib5=calib5, is_weight=is_weight)
                max_q = quantized_tensor.max()
                min_q = quantized_tensor.min()
                
                for ch in self.abnormal_channels:
                    quantized_tensor[:, ch] /= self.scaling_factors_dict[ch][step]

                restored_tensor = quantized_tensor.view(original_shape)

                # _, log_content = self.calculate_channel_stats(reshaped)
                # print(log_content)

            else:
                restored_tensor = quantizer(tensor, i=i, step=step, calib5=calib5, is_weight=is_weight, adjustment=adjustment)
          
        return restored_tensor
    
    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1):
        """
        Perform forward pass using quantized or non-quantized weights, with channel-wise adjustment.
        """
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)

        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")
        if self.use_input_quant:
            x = self.adjust_and_quantize(x, self.input_quantizer, i=i, step=step, calib5=calib5, is_weight=False,layer_name = layer_name, adjustment=adjustment)
        
        if self.use_weight_quant:
            # if step == 0:
            #     # 获取 abnormal_channels 对应的通道的值
            #     abnormal_values = self.weight[:, self.abnormal_channels]  # 这里假设 tensor 是二维的，第一维是 batch size，第二维是通道数
            #     max_value = abnormal_values.max(dim=0)[0]  # 获取每个通道的最大值
            #     for idx, max_val in zip(self.abnormal_channels, max_value):
            #         logger3.info(f"weight: Channel {idx} max value: {max_val.item()}")

            weight = self.adjust_and_quantize(self.weight, self.weight_quantizer, i=i, step=step, calib5=calib5, is_weight=True,layer_name = layer_name, adjustment=adjustment)
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)

class QuantLinear_ar_outlier(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}):
        super(QuantLinear_ar_outlier, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_group(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False
        self.abnormal_channels = []  # 仅计算一次异常通道
        # self.scaling_factors_dict = { }  # 每次迭代都计算缩放因子
        # self.is_quantization_acceptable = True
        # self.symbol_matrix = None

        self.recorded_x_values = []     # 用于保存 step==0 且 i in 0~99 时的 x
        self.recording_enabled = False  # 控制是否记录
        self.save_dir = "/opt/data/private/GaoJing/deeplearnng/mar/plot/model_comparison"
        os.makedirs(self.save_dir, exist_ok=True)
        self.recorded_x_with_index = []
        self.once = False
        
    
    def __repr__(self):
        s = super(QuantLinear_ar_outlier, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def enable_recording(self):
        self.recording_enabled = True
        self.recorded_x_values = []

    def adjust_and_quantize(self, tensor, quantizer, is_weight=False, i=None, step=None, calib5=False,layer_name = None, adjustment=False,args = None):
        """
        使用 K-Means 进行通道划分，并调整特定通道的值
        """
        if is_weight:
            tensor_clone = tensor.clone() 
            return quantizer(tensor_clone, i=i, step=step, calib5=calib5, is_weight=is_weight, adjustment=adjustment)
        else: # 激活
            if calib5:
                # original_shape = tensor.shape
                reshaped = tensor.view(-1, tensor.shape[-1])  # [..., N]
                max_values, _ = reshaped.max(dim=0)  # [N]
                activation_tensor = reshaped.clone()

                if isinstance(self.abnormal_channels, (list, np.ndarray)) and len(self.abnormal_channels) == 0: # 为空 表示未初始化
                        optimizer = GeneticKMeans(visualize=True)
                        self.abnormal_channels = optimizer.optimize(
                            activation_tensor=activation_tensor,
                            layer_name=layer_name,args = args
                        )

                # 计算正常通道的最大值
                normal_mask = torch.ones_like(max_values, dtype=torch.bool)
                normal_mask[self.abnormal_channels] = False
                cluster_max = max_values[normal_mask].max().item()

                # 将异常通道提取出来，以单独的指标进行量化，再复原
                tensor_clone = tensor.clone() 
                abnormal_tensor = tensor_clone[..., self.abnormal_channels]
                            
                quantized_tensor = quantizer(tensor, group_name="low", i=i, step=step, calib5=calib5, is_weight=is_weight, threshold = cluster_max)
                quantized_abnormal_tensor = quantizer(abnormal_tensor, group_name="high", i=i, step=step, calib5=calib5, is_weight=is_weight)
                quantized_tensor[..., self.abnormal_channels] = quantized_abnormal_tensor
            else:
                abnormal_tensor = tensor[..., self.abnormal_channels]
                quantized_tensor = quantizer(tensor, group_name="low", i=i, step=step, calib5=calib5, is_weight=is_weight)
                quantized_abnormal_tensor = quantizer(abnormal_tensor, group_name="high", i=i, step=step, calib5=calib5, is_weight=is_weight)
                quantized_tensor[..., self.abnormal_channels] = quantized_abnormal_tensor
        return quantized_tensor
    
    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1,args = None):
        """
        Perform forward pass using quantized or non-quantized weights, with channel-wise adjustment.
        """
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)

        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")
        if self.use_input_quant:
            x = self.adjust_and_quantize(x, self.input_quantizer, i=i, step=step, calib5=calib5, is_weight=False,layer_name = layer_name, adjustment=adjustment,args = args)
        
        if self.use_weight_quant:
            weight = self.adjust_and_quantize(self.weight, self.weight_quantizer, i=i, step=step, calib5=calib5, is_weight=True,layer_name = layer_name, adjustment=adjustment)
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)

class QuantConv2d_ar_outlier(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, input_quant_params={}, weight_quant_params={}):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.input_quantizer = UniformQuantizer_group(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False
        self.abnormal_channels = []
        
        self.recorded_x_values = []
        self.recording_enabled = False
        self.save_dir = "/opt/data/private/GaoJing/deeplearnng/mar/plot/model_comparison"
        os.makedirs(self.save_dir, exist_ok=True)
        self.recorded_x_with_index = []
        self.once = False

    def __repr__(self):
        s = super().__repr__()
        return f"({s}, input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def enable_recording(self):
        self.recording_enabled = True
        self.recorded_x_values = []

    def adjust_and_quantize(self, tensor, quantizer, is_weight=False, i=None, step=None, calib5=False,
                            layer_name=None, adjustment=False, args=None):
        if is_weight:
            tensor_clone = tensor.clone()
            return quantizer(tensor_clone, i=i, step=step, calib5=calib5, is_weight=True, adjustment=adjustment)
        else:
            if calib5:
                N, C, H, W = tensor.shape
                reshaped = tensor.permute(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]
                max_values, _ = reshaped.abs().max(dim=0)  # [C]

                if isinstance(self.abnormal_channels, (list, np.ndarray)) and len(self.abnormal_channels) == 0:
                    optimizer = GeneticKMeans(visualize=True)
                    self.abnormal_channels = optimizer.optimize(
                        activation_tensor=reshaped, layer_name=layer_name, args=args
                    )

                normal_mask = torch.ones_like(max_values, dtype=torch.bool)
                normal_mask[self.abnormal_channels] = False
                cluster_max = max_values[normal_mask].max().item()

                tensor_clone = tensor.clone()
                abnormal_tensor = tensor_clone[:, self.abnormal_channels, :, :]

                quantized_tensor = quantizer(tensor, group_name="low", i=i, step=step, calib5=calib5,
                                             is_weight=False, threshold=cluster_max)
                quantized_abnormal = quantizer(abnormal_tensor, group_name="high", i=i, step=step,
                                               calib5=calib5, is_weight=False)
                quantized_tensor[:, self.abnormal_channels, :, :] = quantized_abnormal
            else:
                abnormal_tensor = tensor[:, self.abnormal_channels, :, :]
                quantized_tensor = quantizer(tensor, group_name="low", i=i, step=step,
                                             calib5=calib5, is_weight=False)
                quantized_abnormal = quantizer(abnormal_tensor, group_name="high", i=i, step=step,
                                               calib5=calib5, is_weight=False)
                quantized_tensor[:, self.abnormal_channels, :, :] = quantized_abnormal
        return quantized_tensor

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False,
                layer_name=None, params=None, save_stats=False, adjustment=False,
                num=None, a=False, num_bsz=-1, args=None):
        if params:
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)

        if self.use_input_quant:
            x = self.adjust_and_quantize(x, self.input_quantizer, i=i, step=step, calib5=calib5,
                                         is_weight=False, layer_name=layer_name, adjustment=adjustment, args=args)
        
        if self.use_weight_quant:
            w = self.adjust_and_quantize(self.weight, self.weight_quantizer, i=i, step=step, calib5=calib5,
                                         is_weight=True, layer_name=layer_name, adjustment=adjustment)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        
class QuantLinear_diff_outlier(nn.Linear):
    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}):
        super(QuantLinear_diff_outlier, self).__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer_group_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.abnormal_channels = []  # 只在 step == 0 时设置一次
        self.scaling_factors = { }

    def __repr__(self):
        s = super(QuantLinear_diff_outlier, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def adjust_and_quantize(self, tensor, quantizer, is_weight=False, i=None, step=None, calib5=False,layer_name = None, adjustment=False):
        """
        使用 K-Means 进行通道划分，并调整特定通道的值
        """
        if is_weight:
            tensor_clone = tensor.clone() 
            return quantizer(tensor_clone, i=i, step=step, calib5=calib5, is_weight=is_weight, adjustment=adjustment)
        else: # 激活
            if calib5:
                # original_shape = tensor.shape
                reshaped = tensor.view(-1, tensor.shape[-1])  # [..., N]
                max_values, _ = reshaped.max(dim=0)  # [N]
                activation_tensor = reshaped.clone()

                if isinstance(self.abnormal_channels, (list, np.ndarray)) and len(self.abnormal_channels) == 0: # 为空 表示未初始化
                        optimizer = GeneticKMeans(visualize=True)
                        self.abnormal_channels = optimizer.optimize(
                            activation_tensor=activation_tensor,
                            layer_name=layer_name
                        )

                # 计算正常通道的最大值
                normal_mask = torch.ones_like(max_values, dtype=torch.bool)
                normal_mask[self.abnormal_channels] = False
                cluster_max = max_values[normal_mask].max().item()

                # 将异常通道提取出来，以单独的指标进行量化，再复原
                tensor_clone = tensor.clone() 
                abnormal_tensor = tensor_clone[..., self.abnormal_channels]
                            
                quantized_tensor = quantizer(tensor, group_name="low", i=i, step=step, calib5=calib5, is_weight=is_weight, threshold = cluster_max)
                quantized_abnormal_tensor = quantizer(abnormal_tensor, group_name="high", i=i, step=step, calib5=calib5, is_weight=is_weight)
                quantized_tensor[..., self.abnormal_channels] = quantized_abnormal_tensor
            else:
                abnormal_tensor = tensor[..., self.abnormal_channels]
                quantized_tensor = quantizer(tensor, group_name="low", i=i, step=step, calib5=calib5, is_weight=is_weight)
                quantized_abnormal_tensor = quantizer(abnormal_tensor, group_name="high", i=i, step=step, calib5=calib5, is_weight=is_weight)
                quantized_tensor[..., self.abnormal_channels] = quantized_abnormal_tensor
        return quantized_tensor
    
    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1):
        """
        Perform forward pass using quantized or non-quantized weights, with channel-wise adjustment.
        """
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)

        if self.use_input_quant:
            x = self.adjust_and_quantize(x, self.input_quantizer, i=i, step=step, calib5=calib5, is_weight=False,layer_name = layer_name, adjustment=adjustment)
        
        if self.use_weight_quant:
            weight = self.adjust_and_quantize(self.weight, self.weight_quantizer, i=i, step=step, calib5=calib5, is_weight=True,layer_name = layer_name, adjustment=adjustment)
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)
 

# 直接的缩放
class QuantLinear_diff_scale(nn.Linear):
    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}):
        super(QuantLinear_diff_scale, self).__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.abnormal_channels = []  # 只在 step == 0 时设置一次
        # self.scaling_factors = [None]*6400     # 每一轮 forward 都 append 当前的缩放因子
        self.scaling_factors = { }

    def __repr__(self):
        s = super(QuantLinear_diff_scale, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def plot_activation_for_abnormal_channels(self, layer_name, activations, abnormal_channels, step, i, save_dir="/opt/data/private/GaoJing/deeplearnng/mar/plot/abnormal_channels_plot"):
        """
        绘制并保存异常通道的激活图。
        
        Parameters:
            layer_name (str): 层的名称。
            activations (Tensor): 当前层的激活值。
            abnormal_channels (list): 异常通道的索引。
            step (int): 当前步数。
            i (int): 当前i值，用于命名。
            save_dir (str): 图片保存的根目录。
        """
        
        # 获取异常通道的激活值
        abnormal_activations = activations[:, abnormal_channels]
        abnormal_activations = abnormal_activations.cpu()
        
        # 创建保存目录
        folder_name = f"{layer_name}_i_{i}"
        save_path = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 绘制每个异常通道的激活
        for idx, channel in enumerate(abnormal_channels):
            plt.figure(figsize=(10, 6))
            plt.plot(abnormal_activations[:, idx].cpu().detach().numpy())  # 绘制激活值
            plt.title(f"{layer_name} - Channel {channel} - Step {step} - i {i}")
            plt.xlabel('Position')
            plt.ylabel('Activation Value')
            plt.grid(True)
            
            # 保存图片
            plot_filename = f"{layer_name}_step_{step}_i_{i}_channel_{channel}.png"
            plt.savefig(os.path.join(save_path, plot_filename))
            plt.close()

    def adjust_and_quantize(self, tensor, quantizer, is_weight=False, i=None, step=None, calib5=False,layer_name = None, adjustment=False):
        """
        使用 K-Means 进行通道划分，并调整特定通道的值
        """
        if is_weight:
            # 如果已经嵌入缩放 或者 不处于校准阶段，进行正常量化
            tensor_clone = tensor.clone()
            return quantizer(tensor_clone, i=i, step=step, calib5=calib5, is_weight=is_weight, adjustment=adjustment)
        
        index = step*100+99-i
        original_shape = tensor.shape
        reshaped = tensor.view(-1, tensor.shape[-1])
        activation_tensor = reshaped.clone()


        if (not is_weight and calib5): # 激活的第一次校准
            # [N, 1536]
            max_values, _ = reshaped.max(dim=0)  # [1536]
            max_values = torch.clamp(max_values, min=0)
        #     # 原普通聚类方法的代码：
        #     if self.abnormal_channels is None:
        #         max_values_np = max_values.cpu().numpy().reshape(-1, 1)
        #         kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(max_values_np)
        #         labels = kmeans.labels_
        #         high_value_channels = (labels == kmeans.cluster_centers_.argmax())
        #         self.abnormal_channels = np.where(high_value_channels)[0].tolist()  # 记录异常通道索引

        #     if self.abnormal_channels:
        #         normal_mask = torch.ones_like(max_values, dtype=torch.bool)  # 先全设为 True
        #         normal_mask[self.abnormal_channels] = False  # 异常通道设为 False
        #         cluster_max = max_values[normal_mask].max().item()  # 计算正常通道的最大值
        #     else:
        #         cluster_max = max_values.max().item()  # 如果没有异常通道，取整体最大值

        #     scaling_factor = (cluster_max / max_values.max())
        #     # if scaling_factor == 1.0:
        #     #     self.abnormal_channels = None

        #     # if self.abnormal_channels is not None:
        #     if scaling_factor > 1.0:
        #         scaling_factor = 1
        #     self.scaling_factors[index] = scaling_factor

        #     # if not is_weight:
        #     #     logger3.info(f"Step {step} Timestep {i} {layer_name} activations: high_value_channels: {self.abnormal_channels}, scaling_factor: {scaling_factor}")
        #     # else:
        #     #     logger3.info(f"Step {step} Timestep {i} {layer_name} weights: {self.abnormal_channels}, scaling_factor: {scaling_factor}")

        # reshaped[:, self.abnormal_channels] *= self.scaling_factors[index]

        # 经过改良之后的配合遗传算法的k-means算法
            if isinstance(self.abnormal_channels, (list, np.ndarray)) and len(self.abnormal_channels) == 0:
                # max_values_np = max_values.cpu().numpy().reshape(-1, 1)
                # # 第一次聚类
                # best_k, results = find_optimal_k(max_values_np, reshaped, max_values_np)

                # if best_k == 2:
                #     # 情况1：直接使用首次聚类结果
                #     labels = results[best_k]['labels']
                #     logger4.info(f"Optimal number of clusters: {best_k} (直接使用)")
                    
                #     # 确定主簇（最大值最小的簇）
                #     cluster_max_values = [np.max(max_values_np[labels == i]) for i in range(best_k)]
                #     main_cluster = np.argmin(cluster_max_values)
                #     # 首次聚类可视化
                #     visualize_results(max_values_np, layer_name, results)
                #     # 记录异常通道（非主簇的所有通道）
                #     self.abnormal_channels = np.where(labels != main_cluster)[0].tolist()
                #     logger4.info(f"异常通道列表（共{len(self.abnormal_channels)}个）: {self.abnormal_channels}")
                    
                # else:
                #     # 情况2：对簇最大值进行二次聚类
                #     logger4.info(f"首次聚类得到{best_k}簇，进行二次聚类...")
                    
                #     # 提取每簇最大值
                #     first_labels = results[best_k]['labels']
                #     cluster_maxima = [np.max(max_values_np[first_labels == i]) for i in range(best_k)]
                #     cluster_maxima = np.array(cluster_maxima).reshape(-1, 1)

                    
                #     # 强制k=2的二次聚类
                #     kmeans = KMeans(n_clusters=2, random_state=42).fit(cluster_maxima)
                #     second_labels = kmeans.labels_
                #     final_centers = kmeans.cluster_centers_.flatten()  # 二次聚类中心
                    
                #     # 确定主簇（最大值较小的簇）
                #     if np.mean(cluster_maxima[second_labels == 0]) < np.mean(cluster_maxima[second_labels == 1]):
                #         main_cluster_group = 0
                #     else:
                #         main_cluster_group = 1
                    
                #     # 映射回原始标签
                #     main_clusters = [i for i in range(best_k) if second_labels[i] == main_cluster_group]
                #     final_labels = np.zeros_like(first_labels)
                    
                #     for new_label, orig_label in enumerate(range(best_k)):
                #         mask = (first_labels == orig_label)
                #         if orig_label in main_clusters:
                #             final_labels[mask] = 0  # 主簇
                #         else:
                #             final_labels[mask] = 1  # 异常簇
                    
                #     main_cluster = 0  # 主簇标签已重新映射为0
                #     logger4.info(f"二次聚类完成，主簇包含{len(main_clusters)}个子簇")
                #     visualize_results(max_values_np, layer_name, results)
                #     # second_results = {
                #     #     'original_k': best_k,
                #     #     'second_pass_labels': final_labels,
                #     #     'second_pass_centers': final_centers,
                #     #     'main_cluster': main_cluster
                #     # }
                #     # visualize_results(max_values_np, layer_name, second_results, is_second_pass=True)

                #     # 记录异常通道（非主簇的所有通道）
                #     self.abnormal_channels = np.where(final_labels != main_cluster)[0].tolist()
                #     logger4.info(f"异常通道列表（共{len(self.abnormal_channels)}个）: {self.abnormal_channels}")
                
                optimizer = GeneticKMeans(visualize=True)
                self.abnormal_channels = optimizer.optimize(
                    activation_tensor=activation_tensor,
                    layer_name=layer_name
                )
                
                # if layer_name == "res_block11_mlp[2]":
                #     self.abnormal_channels = [211, 445, 818]

                # 计算正常通道的最大值
                normal_mask = torch.ones_like(max_values, dtype=torch.bool)
                normal_mask[self.abnormal_channels] = False
                cluster_max = max_values[normal_mask].max().item()
                    
                # 初始化缩放因子存储结构（字典：通道索引 -> 缩放因子列表）
                if not hasattr(self, 'scaling_factors_dict'):
                    self.scaling_factors_dict = {}
                    
                # 为每个异常通道单独计算并存储缩放因子
                for ch in self.abnormal_channels:
                        # 计算该通道的缩放因子（与正常通道最大值的比值）
                        scaling_factor = cluster_max / max_values[ch].item() if max_values[ch].item() != 0 else 1.0
                        
                        # 限制缩放因子不超过1
                        if scaling_factor > 1.0:
                            scaling_factor = 1.0
                        
                        # 存储缩放因子（初始化该通道的列表或追加新值）
                        if ch not in self.scaling_factors_dict:
                            self.scaling_factors_dict[ch] = []
                        self.scaling_factors_dict[ch].append(scaling_factor)
                        
                        # 应用最新的缩放因子
                        reshaped[:, ch] *= scaling_factor
            else:
                # 已有异常通道记录的情况
                normal_mask = torch.ones_like(max_values, dtype=torch.bool)
                normal_mask[self.abnormal_channels] = False
                cluster_max = max_values[normal_mask].max().item()
                
                # 为每个异常通道更新缩放因子
                for ch in self.abnormal_channels:
                    scaling_factor = cluster_max / max_values[ch].item() if max_values[ch].item() != 0 else 1.0
                    
                    if scaling_factor > 1.0:
                        scaling_factor = 1.0
                    
                    # 确保该通道的记录存在
                    if ch not in self.scaling_factors_dict:
                        self.scaling_factors_dict[ch] = []
                    
                    # 存储并应用最新的缩放因子
                    self.scaling_factors_dict[ch].append(scaling_factor)
                    reshaped[:, ch] *= scaling_factor
        else:
            # # 保存200套
            for ch in self.abnormal_channels:
            #     if index < 3200:
            #             # 前3200数据：按 %100 分组，使用前100组
            #             group_idx = index % 100
            #             scaling_factor = self.scaling_factors_dict[ch][group_idx]
            #     else:
            #             # 后3200数据：按 %100 分组，使用后100组
            #             group_idx = 100 + (index % 100)
            #             scaling_factor = self.scaling_factors_dict[ch][group_idx]
            #     reshaped[:, ch] *= scaling_factor

            # # 保存400套
            # for ch in self.abnormal_channels:
            #         segment_size = 1600  # 每段1600个数据
            #         segment_num = index // segment_size  # 确定属于哪个分段（0-3）
            #         group_idx = (segment_num * 100) + (index % 100)  # 计算全局组索引
                    
            #         # 安全获取
            #         if group_idx < len(self.scaling_factors_dict[ch]):
            #             scaling_factor = self.scaling_factors_dict[ch][group_idx]
            #         else:
            #             scaling_factor = 1.0  # 默认值
            #             # logger4.warning(f"Channel {ch}: Index {group_idx} out of range, using default scaling factor")

            #         reshaped[:, ch] *= scaling_factor

                reshaped[:, ch] *= self.scaling_factors_dict[ch][0]
                # [int((index))]
                # / 10 --640套
                # % 100 --100套
                # % 100 奇数不动，偶数再除二  --50套
                # % 100 再 / 10 --10套

        quantized_tensor = quantizer(reshaped, i=i, step=step, calib5=calib5, is_weight=is_weight)

        # quantized_tensor[:, self.abnormal_channels] /= scaling_factor

        restored_tensor = quantized_tensor.view(original_shape)

        return restored_tensor
    
    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1):
        """
        Perform forward pass using quantized or non-quantized weights, with channel-wise adjustment.
        """
        # if step ==0 and i == 99:
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)
        index = step*100+99-i
        if self.use_input_quant:
            x = self.adjust_and_quantize(x, self.input_quantizer, i=i, step=step, calib5=calib5, is_weight=False,layer_name = layer_name, adjustment=adjustment)
        
        if self.use_weight_quant:
            weight_clone = self.weight.clone()
            if calib5:
                for ch in self.abnormal_channels:
                    scaling_factor = self.scaling_factors_dict[ch][index]
                    # logger3.info(f"scaling_factor of step {step} timestep {i}: {scaling_factor}")
                    weight_clone[:, ch] /= scaling_factor
            # else:
            #     # for ch in self.abnormal_channels:
            #     #     # scaling_factor = self.scaling_factors_dict[ch][int((index)/20)]
            #     #     if index < 3200:
            #     #         # 前3200数据：按 %100 分组，使用前100组
            #     #         group_idx = index % 100
            #     #         scaling_factor = self.scaling_factors_dict[ch][group_idx]
            #     #     else:
            #     #         # 后3200数据：按 %100 分组，使用后100组
            #     #         group_idx = 100 + (index % 100)
            #     #         scaling_factor = self.scaling_factors_dict[ch][group_idx]

            #     # 保存400套

            #     # for ch in self.abnormal_channels:
            #     #     segment_size = 1600  # 每段1600个数据
            #     #     segment_num = index // segment_size  # 确定属于哪个分段（0-3）
            #     #     group_idx = (segment_num * 100) + (index % 100)  # 计算全局组索引
                    
            #     #     # 安全获取
            #     #     if group_idx < len(self.scaling_factors_dict[ch]):
            #     #         scaling_factor = self.scaling_factors_dict[ch][group_idx]
            #     #     else:
            #     #         scaling_factor = 1.0  # 默认值
            #     #         # logger4.warning(f"Channel {ch}: Index {group_idx} out of range, using default scaling factor")

            #         # logger3.info(f"scaling_factor of step {step} timestep {i}: {scaling_factor}")
            #         weight_clone[:, ch] /= scaling_factor

            weight = self.adjust_and_quantize(weight_clone, self.weight_quantizer, i=i, step=step, calib5=calib5, is_weight=True, layer_name = layer_name, adjustment=adjustment)
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)
   
# 阈值截断方法
class QuantLinear_diff_scale2(nn.Linear):
    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}):
        super(QuantLinear_diff_scale2, self).__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.abnormal_channels = None   # 只在 step == 0 时设置一次
        self.scaling_factors = [None] * 6400    # 每一轮 forward 都 append 当前的缩放因子

    def __repr__(self):
        s = super(QuantLinear_diff_scale2, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def plot_activation_for_abnormal_channels(self, layer_name, activations, abnormal_channels, step, i, save_dir="/opt/data/private/GaoJing/deeplearnng/mar/plot/abnormal_channels_plot"):
        """
        绘制并保存异常通道的激活图。
        
        Parameters:
            layer_name (str): 层的名称。
            activations (Tensor): 当前层的激活值。
            abnormal_channels (list): 异常通道的索引。
            step (int): 当前步数。
            i (int): 当前i值，用于命名。
            save_dir (str): 图片保存的根目录。
        """
        
        # 获取异常通道的激活值
        abnormal_activations = activations[:, abnormal_channels]
        abnormal_activations = abnormal_activations.cpu()
        
        # 创建保存目录
        folder_name = f"{layer_name}_i_{i}"
        save_path = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 绘制每个异常通道的激活
        for idx, channel in enumerate(abnormal_channels):
            plt.figure(figsize=(10, 6))
            plt.plot(abnormal_activations[:, idx].cpu().detach().numpy())  # 绘制激活值
            plt.title(f"{layer_name} - Channel {channel} - Step {step} - i {i}")
            plt.xlabel('Position')
            plt.ylabel('Activation Value')
            plt.grid(True)
            
            # 保存图片
            plot_filename = f"{layer_name}_step_{step}_i_{i}_channel_{channel}.png"
            plt.savefig(os.path.join(save_path, plot_filename))
            plt.close()

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False,
                layer_name=None, params=None, save_stats=False, adjustment=False,
                num=None, a=False, num_bsz=-1):

        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")
        index = step * 100 + i
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)
            # self.input_quantizer.channel_wise = params.get('channel_wise', self.input_quantizer.channel_wise)

        if self.use_input_quant:
            x_clone = x.clone()
            original_shape = x_clone.shape
            reshaped_x = x_clone.view(-1, x.shape[-1])  # [N, C]
            if calib5:
                max_values, _ = reshaped_x.max(dim=0)  # 每个通道最大值
                if step == 0 and self.abnormal_channels is None:
                    # KMeans 聚类找出异常通道
                    max_np = max_values.cpu().numpy().reshape(-1, 1)
                    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(max_np)
                    labels = kmeans.labels_
                    high_cluster = kmeans.cluster_centers_.argmax()
                    self.abnormal_channels = np.where(labels == high_cluster)[0].tolist()

                if self.abnormal_channels:
                    mask = torch.ones_like(max_values, dtype=torch.bool)
                    mask[self.abnormal_channels] = False
                    cluster_max = max_values[mask].max().item() if mask.any() else max_values.max().item() # 正常通道最大值
                    full_max = max_values.max().item()  # 异常通道最大值
                else:
                    cluster_max = max_values.max().item()
                    full_max = cluster_max

                # self.plot_activation_for_abnormal_channels(layer_name, x, self.abnormal_channels, step, i)

                # logger3.info(f"cluster_max: {cluster_max} full_max：{full_max}")
                abnormal_data = reshaped_x[:, self.abnormal_channels]  # [N, len(abnormal_channels)]
                abnormal_mean = abnormal_data.mean().item()

                scale_factor = (cluster_max / abnormal_mean)

                # scale_factor = (cluster_max / full_max)
                # if scale_factor < 1.0:
                #     for ch in self.abnormal_channels:
                #         ch_max = reshaped_x[:, ch].mean()
                #         reshaped_x[:, ch] = ch_max


                # scale_factor = (cluster_max / full_max)
                if scale_factor > 1.0:
                    scale_factor = 1.0
                self.scaling_factors[index] = scale_factor
                # self.scaling_factors.append(scale_factor)
                # x_max_before_quant = x.max()
                # x_min_before_quant = x.min()

                quantized_x = self.input_quantizer(reshaped_x, i=i, step=step, calib5=calib5,
                                     is_weight=False, adjustment=adjustment,threshold = cluster_max)
                # quantized_x = self.input_quantizer(reshaped_x, i=i, step=step, calib5=calib5,
                #                      is_weight=False, adjustment=adjustment)

                restored_x = quantized_x.view(original_shape)
                # restored_x = reshaped_x.view(original_shape)

            else:
                quantized_x = self.input_quantizer(reshaped_x, i=i, step=step, calib5=calib5,
                                     is_weight=False, adjustment=adjustment)
                restored_x = quantized_x.view(original_shape)
        else:
            restored_x = x


        weight = self.weight.clone()
        if self.use_weight_quant:
            # if calib5:
                if self.abnormal_channels and self.scaling_factors:
                    cur_factor = self.scaling_factors[index]
                    if cur_factor < 1.0:
                        # weight[self.abnormal_channels,:] /= cur_factor
                        weight[:,self.abnormal_channels] /= cur_factor     

                weight = self.weight_quantizer(weight, i=i, step=step, calib5=calib5,
                                           is_weight=True, adjustment=adjustment)

        return F.linear(restored_x, weight, self.bias)

class QuantLinear_scaling(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 i = None):
        super(QuantLinear_scaling, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_group_scaling(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_input_quant = False
        self.use_weight_quant = False

        self.initial_zs = -5.545
        self.initial_ss = torch.zeros(6400, device=self.device)
        # self.adjustment_factor = torch.zeros(6400, device=self.device)
        self.threshold_input = torch.zeros(6400, device=self.device)
        self.threshold_weight = torch.zeros(6400, device=self.device)

        self.i = None
        

    def __repr__(self):
        s = super(QuantLinear_scaling, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def get_iqr_threshold(self, values):
        optimal_threshold = find_optimal_threshold(values)
        return torch.tensor(optimal_threshold, dtype=torch.float32, device=values.device)
 
    def group_values(self, values, threshold,step = None):
        low_mask = values <= threshold
        return low_mask  # 返回low_mask，保持原有行为
    
    def quantize_weight_group(self, values, mask, i, step, calib5, adjustment, group_name="low",threshold = None):
        mask = mask.bool()
        masked_values = values.clone()
        masked_values[~mask] = threshold
        quantized = self.weight_quantizer(masked_values, i=i, step=step, calib5=calib5, is_weight=True, group_name=group_name)
        quantized[~mask] = values[~mask]  # 对不需要量化的部分恢复原始值
        return quantized

    def quantize_input_group(self, values, mask, i, step, calib5, adjustment, group_name="low",threshold = None):
     
        mask = mask.bool()
        masked_values = values.clone()
        masked_values[~mask] = threshold
        quantized = self.input_quantizer(masked_values, i=i, step=step, calib5=calib5, is_weight=False, group_name=group_name)
        quantized[~mask] = values[~mask]  # 对不需要量化的部分恢复原始值
        return quantized

    def quantize_input_separated(self, values, threshold, i, step, calib5, adjustment):
        """
        Perform quantization for the full input by splitting the input into two parts 
        and quantizing them separately based on the threshold.
        """
        values_abs = values.abs()
        # min = values_abs.min()
        # max = values_abs.max()
    
        if threshold >= values.max():
            low_input_quantized = self.input_quantizer(values, group_name="low", 
                                                        i=i, step=step, calib5=calib5, adjustment=adjustment)
        else:
            # 切分矩阵：低值组（小于阈值的部分），高值组（大于阈值的部分）
            # low_values = values.clone()  # 保持原数据的拷贝
            high_values = values.clone()

            # 将大于阈值的部分放到high_values
            values[values_abs > threshold] = threshold  # 将超过阈值的部分裁剪
            # min = values.min()
            # max = values.max()

            high_values[values_abs <= threshold] = threshold  # 将小于阈值的部分裁剪为阈值
            # min = high_values.min()
            # max = high_values.max()
            # sign_mask = torch.sign(high_values) 
            # high_values_abs = high_values.abs()
            # min = high_values_abs.min()
            # max = high_values_abs.max()

            # 分别对低值组和高值组进行量化
            low_input_quantized = self.input_quantizer(values, group_name="low", 
                                                        i=i, step=step, calib5=calib5, adjustment=adjustment)
            # min = low_input_quantized.min()
            # max = low_input_quantized.max()
            high_input_quantized = self.input_quantizer(high_values, group_name="high", 
                                                            i=i, step=step, calib5=calib5, adjustment=adjustment)
            # min = high_input_quantized.min()
            # max = high_input_quantized.max()
            # high_input_quantized = high_input_quantized*sign_mask
            # min = high_input_quantized.min()
            # max = high_input_quantized.max()

            # 将量化后的低值组和高值组拼接回原矩阵
            # quantized_values = low_input_quantized.clone()  # 使用低值量化结果作为基础
            low_input_quantized[high_values != threshold] = high_input_quantized[high_values != threshold]  # 恢复高值部分

        return low_input_quantized
    
    def quantize_weight_separated(self, values, threshold, i, step, calib5, adjustment):
        """
        Perform quantization for the full input by splitting the input into two parts 
        and quantizing them separately based on the threshold.
        """
        values_abs = values.abs()
        # min = values_abs.min()
        # max = values_abs.max()
    
        if threshold >= values.max():
            low_weight_quantized = self.weight_quantizer(values, group_name="low", 
                                                        i=i, step=step, calib5=calib5, adjustment=adjustment, is_weight=True)
        else:
            # 切分矩阵：低值组（小于阈值的部分），高值组（大于阈值的部分）
            low_values = values.clone()  # 保持原数据的拷贝
            high_values = values.clone()

            # 将大于阈值的部分放到high_values
            low_values[values_abs > threshold] = threshold  # 将超过阈值的部分裁剪
            # min = low_values.min()
            # max = low_values.max()

            high_values[values_abs <= threshold] = threshold  # 将小于阈值的部分裁剪为阈值
            # min = high_values.min()
            # max = high_values.max()
            sign_mask = torch.sign(high_values) 
            high_values_abs = high_values.abs()

            # 分别对低值组和高值组进行量化
            low_weight_quantized = self.weight_quantizer(low_values, group_name="low", 
                                                        i=i, step=step, calib5=calib5, adjustment=adjustment, is_weight=True)
            # min = low_weight_quantized.min()
            # max = low_weight_quantized.max()
            high_weight_quantized = self.weight_quantizer(high_values_abs, group_name="high", 
                                                            i=i, step=step, calib5=calib5, adjustment=adjustment, is_weight=True)
            # min = high_weight_quantized.min()
            # max = high_weight_quantized.max()
            high_weight_quantized = high_weight_quantized*sign_mask
            # min = high_weight_quantized.min()
            # max = high_weight_quantized.max()

            # 将量化后的低值组和高值组拼接回原矩阵
            # quantized_values = low_input_quantized.clone()  # 使用低值量化结果作为基础
            low_weight_quantized[high_values != threshold] = high_weight_quantized[high_values != threshold]  # 恢复高值部分
        return low_weight_quantized
    
    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1,sign_scaling = False,scale_quant = None,shift_quant = None):
        """
        Perform forward pass using quantized or non-quantized weights, and save statistics/images to separate directories.
        """
        # if step ==0 and i == 98:
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}")
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)
            # self.input_quantizer.channel_wise = params.get('channel_wise', self.input_quantizer.channel_wise)
        self.i = i
        if self.use_input_quant:
            # if (layer_name == "input_proj" or layer_name == "final_linear") and adjustment:
            #         calib5 = False
            #         min = x.min()
            #         max = x.max()
            #         if (max>self.initial_zs) or (min<-self.initial_zs) :
            #             sign_scaling = True
            #             self.initial_ss[step * 100 + i] = (max-min)/((-self.initial_zs)*2)
            #             logger3.info(f"adjustment_factor: {self.initial_ss[step*100+i]} zero_point: {self.initial_zs}")
            #             x = (x  - min) / self.initial_ss[step * 100 + i] + self.initial_zs
                   
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir2)
            
            x_quant = self.input_quantizer(x, i=i, step=step, calib5=calib5,adjustment = adjustment,layer_name = layer_name,scale_quant = scale_quant,shift_quant = shift_quant)

            # if (layer_name == "input_proj" or layer_name == "final_linear") and adjustment:
            #     calib5 = calib5
            #     if sign_scaling:
            #         x_quant = ( x_quant - self.initial_zs ) * self.initial_ss[step * 100 + i] + min
            
            if draw:
                save_and_plot(x, f"Quantized Activation Of Step{step} Timestep{i} {layer_name}", f"quantized_activation_of_step{step}_time{i}_{layer_name}.png", output_dir3)
            
        else:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir1)
        
        if self.use_weight_quant:
            if draw:
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir5)
            w = self.weight_quantizer(self.weight, i=i, step=step, calib5=calib5, is_weight=True)

            if draw:
                save_and_plot(w, f"Quantized Weights Of Step{step} Timestep{i} {layer_name}", f"quantized_weights_of_step{step}_time{i}_{layer_name}.png", output_dir6)
            
        else:
            # 不进行量化，直接使用原始权重
            w = self.weight
            if draw:
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir4)
            
        if save_stats:
                # if calib5:
                #     layer_name = layer_name + "_calibration"
                # else:
                #     layer_name = layer_name + f"_bsz_{num_bsz}"
                # if step == 0 and (i == -1 or i in [99]) and (num == 0 or num == None):
                # # if step in [0, 1, 63] and (i == -1 or i in [99, 98, 97]) and (num == 0 or num == None):
                    save_quantization_stats(
                            smoothed_x=x,
                            smoothed_weight=self.weight,
                            # smoothed_x_quant=x_quant,
                            # smoothed_weight_quant=w,
                            step=step,
                            i = i,
                            layer_name=layer_name,
                            dim_ac = 2,
                            dim = 1
                        )
                
                    save_plots(smoothed_x=x,
                            smoothed_weight=self.weight,
                            # smoothed_x_quant=x_quant,
                            # smoothed_weight_quant=w,
                            step=step,
                            i = i,
                            layer_name=layer_name,
                            dim_ac = 2,
                            dim = 1
                            )

        # 执行线性变换
        if self.use_input_quant:
            out = F.linear(x_quant, weight=w, bias=self.bias)
            # logger3.info(f"Output Of Step{step} Timestep{i} {layer_name}: min={out.min()}, max={out.max()}")

        else:
            out = F.linear(x, weight=w, bias=self.bias)
        return out

class QuantLinear_scale_channels(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 i = None):
        super(QuantLinear_scale_channels, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_scale_channels(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_input_quant = False
        self.use_weight_quant = False
        self.threshold = 5.545
        self.abnormal_channels = None
        self.count = 0 

        self.recorded_x_values = []     # 用于保存 step==0 且 i in 0~99 时的 x
        self.recording_enabled = False  # 控制是否记录
        self.save_dir = "/opt/data/private/GaoJing/mar_1/visualizations/boxplot_output"
        os.makedirs(self.save_dir, exist_ok=True)
        self.recorded_x_with_index = []

        self.i = None
        

    def __repr__(self):
        s = super(QuantLinear_scale_channels, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant
    
    def enable_recording(self):
        self.recording_enabled = True
        self.recorded_x_values = []


    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1,sign_scaling = False,scale_quant = None,shift_quant = None,args=None):
        """
        Perform forward pass using quantized or non-quantized weights, and save statistics/images to separate directories.
        """
    
        if params:
        #     # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)
        #     self.input_quantizer.channel_wise = params.get('channel_wise', self.input_quantizer.channel_wise)
        self.i = i
        self.threshold = args.Gaussian_bound
        
        if self.use_input_quant: 
            if x.abs().max() >= self.threshold:
                self.count += 1
                if i==0 and step == 63:
                    logger4.info(f"{layer_name}, when threhold is set as {self.threshold}, SR is triggered {self.count} times.")
                if len(x.shape) < 2:
                    raise ValueError("Input tensor must have at least 2 dimensions")
                channels_dim = -1  # 最后一个维度作为通道维度
                # 3. 计算每个通道的最大值
                if len(x.shape) == 2:
                    channel_max = x.abs().max(dim=0)[0]
                    original_shape = x.shape
                    merged_shape = (-1, original_shape[channels_dim])
                    merged_tensor = x.clone()
                else:
                    original_shape = x.shape
                    merged_shape = (-1, original_shape[channels_dim])
                    merged_tensor = x.reshape(merged_shape)
                    channel_max = merged_tensor.abs().max(dim=0)[0]
                self.abnormal_channels = torch.where(channel_max >= self.threshold)[0].tolist()
                # logger3.info(f"step: {step}, timestep: {i}")
                # logger3.info(f"Found {len(self.abnormal_channels)} abnormal channels (max >= {self.threshold}):")
                # logger3.info(f"Abnormal channels: {self.abnormal_channels}")

                # normal_mask = torch.ones_like(channel_max, dtype=torch.bool)
                # normal_mask[self.abnormal_channels] = False   
                # if normal_mask.sum() == 0:
                #     normal_max = self.threshold
                # else:
                #     normal_max = channel_max[normal_mask].max().item()
                abnormal_max = channel_max.max().item()

                scale_ratio = (self.threshold / abnormal_max)
                # logger3.info(f"Abnormal max: {abnormal_max}, Scale ratio: {scale_ratio}, min:{x.min()}")
                self.abnormal_channels = torch.tensor(self.abnormal_channels, device=x.device)
                # channel_data = merged_tensor[..., 1]
                # logger3.info(f"Abnormal Channel (index=1): {channel_data}")

                # scale_tensor = torch.ones_like(merged_tensor, device=x.device)
                # scale_tensor[..., self.abnormal_channels] = scale_tensor[..., self.abnormal_channels] * scale_ratio
                # merged_tensor *= scale_tensor
                # logger3.info(f"before scaling, min:{merged_tensor.min()} max:{merged_tensor.max()}")
                merged_tensor[..., self.abnormal_channels] *= scale_ratio
                # channel_data = merged_tensor[..., 1]
                # logger3.info(f"Abnormal Channel (index=1): {channel_data}")
                # logger3.info(f"after scaling, min:{merged_tensor.min()} max:{merged_tensor.max()}")
                x_quant = self.input_quantizer(merged_tensor, i=i, step=step, calib5=calib5,adjustment = adjustment)  
                # logger3.info(f"after quant, min:{x_quant.min()} max:{x_quant.max()}")  
                # channel_data = x_quant[..., 1]
                # logger3.info(f"Abnormal Channel (index=1): {channel_data}") 
                x_quant[..., self.abnormal_channels] /= scale_ratio   
                # channel_data = x_quant[..., 1]
                # logger3.info(f"Abnormal Channel (index=1,): {channel_data}")
                # x_quant /= scale_tensor
                # logger3.info(f"after rescaling, min:{x_quant.min()} max:{x_quant.max()}")
                x_quant = x_quant.view(original_shape)

            else:   
                # 如果是H模型，就做一下全动态？
                x_quant = self.input_quantizer(x, i=i, step=step, calib5=calib5,adjustment = adjustment)            
        
        if self.use_weight_quant:
            weight_clone = self.weight
            w = self.weight_quantizer(weight_clone, i=i, step=step, calib5=calib5, is_weight=True)            
        else:
            w = self.weight
        if self.use_input_quant:
            out = F.linear(x_quant, weight=w, bias=self.bias)
            # logger3.info(f"Output Of Step{step} Timestep{i} {layer_name}: min={out.min()}, max={out.max()}")

        else:
            out = F.linear(x, weight=w, bias=self.bias)
        return out

class QuantLinearWithGrouping(nn.Linear):
    """
    Class to quantize weights and activations of given Linear layer using grouping based on IQR (Interquartile Range)
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 i=None, device='cuda'):
        super(QuantLinearWithGrouping, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_group(**input_quant_params)
        # self.input_quantizer = UniformQuantizer_ar(**weight_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.i = None
        self.threshold_input = torch.zeros(64, device=device)
        # self.threshold_weight = torch.zeros(64, device=device)
        # self.lut_table = [None]*64
        # self.saved_coords = None
        # self.values_shape_1 = torch.zeros(64, device=device)
        # self.values_valid_max_shape_1 = torch.zeros(64, device=device)
        # self.saved_coords_dict = {}  # 用于存储坐标哈希表
        # self.channel_indices_input = [None]
        # self.channel_indices_input1 = [None]
        # self.channel_indices_input_init = False

    def __repr__(self):
        s = super(QuantLinearWithGrouping, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def get_iqr_threshold(self, values,by_channel = False, i=None, step=None, is_weight=False):
        """
        Calculate the IQR threshold for grouping the values based on the Interquartile Range.
        """
        # flat_values = values.reshape(-1).cpu().detach().numpy()
        # q99_9 = np.percentile(flat_values,99.9)
        # min = values.min()
        # max = values.max()
        # if by_channel:
        #     optimal_threshold,channel_indices = find_optimal_threshold(values,by_channel)
        # else:
        optimal_threshold = find_optimal_threshold(values)
        # if not is_weight and not self.channel_indices_input_init:
        #     self.channel_indices_input = channel_indices
        #     self.channel_indices_input_init = True
        # min = values.min()
        # max = values.max()
        return torch.tensor(optimal_threshold, dtype=torch.float32, device=values.device)

    def quantize_input_separated(self, values, threshold, i, step, calib5, adjustment):
        """
        Perform quantization for the full input by splitting the input into two parts 
        and quantizing them separately based on the threshold.
        """
        # batch_size = values.size(0)
        high_values = values.clone()
        
        if threshold >= values.max():
            low_input_quantized = self.input_quantizer(values, group_name="low", 
                                                        i=i, step=step, calib5=calib5, adjustment=adjustment,threshold = threshold)
        else:
            if calib5:
                # 直接使用保存的位置信息
                # self.values_shape_1[step] = values.size(1)
                mask = high_values > threshold
                # indices = self.get_threshold_exceeding_indices(high_values, threshold, logger3)

                self.high_value_coords = torch.nonzero(mask)  # 获取大于阈值的所有位置坐标，并保存到列表
                high_values_to_quantize = high_values[self.high_value_coords[:, 0], self.high_value_coords[:, 1], self.high_value_coords[:, 2]]
            else:
                # if batch_size == 64:

                #     # high_values_to_quantize = []
                #     # for coord in self.saved_coords:
                #     #     idx = self.saved_coords_dict.get(tuple(coord.tolist()))
                #     #     if idx is not None:
                #     #         high_values_to_quantize.append(values[coord[0], coord[1], coord[2]])
                #     # high_values_to_quantize = torch.stack(high_values_to_quantize)
                #     high_values_to_quantize = values[self.saved_coords[:, 0], self.saved_coords[:, 1], self.saved_coords[:, 2]]
                # else:
                    mask = high_values > threshold
                    self.high_value_coords = torch.nonzero(mask)
                    high_values_to_quantize = high_values[self.high_value_coords[:, 0], self.high_value_coords[:, 1], self.high_value_coords[:, 2]]

            # 分别对低值组和高值组进行量化
            low_input_quantized = self.input_quantizer(values, group_name="low", 
                                                        i=i, step=step, calib5=calib5, adjustment=adjustment,threshold = threshold)
            high_input_quantized = self.input_quantizer(high_values_to_quantize, group_name="high", 
                                                            i=i, step=step, calib5=calib5, adjustment=adjustment,threshold = threshold)
            # if calib5 or (not calib5 and batch_size != 64):
            low_input_quantized[self.high_value_coords[:, 0], self.high_value_coords[:, 1], self.high_value_coords[:, 2]] = high_input_quantized
            # else:
                # 对于low_input_quantized，使用相同的有效坐标来更新值
            # low_input_quantized[self.saved_coords[:, 0], self.saved_coords[:, 1], self.saved_coords[:, 2]] = high_input_quantized

        return low_input_quantized

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats=False, adjustment=False, num=None, a=False, num_bsz=-1):
        """
        Perform forward pass using quantized or non-quantized weights and activations, and save statistics/images to separate directories.
        """
        self.i = i
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name} x: min={x.min()}, max={x.max()}, mean={x.mean()}")
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name} weight: min={self.weight.min()}, max={self.weight.max()}, mean={self.weight.mean()}")
        # logger3.info(f"输入激活x的形状为：{x.shape}")
        # save_sorted_activations_and_weights("fc2",x, self.weight, outpath)

        if self.use_input_quant:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir2)
            
            if torch.all(torch.eq(x, 0)):  # If all elements are zero
                x_quantized = x
            else:

                # 改进1
                # sign_mask = torch.sign(x) 
                # x_abs = x.abs()

                if calib5:
                    # 对输入激活进行分组量化
                    self.threshold_input[step] = self.get_iqr_threshold(x)
                    logger3.info(f"激活分界阈值为：{self.threshold_input[step]}")

                    # mask = x > self.threshold_input[step]
                    # indices = torch.nonzero(mask)  # Get indices of elements > threshold
                    # second_dim_indices = indices[:, 1].tolist()
                    # self.channel_indices_input1 = sorted(list(set(second_dim_indices)))


                    # compute_threshold_statistics(x,self.threshold_input[step],layer_name)
                    # x_quantized = self.calibrate_and_save_lut(x, self.threshold_input[step], i=i, step=step, calib5=calib5, adjustment=adjustment)
                # else:
                #     x_quantized = self.evaluate_with_lut(x, self.threshold_input[step], i=i, step=step, calib5=calib5, adjustment=adjustment)
                #     self.threshold_input[step] = self.threshold_input[step].to(torch.float16)

                # low_input_mask= self.group_values(x_abs, self.threshold_input[step])
                # x_min = x_abs.min()
                # x_max = x_abs.max()
                # if step ==0:
                #     false_indices = torch.nonzero(low_input_mask == False)
                #     logger3.info(f"False values found at indices: {false_indices.tolist()}")
                # x_quantized = self.quantize_input_separated(x_abs, self.threshold_input[step], i, step, calib5, adjustment)* sign_mask

                x_quantized = self.quantize_input_separated(x, self.threshold_input[step], i, step, calib5, adjustment)
                # x_quantized = self.input_quantizer(x, i=i, step=step, calib5=calib5,adjustment = adjustment)

                # x_quantized = self.quantize_with_split_optimized(x_abs, self.threshold_input[step], i, step, calib5, adjustment)* sign_mask
                
                # if torch.all(low_input_mask): # 高值组不需要量化了
                #     x_quantized = self.quantize_input_group(x_abs, low_input_mask, i, step, calib5,adjustment,group_name="low",threshold = self.threshold_input[step])*sign_mask
                # else:
                #     low_input_quantized = self.quantize_input_group(x_abs, low_input_mask, i, step, calib5, adjustment, group_name="low",threshold = self.threshold_input[step])
                #     x_quantized = self.quantize_input_group(low_input_quantized, ~low_input_mask, i, step, calib5, adjustment, group_name="high",threshold = self.threshold_input[step])* sign_mask # 改进2

                if draw:
                    save_and_plot(x_quantized, f"Quantized Activation Of Step{step} Timestep{i} {layer_name}", f"quantized_activation_of_step{step}_time{i}_{layer_name}.png", output_dir3)
        else:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir1)
            x_quantized = x
        torch.cuda.empty_cache()

        if self.use_weight_quant:
            quantized_weights = self.weight_quantizer(self.weight, i=i, step=step, calib5=calib5, is_weight=True)
            # # 改进
            # # sign_mask = torch.sign(self.weight) 
            # # min = self.weight.min()
            # # max = self.weight.max()
            # # weight_abs = self.weight.abs()
            # if calib5:
            #     # 计算权重的 IQR 阈值并进行分组
            #     self.threshold_weight[step] = self.get_iqr_threshold(self.weight)
            #     logger3.info(f"权重分界阈值为：{self.threshold_weight[step]}")
            # # else:
            # #     self.threshold_weight[step] = self.threshold_weight[step].to(torch.float16)
            # # quantized_weights = self.quantize_weight_separated(weight_abs, self.threshold_weight[step], i, step, calib5, adjustment)* sign_mask
            # quantized_weights = self.quantize_weight_separated(self.weight, self.threshold_weight[step], i, step, calib5, adjustment)
            # # min = self.weight.min()
            # # max = self.weight.max()
            # # low_weight_mask = self.group_values(weight_abs, self.threshold_weight[step])
            
            # # if torch.all(low_weight_mask): # 高值组不需要量化了
            # #     quantized_weights = self.quantize_weight_group(weight_abs, low_weight_mask, i, step, calib5,adjustment,group_name="low",threshold = self.threshold_weight[step])*sign_mask
            # # else:
            # #     low_weight_quantized = self.quantize_weight_group(weight_abs, low_weight_mask, i, step, calib5, adjustment, group_name="low",threshold = self.threshold_weight[step])
            # #     quantized_weights = self.quantize_weight_group(low_weight_quantized, ~low_weight_mask, i, step, calib5, adjustment, group_name="high",threshold = self.threshold_weight[step])*sign_mas



            if draw:
                save_and_plot(quantized_weights, f"Quantized Weights Of Step{step} Timestep{i} {layer_name}", f"quantized_weights_of_step{step}_time{i}_{layer_name}.png", output_dir6)
        else:
            quantized_weights = self.weight
            if draw:
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir4)

        if save_stats:
                # if calib5:
                #     layer_name = layer_name + "_calibration"
                # else:
                #     layer_name = layer_name + f"_bsz_{num_bsz}"
                # if step == 0 and (i == -1 or i in [99]) and (num == 0 or num == None or num == -1):
                # # if step in [0, 1, 63] and (i == -1 or i in [99, 98, 97]) and (num == 0 or num == None):
                    save_quantization_stats(
                            smoothed_x=x,
                            smoothed_weight=self.weight,
                            # smoothed_x_quant=x_quant,
                            # smoothed_weight_quant=w,
                            step=step,
                            i = i,
                            layer_name=layer_name,
                            dim_ac = 2,
                            dim = 1
                        )
                
                    save_plots(smoothed_x=x,
                            smoothed_weight=self.weight,
                            # smoothed_x_quant=x_quant,
                            # smoothed_weight_quant=w,
                            step=step,
                            i = i,
                            layer_name=layer_name,
                            dim_ac = 2,
                            dim = 1
                            )


        torch.cuda.empty_cache()

        # 执行线性变换
        out = F.linear(x_quantized, weight=quantized_weights, bias=self.bias)
        # logger3.info(f"Output Of Step{step} Timestep{i} {layer_name}: min={out.min()}, max={out.max()}, mean={out.mean()}, shape={out.shape}")
        return out
    

class QuantLinearWithGrouping_diff(nn.Linear):
    """
    Class to quantize weights and activations of given Linear layer using grouping based on IQR (Interquartile Range)
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 i=None, device='cuda'):
        super(QuantLinearWithGrouping_diff, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer_group_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_group_diff(**weight_quant_params)
        # self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.i = None
        self.threshold_input = torch.zeros(6400, device=device)
        self.threshold_input_init = False
        self.threshold_weight = torch.zeros(1, device=device)
        self.threshold_weight_init = False
        self.channel_indices_input = None
        self.channel_indices_weight = None
        self.channel_indices_weight_init = False
        self.channel_indices_input_init = False
        self.sign_mask = None
        self.high_value_coords = None

    def __repr__(self):
        s = super(QuantLinearWithGrouping_diff, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def get_iqr_threshold(self, values,by_channel = False, i=None, step=None, is_weight=False):
        """
        Calculate the IQR threshold for grouping the values based on the Interquartile Range.
        """
        # from skimage.filters import threshold_otsu
        # flat_values = values.reshape(-1).cpu().detach().numpy()
        # # q75, q25 = np.percentile(flat_values, [75, 25])
        # # iqr = q75 - q25
        # # threshold = q75 + 1.5 * iqr
        # # return threshold
        # # q99 = np.percentile(flat_values,99)
        # # from skimage.filters import threshold_otsu
        # threshold = threshold_otsu(flat_values)
        # # q99_9 = np.percentile(flat_values,99.9)
        # # # q99_99 = np.percentile(flat_values,99.99)

        # return q99_9
        # if is_weight:
        #     values_clone = values.abs()
        if by_channel:
            optimal_threshold,channel_indices = find_optimal_threshold(values,by_channel)
        else:
            optimal_threshold = find_optimal_threshold(values,by_channel,is_weight= is_weight)
        # if not is_weight and not self.channel_indices_input_init:
        #     self.channel_indices_input = channel_indices
        #     self.channel_indices_input_init = True
        return torch.tensor(optimal_threshold, dtype=torch.float32, device=values.device)
        # return torch.tensor(threshold, dtype=torch.float32, device=values.device)
    
        # from sklearn.cluster import KMeans
        # sorted_values = np.sort(flat_values)
        # values_reshaped = sorted_values.reshape(-1, 1)  # K-means 需要二维数据

        # # 运行 K-means 聚类，分成两个簇
        # kmeans = KMeans(n_clusters=2)
        # kmeans.fit(values_reshaped)

        # # 获取每个数据点所属的簇标签
        # labels = kmeans.labels_

        # # 找到两个簇的分界点
        # # 第一个簇的最后一个元素和第二个簇的第一个元素之间就是临界点
        # cluster_1_last_index = np.max(np.where(labels == 0))  # 第一个簇的最后一个点的索引
        # cluster_2_first_index = np.min(np.where(labels == 1))  # 第二个簇的第一个点的索引

        # # 临界点的两个数据值
        # critical_point_1 = sorted_values[cluster_1_last_index]
        # critical_point_2 = sorted_values[cluster_2_first_index]

        # # 计算临界点的中点，作为阈值
        # threshold = (critical_point_1 + critical_point_2) / 2

        # return threshold

    def quantize_input_separated(self, values, threshold, i, step, calib5, adjustment):
        """
        Perform quantization for the full input by splitting the input into two parts 
        and quantizing them separately based on the threshold.
        """
        if self.channel_indices_input_init and self.channel_indices_input is not None:
            # 获取高值通道索引
            high_indices = self.channel_indices_input

            if not high_indices:  # 如果没有高值通道，全部走低值量化
                return self.input_quantizer(values, group_name="low", i=i, step=step, calib5=calib5, adjustment=adjustment)

            # 对高值通道进行量化
            high_values = values[..., high_indices]
            high_quantized = self.input_quantizer(high_values, group_name="high", i=i, step=step, calib5=calib5, adjustment=adjustment)

            # 制作低值组数据：
            # 复制原始张量，并将高值通道位置置零，这样在低值组量化时，高值数据不会对量化参数产生影响
            low_values = values.clone()
            low_values[..., high_indices] = 0

            # 对整个低值组（实际上高值位置已经为0）进行量化
            low_quantized = self.input_quantizer(low_values, group_name="low", i=i, step=step, calib5=calib5, adjustment=adjustment)

            # 最后将高值组的量化结果覆盖到低值组结果的对应位置
            low_quantized[..., high_indices] = high_quantized

        else:

            values_clone = values.clone() 
            # high_values = values.clone()
            
            if threshold >= values.max():
                low_quantized = self.input_quantizer(values_clone, group_name="low", 
                                                            i=i, step=step, calib5=calib5, adjustment=adjustment,is_weight = False)
            else:
                # if calib5:
                mask = values_clone > threshold
                    # indices = self.get_threshold_exceeding_indices(high_values, threshold, logger3)
                self.high_value_coords = torch.nonzero(mask)  # 获取大于阈值的所有位置坐标，并保存到列表
                    # self.high_value_coords = self.high_value_coords.squeeze(1)  # 确保是一个一维的索引张量
                    # self.sign_mask = torch.sign(high_values_to_quantize)
                # high_values_to_quantize = values_clone[self.high_value_coords]
                high_values_to_quantize = values_clone[self.high_value_coords[:, 0], self.high_value_coords[:, 1]]

                
                # high_values_abs = high_values_to_quantize.abs() 
                # 分别对低值组和高值组进行量化
                low_quantized = self.input_quantizer(values_clone, group_name="low", 
                                                            i=i, step=step, calib5=calib5, adjustment=adjustment,is_weight = False,threshold = threshold)
                high_quantized = self.input_quantizer(high_values_to_quantize, group_name="high", 
                                                                i=i, step=step, calib5=calib5, adjustment=adjustment,is_weight = False)
                # high_weight_quantized = high_weight_quantized*self.sign_mask
            
                
                low_quantized[self.high_value_coords[:, 0], self.high_value_coords[:, 1]] = high_quantized


        return low_quantized
    
    def quantize_weight_separated(self, values,threshold, i, step, calib5, adjustment):
        """
        对权重张量进行量化，将数据分为高值组和低值组分别量化，
        低值组量化时先将高值通道位置置零，然后再将高值组的量化结果替换回来。
        """
        values_clone = values.clone() 
        # high_values = values.clone()
        
        if threshold >= values.max():
            low_weight_quantized = self.weight_quantizer(values_clone, group_name="low", 
                                                        i=i, step=step, calib5=calib5, adjustment=adjustment,is_weight = True,threshold = threshold)
        else:
            # if calib5:
            mask = values_clone > threshold
                # indices = self.get_threshold_exceeding_indices(high_values, threshold, logger3)
            self.high_value_coords = torch.nonzero(mask)  # 获取大于阈值的所有位置坐标，并保存到列表
                # self.high_value_coords = self.high_value_coords.squeeze(1)  # 确保是一个一维的索引张量
                # self.sign_mask = torch.sign(high_values_to_quantize)
            # high_values_to_quantize = values_clone[self.high_value_coords]
            high_values_to_quantize = values_clone[self.high_value_coords[:, 0], self.high_value_coords[:, 1]]

            
            # high_values_abs = high_values_to_quantize.abs() 
            # 分别对低值组和高值组进行量化
            low_weight_quantized = self.weight_quantizer(values_clone, group_name="low", 
                                                        i=i, step=step, calib5=calib5, is_weight = True,threshold = threshold)
            high_weight_quantized = self.weight_quantizer(high_values_to_quantize, group_name="high", 
                                                            i=i, step=step, calib5=calib5, is_weight = True,threshold = threshold)
            # high_weight_quantized = high_weight_quantized*self.sign_mask
           
            
            low_weight_quantized[self.high_value_coords[:, 0], self.high_value_coords[:, 1]] = high_weight_quantized

        return low_weight_quantized

    def process_threshold(self, tensor, step, i):
        """
        Process the tensor based on the threshold to identify feature dimensions (last dimension) 
        that should be handled separately based on the threshold.

        Args:
            tensor (torch.Tensor): The input tensor of shape [batch_size, ..., feature_dim].
            step (int): The current step.
            i (int): The index for tracking the threshold.
        """
        # 获取阈值
        threshold = self.threshold_input[step * 100 + i]

        # 获取张量的形状
        shape = tensor.shape
        feature_dim = shape[-1]  # 最后一个维度是特征维度

        # 用于存储符合条件的特征维度的索引
        valid_feature_indices = []

        # 第一轮：先找到最小值大于阈值的特征维度
        for f in range(feature_dim):
            # 提取该特征维度的数据，假设最后一个维度是特征维度
            feature_data = tensor[..., f].cpu().numpy()  # 获取该特征维度的所有数据

            # 找到该特征维度的最小值
            min_value = feature_data.min()

            # 判断该特征维度的最小值是否大于阈值
            if min_value > threshold:
                valid_feature_indices.append(f)

        # 第二轮：确认除了这些候选特征外，其他特征维度的最大值是否小于等于阈值
        for f in range(feature_dim):
            # 仅对不在候选特征维度列表中的特征维度进行判断
            if f not in valid_feature_indices:
                feature_data = tensor[..., f].cpu().numpy()  # 获取该特征维度的所有数据

                # 找到该特征维度的最大值
                max_value = feature_data.max()

                # 判断该特征维度的最大值是否小于等于阈值
                if max_value > threshold:
                    # 如果有一个特征维度的最大值大于阈值，立即跳出，说明该特征不符合条件
                    print(f"Feature dimension {f} max value {max_value} is above threshold, aborting validation.")
                    return  # 如果不满足条件，直接返回并不再验证 valid_feature_indices

        # 如果第二步验证通过，最终确定 valid_feature_indices 为符合条件的特征维度
        self.channel_indices_input_init = True
        self.channel_indices_input = valid_feature_indices
        self.threshold_input_init = True

        # 打印符合条件的特征维度索引
        print(f"Valid feature dimensions for step {step}, i {i}: {valid_feature_indices}")

    
    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None, params=None, save_stats=False, adjustment=False, num=None, a=False, num_bsz=-1):
        """
        Perform forward pass using quantized or non-quantized weights and activations, and save statistics/images to separate directories.
        """
        self.i = i
        # logger3.info(f"输入激活x的形状为：{x.shape}")
        # save_sorted_activations_and_weights("fc2",x, self.weight, outpath)
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name} x: min={x.min()}, max={x.max()}, mean={x.mean()}")
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name} weight: min={self.weight.min()}, max={self.weight.max()}, mean={self.weight.mean()}")

        if self.use_input_quant:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir2)

            # if torch.all(torch.eq(x, 0)):  # If all elements are zero
            #     x_quantized = x
            # else:
            # if calib5 and not self.threshold_input_init:
            if calib5 and not self.threshold_input_init:
                    # 对输入激活进行 IQR 分组量化
                    self.threshold_input[step*100+i] = self.get_iqr_threshold(x,False,i,step)
                    if step==0 and i == 99:
                        self.process_threshold(x,step,i)
                    # if self.channel_indices_input is not None:
                    #     self.threshold_input_init = True
                    # logger3.info(f"激活分界特征通道为：{self.channel_indices_input}")
                    # compute_threshold_statistics(x,self.threshold_input[step*100+i],layer_name)
            if self.threshold_input[step*100+i] is None:
                thrshold = self.threshold_input[99]
            else:
                thrshold = self.threshold_input[step*100+i]
      
            x_quantized = self.quantize_input_separated(x, thrshold, i, step, calib5, adjustment)
            if draw:
                    save_and_plot(x_quantized, f"Quantized Activation Of Step{step} Timestep{i} {layer_name}", f"quantized_activation_of_step{step}_time{i}_{layer_name}.png", output_dir3)
        else:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir1)
            x_quantized = x
        # torch.cuda.empty_cache()

        # min=self.weight.min()

        if self.use_weight_quant:
            if calib5 and not self.threshold_weight_init:
                self.threshold_weight = self.get_iqr_threshold(self.weight,False,i,step,True)
                self.threshold_weight_init = True
                logger3.info(f"权重分界阈值为：{self.threshold_weight}")
            quantized_weights = self.quantize_weight_separated(self.weight,self.threshold_weight, i=i, step=step, calib5=calib5,adjustment = adjustment)
            # quantized_weights = self.weight_quantizer(self.weight, i=i, step=step, calib5=calib5, is_weight=True)
        
            if draw:
                save_and_plot(quantized_weights, f"Quantized Weights Of Step{step} Timestep{i} {layer_name}", f"quantized_weights_of_step{step}_time{i}_{layer_name}.png", output_dir6)
        else:
            quantized_weights = self.weight
            if draw:
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir4)

        # torch.cuda.empty_cache()
        if save_stats:
                # if calib5:
                #     layer_name = layer_name + "_calibration"
                # else:
                #     layer_name = layer_name + f"_bsz_{num_bsz}"
                # if step == 0 and (i == -1 or i in [99]) and (num == 0 or num == None):
                # # if step in [0, 1, 63] and (i == -1 or i in [99, 98, 97]) and (num == 0 or num == None):
                    save_quantization_stats(
                            smoothed_x=x,
                            smoothed_weight=self.weight,
                            # smoothed_x_quant=x_quant,
                            # smoothed_weight_quant=w,
                            step=step,
                            i = i,
                            layer_name=layer_name,
                            dim_ac = 2,
                            dim = 1
                        )
                
                    save_plots(smoothed_x=x,
                            smoothed_weight=self.weight,
                            # smoothed_x_quant=x_quant,
                            # smoothed_weight_quant=w,
                            step=step,
                            i = i,
                            layer_name=layer_name,
                            dim_ac = 2,
                            dim = 1
                            )


        # 执行线性变换
        if layer_name == 'final_adaLN_modulation[1]':
            out = F.linear(x_quantized, weight=quantized_weights*self.sign_mask, bias=self.bias)
        else:
            out = F.linear(x_quantized, weight=quantized_weights, bias=self.bias)
        # out = F.linear(x_quantized, weight=quantized_weights, bias=self.bias)
        # logger3.info(f"Output Of Step{step} Timestep{i} {layer_name}: min={out.min()}, max={out.max()}, mean={out.mean()}, shape={out.shape}")

        return out
    

class SmoothLinear_ar(nn.Linear):
    """
    Class to apply smooth quantization to both input activations and layer weights.
    """
    def __init__(self, in_features, out_features, alpha=0.5, 
                 input_quant_params={},
                 weight_quant_params={},
                 i = None, device='cuda'):
        """
        Initialize SmoothLinear with given features and alpha.

        Parameters:
        - in_features: Number of input features.
        - out_features: Number of output features.
        - alpha: Balancing parameter for activation and weight scaling.
        """
        super(SmoothLinear_ar, self).__init__(in_features, out_features)
        self.alpha = alpha
        self.i = None
        # self.smooth_scales = None  # Scaling factors computed during calibration

        # self.diffloss_scales = torch.zeros(6400, device=device)  # 用 6400 大小的张量代替列表
        # self.diffloss_counts = torch.zeros(6400, dtype=torch.long, device=device)  # 长整型张量
        # self.diffloss_inits = torch.zeros(6400, dtype=torch.bool, device=device)  # 布尔型张量

        self.other_scales = torch.zeros(64, device=device)  # 用 64 大小的张量代替列表
        self.other_counts = torch.zeros(64, dtype=torch.long, device=device)  # 长整型张量
        self.other_inits = torch.zeros(64, dtype=torch.bool, device=device)  # 布尔型张量

        # self.diffloss_scales = [torch.zeros(1, device=device) for _ in range(6400)]
        # self.diffloss_counts = [torch.zeros(1, device=device) for _ in range(6400)]
        # self.diffloss_inits = [torch.zeros(1, device=device) for _ in range(6400)]

        # self.other_scales = [torch.zeros(1, device=device) for _ in range(64)]
        # self.other_counts = [torch.zeros(1, device=device) for _ in range(64)]
        # self.other_inits = [torch.zeros(1, device=device) for _ in range(64)]


        self.input_quantizer = UniformQuantizer_ar(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super(SmoothLinear_ar, self).__repr__()
        s = f"({s}, alpha={self.alpha}, input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def compute_scales(self, act_scales, weight_scales):
        # 计算平滑因子
        scales = (act_scales.pow(self.alpha) / weight_scales.pow(1 - self.alpha)).clamp(min=1e-5)
        return scales
    

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None,alpha = 0.5, save_stats=False, num = None, adjustment = False,num_bsz = None, params=None):
        """
        Forward pass with smooth quantization applied.

        Parameters:
        - x: Input activations.

        Returns:
        - Output tensor after applying smooth quantization.
        """
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)
            self.input_quantizer.channel_wise = params.get('channel_wise', self.input_quantizer.channel_wise)
        # if self.use_input_quant and self.use_weight_quant:
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")
        self.i = i
        self.alpha = alpha
        # self.weight = self.weight.to(device='cuda')
        # layer_name = self.__class__.__name__ if layer_name is None else layer_name
        
        if self.use_input_quant:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir2)
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir5)
                
            # 记录原始激活值
            if i >= 0:  # diffloss 部分
                layer_type = "diffloss"
                # total_steps = 6400
                index = step * 100 + i
            else:  # 其他部分
                layer_type = "ar"
                # total_steps = 64
                index = step

            if calib5:
                # 计算激活的 scale 和权重的 scale
                act_scales = x.abs().max().clamp(min=1e-5)
                weight_scales = self.weight.abs().max().clamp(min=1e-5)

                # 计算平滑后的 scales
                new_scale = self.compute_scales(act_scales, weight_scales)
                if not isinstance(new_scale, torch.Tensor):
                    new_scale = torch.tensor(new_scale, device=act_scales.device)
                # logger3.info(f"Initialized scales: Scale = {new_scale.item()}")
            else:
                new_scale = None

            # 在 forward 方法中直接更新或初始化 scales
            if i >= 0:
                if calib5:
                    if self.diffloss_inits[index]:
                        # 更新累计值
                        self.diffloss_scales[index] += new_scale
                        self.diffloss_counts[index] += 1
                        if not isinstance(new_scale, torch.Tensor):
                            new_scale = torch.tensor(new_scale, device=act_scales.device)
                        # logger3.info(
                        #     f"{layer_type.capitalize()} Timestep {index}: Scale = {new_scale.item()} "
                        #     f"for calibration iteration {self.diffloss_counts[index].item()}, "
                        #     f"Accumulated Scale = {self.diffloss_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                    else:
                        # 初始化
                        self.diffloss_scales[index] = new_scale
                        self.diffloss_counts[index] = 1
                        self.diffloss_inits[index] = True
                        # logger3.info(
                            # f"{layer_type.capitalize()} Timestep {index}: First Initialization Scale = {new_scale.item()} "
                            # f"for calibration iteration 1, "
                            # f"Total Scale = {self.diffloss_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                else:
                    # if not self.diffloss_inits[index]:
                    #     raise ValueError(
                    #         f"{layer_type.capitalize()} Timestep {index}: Cannot use uninitialized scales when calib5=False"
                    #     )

                    if self.diffloss_counts[index] > 0:
                        # 只有在 `calib5=False` 且是第一次推理时，计算平均值
                        # logger3.info(
                        #     f"{layer_type.capitalize()} Timestep {index}: Total Scale = {self.diffloss_scales[index].item()}"
                        # )
                        self.diffloss_scales[index] /= self.diffloss_counts[index]
                        # logger3.info(
                        #     f"{layer_type.capitalize()} Timestep {index}: Averaged Scale = {self.diffloss_scales[index].item()}"
                        # )
                        self.diffloss_counts[index] = 0  # 清空计数
                        scales = self.diffloss_scales[index].clone()
                    else:
                        scales = self.diffloss_scales[index].clone()

                    # 使用固定值
                    # logger3.info(
                    #     f"Using fixed {layer_type} values for Timestep {index}: Scale = {scales.item()}"
                    # )
            else:  # 其他部分
                if calib5:
                    if self.other_inits[index]:
                        # 更新累计值
                        self.other_scales[index] += new_scale
                        self.other_counts[index] += 1
                        # logger3.info(
                            # f"{layer_type.capitalize()} num_iter {index}: Scale = {new_scale} "
                            # f"for calibration iteration {self.other_counts[index].item()}, "
                            # f"Accumulated Scale = {self.other_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                    else:
                        # 初始化
                        self.other_scales[index] = new_scale
                        self.other_counts[index] = 1
                        self.other_inits[index] = True
                        # logger3.info(
                        #     f"{layer_type.capitalize()} num_iter {index}: First Initialization Scale = {new_scale.item()} "
                            # f"for calibration iteration 1, "
                            # f"Total Scale = {self.other_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                else:
                    # if not self.other_inits[index]:
                    #     raise ValueError(
                    #         f"{layer_type.capitalize()} num_iter {index}: Cannot use uninitialized scales when calib5=False"
                    #     )

                    if self.other_counts[index] > 0:
                        # 只有在 `calib5=False` 且是第一次推理时，计算平均值
                        # logger3.info(
                        #     f"{layer_type.capitalize()} num_iter {index}: Total Scale = {self.other_scales[index].item()}"
                        # )
                        self.other_scales[index] /= self.other_counts[index]
                        # logger3.info(
                        #     f"{layer_type.capitalize()} num_iter {index}: Averaged Scale = {self.other_scales[index].item()}"
                        # )
                        self.other_counts[index] = 0  # 清空计数
                        scales = self.other_scales[index].clone()
                    else:
                        scales = self.other_scales[index].clone()

                    # 使用固定值
                    # logger3.info(
                    #     f"Using fixed {layer_type} values for num_iter {index}: Scale = {scales.item()}"
                    # )

            # 根据 scales 调整输入 x 和权重
            if torch.all(x == 0):  # 如果 x 是全零矩阵
                smoothed_x = x
                smoothed_weight = self.weight
            elif scales is not None:  # 否则，如果 scales 不为 None
                smoothed_x = x / scales
                smoothed_weight = self.weight * scales
            else:  # 如果 scales 为 None
                smoothed_x = x
                smoothed_weight = self.weight

            if draw:
                save_and_plot(smoothed_x, f"Original Activation Of Step{step} Timestep{i} {layer_name} But After Smoothquant", f"original_activation_of_step{step}_time{i}_{layer_name} but after smoothquant.png", output_dir2)
                save_and_plot(smoothed_weight, f"Original Weights Of Step{step} Timestep{i} {layer_name} But After Smoothquant", f"original_weights_of_step{step}_time{i}_{layer_name} but after smoothquant.png", output_dir5)


            # 记录原始激活值
            # logger1.info(f"Original Activation: shape={smoothed_x.shape}, min={smoothed_x.min()}, max={smoothed_x.max()}, mean={smoothed_x.mean()}")
            # logger1.info(f"Original Weights: shape={smoothed_weight.shape}, min={smoothed_weight.min()}, max={smoothed_weight.max()}, mean={smoothed_weight.mean()}")


            smoothed_x_quant = self.input_quantizer(smoothed_x, i=i, step=step, calib5=calib5,adjustment = adjustment)
            smoothed_weight_quant = self.weight_quantizer(smoothed_weight, i=i, step=step, calib5=calib5, is_weight=True)

            # 记录量化后的激活值
            # logger2.info(f"Quantized Activation: shape={smoothed_x_quant.shape}, min={smoothed_x_quant.min()}, max={smoothed_x_quant.max()}, mean={smoothed_x_quant.mean()}")
            # logger2.info(f"Quantized Weights: shape={smoothed_weight_quant.shape}, min={smoothed_weight_quant.min()}, max={smoothed_weight_quant.max()}, mean={smoothed_weight_quant.mean()}")

            if draw:
                save_and_plot(smoothed_x_quant, f"Quantized Activation Of Step{step} Timestep{i} {layer_name}", f"quantized_activation_of_step{step}_time{i}_{layer_name}.png", output_dir3)
                save_and_plot(smoothed_weight_quant, f"Quantized Weights Of Step{step} Timestep{i} {layer_name}", f"quantized_weights_of_step{step}_time{i}_{layer_name}.png", output_dir6)
            
            if save_stats:
                if calib5:
                    layer_name = layer_name + "_calibration"
                else:
                    layer_name = layer_name + f"_bsz_{num_bsz}"
                # if step == 0 and (i == -1 or i in [99, 98, 97]) and (num == 0 or num == None):
                if step in [0, 1, 63] and (i == -1 or i in [99, 98, 97]):
                    #  and (num == 0 or num == None)
                    save_quantization_stats(
                        smoothed_x=smoothed_x,
                        smoothed_weight=smoothed_weight,
                        smoothed_x_quant=smoothed_x_quant,
                        smoothed_weight_quant=smoothed_weight_quant,
                        step=step,
                        i = i,
                        layer_name=layer_name,
                        dim_ac = 2,
                        dim = 1
                    )
            
                    # save_plots(smoothed_x=smoothed_x,
                    #     smoothed_weight=smoothed_weight,
                    #     smoothed_x_quant=smoothed_x_quant,
                    #     smoothed_weight_quant=smoothed_weight_quant,
                    #     step=step,
                    #     i = i,
                    #     layer_name=layer_name,
                    #     dim_ac = 2,
                    #     dim = 1)


                # # 打印每个通道的权重
                # for channel_weight_idx in range(channel_count):
                #     # channel_weights = smoothed_weight[:,channel_weight_idx].cpu().detach().numpy() # 1
                #     channel_weights = smoothed_weight[channel_weight_idx, :].cpu().detach().numpy()  # 0 获取每个通道的权重
                #     save_and_plot(
                #         channel_weights,
                #         f"Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_weights_channel{channel_weight_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_weights",
                #     )

                # for channel_weight_idx in range(channel_count):
                #     # channel_weights = smoothed_weight[:,channel_weight_idx].cpu().detach().numpy() # 1
                #     channel_weights = smoothed_weight_quant[channel_weight_idx, :].cpu().detach().numpy()  # 0 获取每个通道的权重
                #     save_and_plot(
                #         channel_weights,
                #         f"Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_weights_channel{channel_weight_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_weights_quant",
                #     )

                # # 打印每个通道的激活
                # for channel_activation_idx in range(activation_count):
                #     channel_activations = smoothed_x[channel_activation_idx, :].cpu().detach().numpy()  # 获取每个通道的激活
                #     save_and_plot(
                #         channel_activations,
                #         f"Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_activations_channel{channel_activation_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_activations",
                #     )

                # for channel_activation_idx in range(activation_count):
                #     channel_activations = smoothed_x_quant[channel_activation_idx, :].cpu().detach().numpy()  # 获取每个通道的激活
                #     save_and_plot(
                #         channel_activations,
                #         f"Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_activations_channel{channel_activation_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_activations_quant",
                #     )

            # out_original = F.linear(smoothed_x, weight=smoothed_weight, bias=self.bias)
            out = F.linear(smoothed_x_quant, weight=smoothed_weight_quant, bias=self.bias)

            # logger1.info(f"Original output after linear: min={out_original.min()}, max={out_original.max()}, mean={out_original.mean()}")
            # logger2.info(f"Quantized output after linear: min={out.min()}, max={out.max()}, mean={out.mean()}")
            # logger5.info(f"Difference in smoothquant: min={out_original.min() - out.min()}, max={out_original.max() - out.max()}, mean={out_original.mean() - out.mean()}")
            
        else:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir1)
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir4)

            # logger3.info(f"Original Activation: shape={x.shape}, min={x.min()}, max={x.max()}, mean={x.mean()}")
            # logger3.info(f"Original Weights: shape={self.weight.shape}, min={self.weight.min()}, max={self.weight.max()}, mean={self.weight.mean()}")

            out = F.linear(x, weight=self.weight, bias=self.bias)
        # logger3.info(f"Output Of Step{step} Timestep{i} {layer_name}: min={out.min()}, max={out.max()}, mean={out.mean()}")
        return out


class SmoothLinear_diff(nn.Linear):
    """
    Class to apply smooth quantization to both input activations and layer weights.
    """
    def __init__(self, in_features, out_features, alpha=0.5, 
                 input_quant_params={},
                 weight_quant_params={},
                 i = None, device='cuda'):
        """
        Initialize SmoothLinear with given features and alpha.

        Parameters:
        - in_features: Number of input features.
        - out_features: Number of output features.
        - alpha: Balancing parameter for activation and weight scaling.
        """
        super(SmoothLinear_diff, self).__init__(in_features, out_features)
        self.alpha = alpha
        self.i = None
        # self.smooth_scales = None  # Scaling factors computed during calibration

        self.diffloss_scales = torch.zeros(6400, device=device)  # 用 6400 大小的张量代替列表
        self.diffloss_counts = torch.zeros(6400, dtype=torch.long, device=device)  # 长整型张量
        self.diffloss_inits = torch.zeros(6400, dtype=torch.bool, device=device)  # 布尔型张量

        # self.other_scales = torch.zeros(64, device=device)  # 用 64 大小的张量代替列表
        # self.other_counts = torch.zeros(64, dtype=torch.long, device=device)  # 长整型张量
        # self.other_inits = torch.zeros(64, dtype=torch.bool, device=device)  # 布尔型张量

        self.input_quantizer = UniformQuantizer_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super(SmoothLinear_diff, self).__repr__()
        s = f"({s}, alpha={self.alpha}, input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"
        return s

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def compute_scales(self, act_scales, weight_scales):
        # 计算平滑因子
        scales = (act_scales.pow(self.alpha) / weight_scales.pow(1 - self.alpha)).clamp(min=1e-5)
        return scales
    

    def forward(self, x, i=None, step=None, calib5=False, d=False, draw=False, layer_name=None,alpha = 0.5, save_stats=False, num = None, adjustment = False,num_bsz = None, params=None):
        """
        Forward pass with smooth quantization applied.

        Parameters:
        - x: Input activations.

        Returns:
        - Output tensor after applying smooth quantization.
        """
        if params:
            # 动态更新量化参数
            self.weight_quantizer.channel_wise = params.get('channel_wise', self.weight_quantizer.channel_wise)
            # self.input_quantizer.channel_wise = params.get('channel_wise', self.input_quantizer.channel_wise)
        # if self.use_input_quant and self.use_weight_quant:
        # logger3.info(f"Input Of Step{step} Timestep{i} {layer_name}: min={x.min()}, max={x.max()}, mean={x.mean()}")
        self.i = i
        self.alpha = alpha
        # self.weight = self.weight.to(device='cuda')
        # layer_name = self.__class__.__name__ if layer_name is None else layer_name
        
        if self.use_input_quant:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir2)
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir5)
                
            # 记录原始激活值
            if i >= 0:  # diffloss 部分
                layer_type = "diffloss"
                # total_steps = 6400
                index = step * 100 + i
            else:  # 其他部分
                layer_type = "ar"
                # total_steps = 64
                index = step

            if calib5:
                # 计算激活的 scale 和权重的 scale
                act_scales = x.abs().max().clamp(min=1e-5)
                weight_scales = self.weight.abs().max().clamp(min=1e-5)

                # 计算平滑后的 scales
                new_scale = self.compute_scales(act_scales, weight_scales)
                if not isinstance(new_scale, torch.Tensor):
                    new_scale = torch.tensor(new_scale, device=act_scales.device)
                # logger3.info(f"Initialized scales: Scale = {new_scale.item()}")
            else:
                new_scale = None

            # 在 forward 方法中直接更新或初始化 scales
            if i >= 0:
                if calib5:
                    if self.diffloss_inits[index]:
                        # 更新累计值
                        self.diffloss_scales[index] += new_scale
                        self.diffloss_counts[index] += 1
                        if not isinstance(new_scale, torch.Tensor):
                            new_scale = torch.tensor(new_scale, device=act_scales.device)
                        # logger3.info(
                        #     f"{layer_type.capitalize()} Timestep {index}: Scale = {new_scale.item()} "
                        #     f"for calibration iteration {self.diffloss_counts[index].item()}, "
                        #     f"Accumulated Scale = {self.diffloss_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                    else:
                        # 初始化
                        self.diffloss_scales[index] = new_scale
                        self.diffloss_counts[index] = 1
                        self.diffloss_inits[index] = True
                        # logger3.info(
                            # f"{layer_type.capitalize()} Timestep {index}: First Initialization Scale = {new_scale.item()} "
                            # f"for calibration iteration 1, "
                            # f"Total Scale = {self.diffloss_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                else:
                    # if not self.diffloss_inits[index]:
                    #     raise ValueError(
                    #         f"{layer_type.capitalize()} Timestep {index}: Cannot use uninitialized scales when calib5=False"
                    #     )

                    if self.diffloss_counts[index] > 0:
                        # 只有在 `calib5=False` 且是第一次推理时，计算平均值
                        # logger3.info(
                        #     f"{layer_type.capitalize()} Timestep {index}: Total Scale = {self.diffloss_scales[index].item()}"
                        # )
                        self.diffloss_scales[index] /= self.diffloss_counts[index]
                        # logger3.info(
                        #     f"{layer_type.capitalize()} Timestep {index}: Averaged Scale = {self.diffloss_scales[index].item()}"
                        # )
                        self.diffloss_counts[index] = 0  # 清空计数
                        scales = self.diffloss_scales[index].clone()
                    else:
                        scales = self.diffloss_scales[index].clone()

                    # 使用固定值
                    # logger3.info(
                    #     f"Using fixed {layer_type} values for Timestep {index}: Scale = {scales.item()}"
                    # )
            else:  # 其他部分
                if calib5:
                    if self.other_inits[index]:
                        # 更新累计值
                        self.other_scales[index] += new_scale
                        self.other_counts[index] += 1
                        # logger3.info(
                            # f"{layer_type.capitalize()} num_iter {index}: Scale = {new_scale} "
                            # f"for calibration iteration {self.other_counts[index].item()}, "
                            # f"Accumulated Scale = {self.other_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                    else:
                        # 初始化
                        self.other_scales[index] = new_scale
                        self.other_counts[index] = 1
                        self.other_inits[index] = True
                        # logger3.info(
                        #     f"{layer_type.capitalize()} num_iter {index}: First Initialization Scale = {new_scale.item()} "
                            # f"for calibration iteration 1, "
                            # f"Total Scale = {self.other_scales[index].item()}"
                        # )
                        scales = new_scale.clone()
                else:
                    # if not self.other_inits[index]:
                    #     raise ValueError(
                    #         f"{layer_type.capitalize()} num_iter {index}: Cannot use uninitialized scales when calib5=False"
                    #     )

                    if self.other_counts[index] > 0:
                        # 只有在 `calib5=False` 且是第一次推理时，计算平均值
                        # logger3.info(
                        #     f"{layer_type.capitalize()} num_iter {index}: Total Scale = {self.other_scales[index].item()}"
                        # )
                        self.other_scales[index] /= self.other_counts[index]
                        # logger3.info(
                        #     f"{layer_type.capitalize()} num_iter {index}: Averaged Scale = {self.other_scales[index].item()}"
                        # )
                        self.other_counts[index] = 0  # 清空计数
                        scales = self.other_scales[index].clone()
                    else:
                        scales = self.other_scales[index].clone()

                    # 使用固定值
                    # logger3.info(
                    #     f"Using fixed {layer_type} values for num_iter {index}: Scale = {scales.item()}"
                    # )

            # 根据 scales 调整输入 x 和权重
            if torch.all(x == 0):  # 如果 x 是全零矩阵
                smoothed_x = x
                smoothed_weight = self.weight
            elif scales is not None:  # 否则，如果 scales 不为 None
                smoothed_x = x / scales
                smoothed_weight = self.weight * scales
            else:  # 如果 scales 为 None
                smoothed_x = x
                smoothed_weight = self.weight

            if draw:
                save_and_plot(smoothed_x, f"Original Activation Of Step{step} Timestep{i} {layer_name} But After Smoothquant", f"original_activation_of_step{step}_time{i}_{layer_name} but after smoothquant.png", output_dir2)
                save_and_plot(smoothed_weight, f"Original Weights Of Step{step} Timestep{i} {layer_name} But After Smoothquant", f"original_weights_of_step{step}_time{i}_{layer_name} but after smoothquant.png", output_dir5)


            # 记录原始激活值
            # logger1.info(f"Original Activation: shape={smoothed_x.shape}, min={smoothed_x.min()}, max={smoothed_x.max()}, mean={smoothed_x.mean()}")
            # logger1.info(f"Original Weights: shape={smoothed_weight.shape}, min={smoothed_weight.min()}, max={smoothed_weight.max()}, mean={smoothed_weight.mean()}")


            smoothed_x_quant = self.input_quantizer(smoothed_x, i=i, step=step, calib5=calib5,adjustment = adjustment)
            smoothed_weight_quant = self.weight_quantizer(smoothed_weight, i=i, step=step, calib5=calib5, is_weight=True)

            # 记录量化后的激活值
            # logger2.info(f"Quantized Activation: shape={smoothed_x_quant.shape}, min={smoothed_x_quant.min()}, max={smoothed_x_quant.max()}, mean={smoothed_x_quant.mean()}")
            # logger2.info(f"Quantized Weights: shape={smoothed_weight_quant.shape}, min={smoothed_weight_quant.min()}, max={smoothed_weight_quant.max()}, mean={smoothed_weight_quant.mean()}")

            if draw:
                save_and_plot(smoothed_x_quant, f"Quantized Activation Of Step{step} Timestep{i} {layer_name}", f"quantized_activation_of_step{step}_time{i}_{layer_name}.png", output_dir3)
                save_and_plot(smoothed_weight_quant, f"Quantized Weights Of Step{step} Timestep{i} {layer_name}", f"quantized_weights_of_step{step}_time{i}_{layer_name}.png", output_dir6)
            
            if save_stats:
                if calib5:
                    layer_name = layer_name + "_calibration"
                else:
                    layer_name = layer_name + f"_bsz_{num_bsz}"
                # if step == 0 and (i == -1 or i in [99, 98, 97]) and (num == 0 or num == None):
                if step in [0, 1, 63] and (i == -1 or i in [99, 98, 97]):
                    #  and (num == 0 or num == None)
                    save_quantization_stats(
                        smoothed_x=smoothed_x,
                        smoothed_weight=smoothed_weight,
                        smoothed_x_quant=smoothed_x_quant,
                        smoothed_weight_quant=smoothed_weight_quant,
                        step=step,
                        i = i,
                        layer_name=layer_name,
                        dim_ac = 2,
                        dim = 1
                    )
            
                    # save_plots(smoothed_x=smoothed_x,
                    #     smoothed_weight=smoothed_weight,
                    #     smoothed_x_quant=smoothed_x_quant,
                    #     smoothed_weight_quant=smoothed_weight_quant,
                    #     step=step,
                    #     i = i,
                    #     layer_name=layer_name,
                    #     dim_ac = 2,
                    #     dim = 1)


                # # 打印每个通道的权重
                # for channel_weight_idx in range(channel_count):
                #     # channel_weights = smoothed_weight[:,channel_weight_idx].cpu().detach().numpy() # 1
                #     channel_weights = smoothed_weight[channel_weight_idx, :].cpu().detach().numpy()  # 0 获取每个通道的权重
                #     save_and_plot(
                #         channel_weights,
                #         f"Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_weights_channel{channel_weight_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_weights",
                #     )

                # for channel_weight_idx in range(channel_count):
                #     # channel_weights = smoothed_weight[:,channel_weight_idx].cpu().detach().numpy() # 1
                #     channel_weights = smoothed_weight_quant[channel_weight_idx, :].cpu().detach().numpy()  # 0 获取每个通道的权重
                #     save_and_plot(
                #         channel_weights,
                #         f"Weights Distribution of Layer {layer_name}, Channel {channel_weight_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_weights_channel{channel_weight_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_weights_quant",
                #     )

                # # 打印每个通道的激活
                # for channel_activation_idx in range(activation_count):
                #     channel_activations = smoothed_x[channel_activation_idx, :].cpu().detach().numpy()  # 获取每个通道的激活
                #     save_and_plot(
                #         channel_activations,
                #         f"Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_activations_channel{channel_activation_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_activations",
                #     )

                # for channel_activation_idx in range(activation_count):
                #     channel_activations = smoothed_x_quant[channel_activation_idx, :].cpu().detach().numpy()  # 获取每个通道的激活
                #     save_and_plot(
                #         channel_activations,
                #         f"Activations Distribution of Layer {layer_name}, Channel {channel_activation_idx}, Step {step}, timestep{i}",
                #         f"{layer_name}_activations_channel{channel_activation_idx}_step{step}_timestep{i}.png",
                #         "/opt/data/private/GaoJing/deeplearnng/mar/plot/final_layer_adaLN_channel1/each_channels_activations_quant",
                #     )

            # out_original = F.linear(smoothed_x, weight=smoothed_weight, bias=self.bias)
            out = F.linear(smoothed_x_quant, weight=smoothed_weight_quant, bias=self.bias)

            # logger1.info(f"Original output after linear: min={out_original.min()}, max={out_original.max()}, mean={out_original.mean()}")
            # logger2.info(f"Quantized output after linear: min={out.min()}, max={out.max()}, mean={out.mean()}")
            # logger5.info(f"Difference in smoothquant: min={out_original.min() - out.min()}, max={out_original.max() - out.max()}, mean={out_original.mean() - out.mean()}")
            
        else:
            if draw:
                save_and_plot(x, f"Original Activation Of Step{step} Timestep{i} {layer_name}", f"original_activation_of_step{step}_time{i}_{layer_name}.png", output_dir1)
                save_and_plot(self.weight, f"Original Weights Of Step{step} Timestep{i} {layer_name}", f"original_weights_of_step{step}_time{i}_{layer_name}.png", output_dir4)

            # logger3.info(f"Original Activation: shape={x.shape}, min={x.min()}, max={x.max()}, mean={x.mean()}")
            # logger3.info(f"Original Weights: shape={self.weight.shape}, min={self.weight.min()}, max={self.weight.max()}, mean={self.weight.mean()}")

            out = F.linear(x, weight=self.weight, bias=self.bias)
        # logger3.info(f"Output Of Step{step} Timestep{i} {layer_name}: min={out.min()}, max={out.max()}, mean={out.mean()}")
        return out



# def save_sorted_activations_and_weights(layer_name, activations, weights, output_dir):
#         """
#         Save the sorted activations and weights for a given layer to a file.
#         :param layer_name: The name of the layer
#         :param activations: The activations from the forward pass
#         :param weights: The weights of the layer
#         :param output_dir: The directory to save the sorted results
#         """
#         # Ensure the output directory exists
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Flatten activations and weights into 1D arrays
#         activations_flat = activations.flatten().cpu().detach().numpy()
#         weights_flat = weights.flatten().cpu().detach().numpy()

#         # Get sorted indices
#         sorted_activation_indices = np.argsort(activations_flat)
#         sorted_weight_indices = np.argsort(weights_flat)

#         # Create sorted activations and weights (with their corresponding indices)
#         sorted_activations = activations_flat[sorted_activation_indices]
#         sorted_weights = weights_flat[sorted_weight_indices]

#         # Save to files
#         activation_filename = os.path.join(output_dir, f"{layer_name}_sorted_activations.txt")
#         weight_filename = os.path.join(output_dir, f"{layer_name}_sorted_weights.txt")

#         # Write sorted activations to a file
#         with open(activation_filename, 'w') as f:
#             for value in sorted_activations:
#                 f.write(f"{value}\n")
        
#         # Write sorted weights to a file
#         with open(weight_filename, 'w') as f:
#             for value in sorted_weights:
#                 f.write(f"{value}\n")

#         print(f"Sorted activations and weights saved to {output_dir}")
