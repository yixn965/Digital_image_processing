import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from utils.logger_utils import logger1, logger2, logger3, logger4, logger5

logger = logging.getLogger(__name__)

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer_ar(nn.Module):
    """
    用于非对称量化的 PyTorch 模块，支持基于通道的量化以及动态初始化。
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_ar, self).__init__()
        assert 2 <= n_bits <= 8, '不支持的比特宽度'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None
        
        # 初始化设备
        # self.device = 'cuda'  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化激活参数
        
        self.other_activations_zs = torch.zeros(64, device=self.device)
        self.other_activations_ss = torch.zeros(64, device=self.device)
        # self.other_activations_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_inits = torch.zeros(64, dtype=torch.bool, device=self.device)
        # self.other_activations_counts = torch.zeros(64, device=self.device)
        self.diffloss_zs = torch.zeros(6400, device=self.device)
        self.diffloss_ss = torch.zeros(6400, device=self.device)
       
        # 初始化权重参数
       
        # self.other_weights_zs = torch.zeros(64, device=self.device)
        # self.other_weights_ss = torch.zeros(64, device=self.device)

        # self.other_weights_zs = torch.zeros(1, device=self.device)
        # self.other_weights_ss = torch.zeros(1, device=self.device)
        # self.other_weights_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_inits = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.diffloss_weights_zs = torch.zeros(6400, device=self.device)
        self.diffloss_weights_ss = torch.zeros(6400, device=self.device)
        # self.other_weights_counts = torch.zeros(64, device=self.device)

        # self.adjustment_factor = torch.zeros(6400, device=self.device)

        # self.other_activations_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(64)]

        # # other_weights 相关
        self.other_weights_zs = [torch.zeros(1, device=self.device) for _ in range(1)]
        self.other_weights_ss = [torch.zeros(1, device=self.device) for _ in range(1)]
        # self.other_weights_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(64)]


    def forward(self, x: torch.Tensor, i=None, step=0, calib5=False, is_weight=False, adjustment = False, group_name=None):
        self.i = i
        device = x.device  # 获取输入张量的设备
        

        # if is_weight:
        # logger1.info(f"Original x: shape={x.shape}, min={x.min()}, max={x.max()}, mean={x.mean()}")

        if is_weight:  # 处理权重
            if i is not None and i >= 0:  # diffloss 权重
                index =  i
                # if calib5 and (not self.other_weights_inits):
                if calib5:
                        # 初始化
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                        # self.diffloss_weights_ss[index], self.diffloss_weights_zs[index] = self.delta, self.zero_point
                        self.diffloss_weights_ss, self.diffloss_weights_zs = self.delta, self.zero_point
                        # self.other_weights_inits = True
                        # logger3.info(
                        #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                        #     # f"Scale = {self.delta.item()} for calibration iteration 1, "
                        #     # f"Total Zero Point = {self.diffloss_weights_zs[index].item()}, Scale = {self.diffloss_weights_ss[index].item()}"
                        # )
                else:
                    self.zero_point = self.diffloss_weights_zs
                    self.delta = self.diffloss_weights_ss
                    # logger3.info(
                    #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    #     )
            elif i == -1:  # other 权重
                index = step
                # 其他部分的权重处理
                # if calib5 and (not self.other_weights_inits):
                if calib5 or adjustment:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                    self.other_weights_ss, self.other_weights_zs = self.delta, self.zero_point
                    # self.other_weights_inits = True
                        # self.other_weights_counts[index] = 1
                        # self.other_weights_inits[index] = True
                        # logger3.info(
                        #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                        # )
                else:
                    self.zero_point = self.other_weights_zs
                    self.delta = self.other_weights_ss
                    # logger3.info(
                    #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    # )
        else:  # 处理激活
            if i is not None and i >= 0:  # diffloss 激活
                index =  i
                if calib5:
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                        self.diffloss_ss[index], self.diffloss_zs[index] = self.delta, self.zero_point
                        # self.diffloss_counts[index] = 1
                        # self.diffloss_inits[index] = True
                        # logger3.info(
                        #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()}  "
                        # )
                else:
                    self.zero_point = self.diffloss_zs[index]
                    self.delta = self.diffloss_ss[index]
                    # logger3.info(
                    #     f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    # )
            elif i == -1:  # other 激活
                index = step
                # 其他部分的激活处理
                if calib5:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                    self.other_activations_ss[index], self.other_activations_zs[index] = self.delta, self.zero_point
                    
                else:
                    self.zero_point = self.other_activations_zs[index]
                    self.delta = self.other_activations_ss[index]
                    # logger3.info(
                        # f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
                    # )

        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

# 原
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False):
        delta = torch.tensor(0.0)  # 可以是 torch.tensor(0) 代表整数0，或 torch.tensor(0.0) 代表浮点数0
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)  # 计算量化步长
            zero_point = (-x_min / delta).round()  # 计算零点
        return delta, zero_point



class UniformQuantizer_diff(nn.Module):
    """
    用于非对称量化的 PyTorch 模块，支持基于通道的量化以及动态初始化。
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_diff, self).__init__()
        assert 2 <= n_bits <= 8, '不支持的比特宽度'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None
        
        # 初始化设备
        # self.device = 'cuda'  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化激活参数
        self.diffloss_zs = torch.zeros(6400, device=self.device)
        self.diffloss_ss = torch.zeros(6400, device=self.device)
        # self.diffloss_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)
        # self.diffloss_counts = torch.zeros(6400, device=self.device)
        
        # self.diffloss_zs = torch.full((6400,), 127, device=self.device)
        # self.diffloss_ss = torch.full((6400,), 0.04706, device=self.device)
        # 0.02353     --3
        # 0.03137 127 --4
        # 0.03922     --5
        # 0.04706     --6
        # 0.05490     --7
        # 0.04347,128    --5.545

        # 初始化权重参数
        # self.diffloss_weights_zs = torch.zeros(6400, device=self.device)
        # self.diffloss_weights_ss = torch.zeros(6400, device=self.device)
        # self.diffloss_weights_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)
        # self.diffloss_weights_counts = torch.zeros(6400, device=self.device)
        
       
        # self.adjustment_factor = torch.zeros(6400, device=self.device)

        # # diffloss 相关
        # self.diffloss_zs = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_ss = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(6400)]

        
        # # diffloss_weights 相关
        self.diffloss_weights_zs = [torch.zeros(1, device=self.device) for _ in range(6400)]
        self.diffloss_weights_ss = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_weights_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(6400)]

    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False, group_name=None,threshold = None):
        self.i = i
        device = x.device  # 获取输入张量的设备
        

        # if is_weight:
        # logger1.info(f"Original x: shape={x.shape}, min={x.min()}, max={x.max()}, mean={x.mean()}")

        if is_weight:  # 处理权重
            if i is not None and i >= 0:  # diffloss 权重
                index =  i
                if calib5 or adjustment:
                        # 初始化
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                        self.diffloss_weights_ss[index], self.diffloss_weights_zs[index] = self.delta, self.zero_point
                        # logger3.info(
                        #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                        #     # f"Scale = {self.delta.item()} for calibration iteration 1, "
                        #     # f"Total Zero Point = {self.diffloss_weights_zs[index].item()}, Scale = {self.diffloss_weights_ss[index].item()}"
                        # )
                else:
                    self.zero_point = self.diffloss_weights_zs[index]
                    self.delta = self.diffloss_weights_ss[index]
                    # logger3.info(
                    #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    #     )
            elif i == -1:  # other 权重
                index = step
                # 其他部分的权重处理
                if calib5:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                    self.other_weights_ss[index], self.other_weights_zs[index] = self.delta, self.zero_point
                        # self.other_weights_counts[index] = 1
                        # self.other_weights_inits[index] = True
                        # logger3.info(
                        #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                        # )
                else:
                    self.zero_point = self.other_weights_zs[index]
                    self.delta = self.other_weights_ss[index]
                    # logger3.info(
                    #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    # )
        else:  # 处理激活
            if i is not None and i >= 0:  # diffloss 激活
                index = i
                if calib5:
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,threshold = threshold)
                        self.diffloss_ss[index], self.diffloss_zs[index] = self.delta, self.zero_point
                        # self.diffloss_counts[index] = 1
                        # self.diffloss_inits[index] = True
                        # logger3.info(
                        #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()}  "
                        # )
                else:
                    self.zero_point = self.diffloss_zs[index]
                    self.delta = self.diffloss_ss[index]
                    # logger3.info(
                    #     f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    # )
            elif i == -1:  # other 激活
                index = step
                # 其他部分的激活处理
                if calib5:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                    self.other_activations_ss[index], self.other_activations_zs[index] = self.delta, self.zero_point
                        # self.other_activations_counts[index] = 1
                        # self.other_activations_inits[index] = True
                        # logger3.info(
                        #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
                        # )
                else:
                    self.zero_point = self.other_activations_zs[index]
                    self.delta = self.other_activations_ss[index]
                    # logger3.info(
                        # f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
                    # )

        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

# 原
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False,threshold = None):
        delta = torch.tensor(0.0)  # 可以是 torch.tensor(0) 代表整数0，或 torch.tensor(0.0) 代表浮点数0
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            if threshold is not None:
                x_max = threshold
                x_min = x_clone.min()
            else:
                x_max = x_clone.max()
                x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)  # 计算量化步长
            zero_point = (-x_min / delta).round()  # 计算零点
        return delta, zero_point


class UniformQuantizer_group(nn.Module):
    """
    用于非对称量化的 PyTorch 模块，支持基于通道的量化以及动态初始化。
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_group, self).__init__()
        assert 2 <= n_bits <= 8, '不支持的比特宽度'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        # self.delta1 = None
        # self.zero_point1 = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None
        
        # 初始化设备
        # self.device = 'cuda'  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化激活参数
        # self.diffloss_zs = torch.zeros(6400, device=self.device)
        # self.diffloss_ss = torch.zeros(6400, device=self.device)
        # self.diffloss_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)
        # self.diffloss_counts = torch.zeros(6400, device=self.device)
        
        # self.diffloss_zs = torch.full((6400,), 127, device=self.device)
        # self.diffloss_ss = torch.full((6400,), 0.03922, device=self.device)
        # 0.02353     --3
        # 0.03137 127 --4
        # 0.03922     --5
        # 0.04706     --6
        # 0.05490     --7

        self.other_activations_zs = torch.zeros(6400, device=self.device)
        self.other_activations_ss = torch.zeros(6400, device=self.device)
        # self.other_activations_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_inits = torch.zeros(64, dtype=torch.bool, device=self.device)
        # self.other_activations_counts = torch.zeros(64, device=self.device)
        self.other_activations_zs_high = torch.zeros(6400, device=self.device)
        self.other_activations_ss_high = torch.zeros(6400, device=self.device)
        
        # 初始化权重参数
        # self.diffloss_weights_zs = torch.zeros(6400, device=self.device)
        # self.diffloss_weights_ss = torch.zeros(6400, device=self.device)
        # self.diffloss_weights_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)
        # self.diffloss_weights_counts = torch.zeros(6400, device=self.device)
        
        self.other_weights_zs = torch.zeros(6400, device=self.device)
        self.other_weights_ss = torch.zeros(6400, device=self.device)
        # self.other_weights_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_inits = torch.zeros(64, dtype=torch.bool, device=self.device)
        # self.other_weights_counts = torch.zeros(64, device=self.device)
        self.other_weights_zs_high = torch.zeros(6400, device=self.device)
        self.other_weights_ss_high = torch.zeros(6400, device=self.device)

        # self.adjustment_factor = torch.zeros(6400, device=self.device)

        # # diffloss 相关
        # self.diffloss_zs = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_ss = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(6400)]

        # # other_activations 相关
        # self.other_activations_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(64)]

        # # diffloss_weights 相关
        # self.diffloss_weights_zs = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_weights_ss = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_weights_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(6400)]

        # # other_weights 相关
        # self.other_weights_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(64)]


    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False,group_name=None,threshold = None):
        self.i = i
        device = x.device  # 获取输入张量的设备
        # if is_weight:
        # logger1.info(f"Original x: shape={x.shape}, min={x.min()}, max={x.max()}, mean={x.mean()}")
        index = i
        if is_weight:  # 处理权重 
                if calib5:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                    self.other_weights_ss[index], self.other_weights_zs[index] = self.delta, self.zero_point
                else:
                    self.zero_point = self.other_weights_zs[index]
                    self.delta = self.other_weights_ss[index]
        else:  # 处理激活
                if calib5:
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, is_weight,group_name,threshold)
                            self.other_activations_ss_high[index], self.other_activations_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,False,group_name,threshold)
                            self.other_activations_ss[index], self.other_activations_zs[index] = self.delta, self.zero_point
                        # self.other_activations_counts[index] = 1
                        # self.other_activations_inits[index] = True
                        # logger3.info(
                        #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
                        # )
                else:
                    if group_name=="high":
                        self.zero_point = self.other_activations_zs_high[index]
                        self.delta = self.other_activations_ss_high[index]
                    else:
                        self.zero_point = self.other_activations_zs[index]
                        self.delta = self.other_activations_ss[index]
                    # logger3.info(
                        # f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
                    # )

        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

# 原
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False, group_name=None,threshold = None):
        delta = torch.tensor(0.0)  # 可以是 torch.tensor(0) 代表整数0，或 torch.tensor(0.0) 代表浮点数0
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            if threshold is not None:
                if group_name == "low":
                    x_max = threshold
                    x_min = x_clone.min()
                else:
                    x_min = x_clone.min()
                    x_max = x_clone.max()
            else:
                x_min = x_clone.min()
                x_max = x_clone.max()
            # best_score = 1e+10
            # best_pct = None
            # if is_weight:
            #     pct_values = [1]
            # else:
            #     pct_values = [
            #         0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 
            #         0.99999999, 0.999999999, 0.9999999999, 
            #         0.99999999999, 0.999999999999, 0.9999999999999, 1
            #     ]
            # for pct in [0.999,0.99999, 0.999999999, 0.99999999999, 1]:
            # for pct in [1]:
            # # for pct in pct_values:
            #     try:
            #         new_max = torch.quantile(x_clone.reshape(-1), pct)
            #         new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
            #     except:
            #         new_max = torch.tensor(np.percentile(
            #             x_clone.reshape(-1).cpu(), pct * 100),
            #             device=x_clone.device,
            #             dtype=torch.float32)
            #         new_min = torch.tensor(np.percentile(
            #             x_clone.reshape(-1).cpu(), (1 - pct) * 100),
            #             device=x_clone.device,
            #             dtype=torch.float32)
                # x_q = self.quantize(x_clone, x_max, x_min)
                # score = lp_loss(x_clone, x_q, p=2, reduction='all')
                # if score < best_score:
                #     best_score = score
                #     best_pct = pct
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)
            zero_point = (- x_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        # print(f"x_float_q: {x_float_q}")
        return x_float_q
    
class UniformQuantizer_group_diff(nn.Module):
    """
    用于非对称量化的 PyTorch 模块，支持基于通道的量化以及动态初始化。
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_group_diff, self).__init__()
        assert 2 <= n_bits <= 8, '不支持的比特宽度'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None
        
        # 初始化设备
        # self.device = 'cuda'  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化激活参数
        # self.diffloss_zs = torch.zeros(6400, device=self.device)
        # self.diffloss_ss = torch.zeros(6400, device=self.device)
        # self.diffloss_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)
        # self.diffloss_counts = torch.zeros(6400, device=self.device)
        
        self.diffloss_zs = torch.zeros(6400, device=self.device)
        self.diffloss_ss = torch.zeros(6400, device=self.device)
        self.diffloss_zs_high = torch.zeros(6400, device=self.device)
        self.diffloss_ss_high = torch.zeros(6400, device=self.device)
        # 0.02353     --3
        # 0.03137 127 --4
        # 0.03922     --5
        # 0.04706     --6
        # 0.05490     --7

        # 初始化权重参数
        self.diffloss_weights_zs = torch.zeros(6400, device=self.device)
        self.diffloss_weights_ss = torch.zeros(6400, device=self.device)
        self.diffloss_weights_zs_high = torch.zeros(6400, device=self.device)
        self.diffloss_weights_ss_high = torch.zeros(6400, device=self.device)
        # self.diffloss_weights_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)
        # self.diffloss_weights_counts = torch.zeros(6400, device=self.device)
        

        # self.adjustment_factor = torch.zeros(6400, device=self.device)

        # # diffloss 相关
        # self.diffloss_zs = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_ss = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(6400)]

        # # other_activations 相关
        # self.other_activations_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_activations_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(64)]

        # # diffloss_weights 相关
        # self.diffloss_weights_zs = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_weights_ss = [torch.zeros(1, device=self.device) for _ in range(6400)]
        # self.diffloss_weights_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(6400)]

        # # other_weights 相关
        # self.other_weights_zs = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
        # self.other_weights_inits = [torch.zeros(1, dtype=torch.bool, device=self.device) for _ in range(64)]


    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False,group_name=None,threshold = None):
        self.i = i
        device = x.device  # 获取输入张量的设备
        

        # if is_weight:
        # logger1.info(f"Original x: shape={x.shape}, min={x.min()}, max={x.max()}, mean={x.mean()}")

        if is_weight:  # 处理权重
            if i is not None and i >= 0:  # diffloss 权重
                index = step * 100 + i
                if calib5:
                        # 初始化
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight,group_name = group_name,threshold = threshold)
                            self.diffloss_weights_ss_high[index], self.diffloss_weights_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight,group_name = group_name,threshold = threshold)
                            self.diffloss_weights_ss[index], self.diffloss_weights_zs[index] = self.delta, self.zero_point
                        # logger3.info(
                        #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                        #     # f"Scale = {self.delta.item()} for calibration iteration 1, "
                        #     # f"Total Zero Point = {self.diffloss_weights_zs[index].item()}, Scale = {self.diffloss_weights_ss[index].item()}"
                        # )
                else:
                    if group_name=="high":
                        self.zero_point = self.diffloss_weights_zs_high[index]
                        self.delta = self.diffloss_weights_ss_high[index]
                    else:
                        self.zero_point = self.diffloss_weights_zs[index]
                        self.delta = self.diffloss_weights_ss[index]
                    # logger3.info(
                    #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    #     )
            elif i == -1:  # other 权重
                index = step
                # 其他部分的权重处理
                if calib5:
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                            self.other_weights_ss_high[index], self.other_weights_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                            self.other_weights_ss[index], self.other_weights_zs[index] = self.delta, self.zero_point
                        # self.other_weights_counts[index] = 1
                        # self.other_weights_inits[index] = True
                        # logger3.info(
                        #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                        # )
                else:
                    if group_name=="high":
                        self.zero_point = self.other_weights_zs_high[index]
                        self.delta = self.other_weights_ss_high[index]
                    else:
                        self.zero_point = self.other_weights_zs[index]
                        self.delta = self.other_weights_ss[index]
                    # logger3.info(
                    #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    # )
        else:  # 处理激活
            if i is not None and i >= 0:  # diffloss 激活
                index = step * 100 + i
                if calib5:
                    if group_name=="high":
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, is_weight,group_name,threshold)
                        self.diffloss_ss_high[index], self.diffloss_zs_high[index] = self.delta, self.zero_point
                    else:
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, is_weight,group_name,threshold)
                        self.diffloss_ss[index], self.diffloss_zs[index] = self.delta, self.zero_point
                        # self.diffloss_counts[index] = 1
                        # self.diffloss_inits[index] = True
                        # logger3.info(
                        #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()}  "
                        # )
                else:
                    if group_name=="high":
                        self.zero_point = self.diffloss_zs_high[index]
                        self.delta = self.diffloss_ss_high[index]
                    else:
                        self.zero_point = self.diffloss_zs[index]
                        self.delta = self.diffloss_ss[index]
                    # logger3.info(
                    #     f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
                    # )
            elif i == -1:  # other 激活
                index = step
                # 其他部分的激活处理
                if calib5:
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                            self.other_activations_ss_high[index], self.other_activations_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                            self.other_activations_ss[index], self.other_activations_zs[index] = self.delta, self.zero_point
                        # self.other_activations_counts[index] = 1
                        # self.other_activations_inits[index] = True
                        # logger3.info(
                        #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
                        # )
                else:
                    if group_name=="high":
                        self.zero_point = self.other_activations_zs_high[index]
                        self.delta = self.other_activations_ss_high[index]
                    else:
                        self.zero_point = self.other_activations_zs[index]
                        self.delta = self.other_activations_ss[index]
                    # logger3.info(
                        # f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
                    # )

        # if adjustment and (not calib5) and (not is_weight):
        #     # 只有那两个层用得到，所以就不用设置条件，凡经过adjustment过程的层均建立数组保存缩放因子
        #     # logger3.info(f"Input: min={x.min()}, max={x.max()}, mean={x.mean()}")
        #     # 默认i一定是0-99的数
        #     self.adjustment_factor[step*100+i] = (x.max() - x.min())/(self.delta * 255)
        #     # logger3.info(f"self.delta: {self.delta}")
        #     # logger3.info(f"adjustment_factor: {self.adjustment_factor[step*100+i]}")
        #     if self.adjustment_factor[step*100+i] > 1:
        #         x = x / self.adjustment_factor[step*100+i]
        #     # logger3.info(f"after adjustment_factor: min={x.min()}, max={x.max()}, mean={x.mean()}")
        
        # if self.zero_point is not None:
        #     self.zero_point = self.zero_point.to(self.device)
        # if self.delta is not None:
        #     self.delta = self.delta.to(self.device)
        # self.zero_point = self.zero_point.to(self.device) 
        # self.delta = self.delta.to(self.device) 
        # if self.delta == 0:
        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        # if adjustment and (not calib5) and (not is_weight):
        #     # logger3.info(f"after quantization : min={x_dequant.min()}, max={x_dequant.max()}, mean={x_dequant.mean()}")
        #     if self.adjustment_factor[step*100+i] > 1:
        #         x_dequant = x_dequant * self.adjustment_factor[step*100+i]
        #     # logger3.info(f"after adjustment_factor : min={x_dequant.min()}, max={x_dequant.max()}, mean={x_dequant.mean()}")


        # if is_weight:
        # logger2.info(f"Quantized x_dequant: shape={x_dequant.shape}, min={x_dequant.min()}, max={x_dequant.max()}, mean={x_dequant.mean()}")
        # logger5.info(f"Difference: min={x.min() - x_dequant.min()}, max={x.max() - x_dequant.max()}, mean={x.mean() - x_dequant.mean()}")
        # mse_loss = F.mse_loss(x_dequant, x)
        # logger5.info(f"MSE between original x and quantized x_dequant: {mse_loss.item()}")

        return x_dequant

# 原
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False,group_name=None,threshold = None):
        delta = torch.tensor(0.0)  # 可以是 torch.tensor(0) 代表整数0，或 torch.tensor(0.0) 代表浮点数0
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            if threshold is not None:
                if group_name == "low":
                    x_max = threshold
                    x_min = x_clone.min()
                else:
                    x_min = x_clone.min()
                    x_max = x_clone.max()
            else:
                x_max = x_clone.max()
                x_min = x_clone.min()
            # best_score = 1e+10
            # best_pct = None
            # if is_weight:
            #     pct_values = [1]
            # else:
            #     pct_values = [
            #         0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 
            #         0.99999999, 0.999999999, 0.9999999999, 
            #         0.99999999999, 0.999999999999, 0.9999999999999, 1
            #     ]
            # for pct in [0.999,0.99999, 0.999999999, 0.99999999999, 1]:
            # for pct in [1]:
            # # for pct in pct_values:
            #     try:
            #         new_max = torch.quantile(x_clone.reshape(-1), pct)
            #         new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
            #     except:
            #         new_max = torch.tensor(np.percentile(
            #             x_clone.reshape(-1).cpu(), pct * 100),
            #             device=x_clone.device,
            #             dtype=torch.float32)
            #         new_min = torch.tensor(np.percentile(
            #             x_clone.reshape(-1).cpu(), (1 - pct) * 100),
            #             device=x_clone.device,
            #             dtype=torch.float32)   
                # x_q = self.quantize(x_clone, new_max, new_min)
                # score = lp_loss(x_clone, x_q, p=2, reduction='all')
                # if score < best_score:
                #     best_score = score
                #     best_pct = pct
                #     delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                #     zero_point = (- new_min / delta).round()
            # logger5.info(f"loss: {best_score}")
            # logger5.info(f"Final best score found with pct = {best_pct}, score = {best_score}")

            # logger5.info("直接使用所有数据的最小值和最大值进行量化，不进行分位数裁剪.")
            # x_clone = x.clone().detach()
            # x_max = x_clone.max()
            # x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)  # 计算量化步长
            zero_point = (-x_min / delta).round()  # 计算零点
            # # x_q = self.quantize(x_clone, x_max, x_min)

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        # print(f"x_float_q: {x_float_q}")
        return x_float_q
    

class UniformQuantizer_group_scaling(nn.Module):
    """
    用于非对称量化的 PyTorch 模块，支持基于通道的量化以及动态初始化。
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_group_scaling, self).__init__()
        assert 2 <= n_bits <= 8, '不支持的比特宽度'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None
        
        # 初始化设备
        # self.device = 'cuda'  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        # self.diffloss_zs = torch.zeros(6400, device=self.device)
        # self.diffloss_ss = torch.zeros(6400, device=self.device)
        # self.diffloss_zs_high = torch.zeros(6400, device=self.device)
        # self.diffloss_ss_high = torch.zeros(6400, device=self.device)
        self.diffloss_zs = torch.full((6400,), 128, device=self.device)
        self.diffloss_ss = torch.full((6400,), 0.04347, device=self.device)
        self.scaling_or_not = torch.full((6400,), False, dtype=torch.bool, device=self.device)  # 全部为 F

        # 0.02353     --3
        # 0.03137 127 --4
        # 0.03922     --5
        # 0.04706     --6
        # 0.05490     --7
        # 0.04347,128    --5.545


    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False, group_name=None,layer_name = None,scale_quant = None,shift_quant = None):
                self.i = i
                device = x.device  # 获取输入张量的设备
                index = step * 100 + i
                # if calib5 or adjustment: #判断是否需要动态确定参数范围,重校准过程在其他参数确定情况下再过一遍
                #     min = x.min()
                #     max = x.max()
                #     if  (min<-5.545) or (max>5.545):
                #         self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                #         # self.diffloss_ss[index], self.diffloss_zs[index] = self.delta, self.zero_point 
                #         self.scaling_or_not[index] = True
                #     else:
                #         self.zero_point = self.diffloss_zs[index]
                #         self.delta = self.diffloss_ss[index]
                #     x_int = torch.round(x  / self.delta) + self.zero_point
                #     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                #     x_dequant = (x_quant - self.zero_point) * self.delta
                    
                # else: # inference
                #     self.zero_point = self.diffloss_zs[index]
                #     self.delta = self.diffloss_ss[index]
            
                #     if self.scaling_or_not[index] == True:
                #         x_int = torch.round((x / scale_quant + shift_quant) / self.delta) + self.zero_point
                #         x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                #         x_dequant = ( (x_quant - self.zero_point) * self.delta - shift_quant ) * scale_quant
                #     else:
                #         x_int = torch.round(x  / self.delta) + self.zero_point
                #         x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                #         x_dequant = (x_quant - self.zero_point) * self.delta
              
                # self.zero_point = self.diffloss_zs[index]
                # self.delta = self.diffloss_ss[index]
            
                # if i>=0:
                #         x_int = torch.round((x / scale_quant + shift_quant) / self.delta) + self.zero_point
                #         x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                #         x_dequant = ( (x_quant - self.zero_point) * self.delta - shift_quant ) * scale_quant
                # else:
                #         x_int = torch.round(x  / self.delta) + self.zero_point
                #         x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                #         x_dequant = (x_quant - self.zero_point) * self.delta

                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                x_int = torch.round(x  / self.delta) + self.zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant = (x_quant - self.zero_point) * self.delta

                return x_dequant
              
# 原
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False):
        delta = torch.tensor(0.0)  # 可以是 torch.tensor(0) 代表整数0，或 torch.tensor(0.0) 代表浮点数0
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)
            zero_point = (- x_min / delta).round()
        return delta, zero_point


class UniformQuantizer_scale_channels(nn.Module):
    """
    用于非对称量化的 PyTorch 模块，支持基于通道的量化以及动态初始化。
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_scale_channels, self).__init__()
        assert 2 <= n_bits <= 8, '不支持的比特宽度'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None
        
        # 初始化设备
        # self.device = 'cuda'  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        # self.diffloss_zs = torch.zeros(6400, device=self.device)
        # self.diffloss_ss = torch.zeros(6400, device=self.device)
        # self.diffloss_zs_high = torch.zeros(6400, device=self.device)
        # self.diffloss_ss_high = torch.zeros(6400, device=self.device)
        self.diffloss_zs = torch.full((6400,), 128, device=self.device)
        self.diffloss_ss = torch.full((6400,), 0.04347, device=self.device)

        # 0.02353     --3
        # 0.03137 127 --4
        # 0.03922     --5
        # 0.04706     --6
        # 0.05490     --7
        # 0.04347,128    --5.545

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False):
        delta = torch.tensor(0.0)  # 可以是 torch.tensor(0) 代表整数0，或 torch.tensor(0.0) 代表浮点数0
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)
            zero_point = (- x_min / delta).round()
        return delta, zero_point

    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False, group_name=None,layer_name = None,scale_quant = None,shift_quant = None):
                self.i = i
                device = x.device  # 获取输入张量的设备
                index = i
              
               
                if calib5:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                else:
                    self.zero_point = self.diffloss_zs[index]
                    self.delta = self.diffloss_ss[index]
            
                # self.zero_point = self.diffloss_zs[index]
                # self.delta = self.diffloss_ss[index]
            
                if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
                    x_int = x + self.zero_point
                else:
                    x_int = torch.round(x / self.delta) + self.zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant = (x_quant - self.zero_point) * self.delta

                return x_dequant





# # repq
#     def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False):
#             delta = torch.tensor(0.0)  # 可以是 torch.tensor(0) 代表整数0，或 torch.tensor(0.0) 代表浮点数0
#             zero_point = torch.tensor(0.0)
#             if channel_wise:
#                 x_clone = x.clone().detach()
#                 n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
#                 if len(x.shape) == 4:
#                     x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
#                 elif len(x.shape) == 2:
#                     x_max = x_clone.abs().max(dim=-1)[0]
#                 elif len(x.shape) == 3:
#                     x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
#                 else:
#                     raise NotImplementedError

#                 delta = x_max.clone()
#                 zero_point = x_max.clone()
#                 # determine the scale and zero point channel-by-channel
#                 for c in range(n_channels):
#                     if len(x.shape) == 3:
#                         delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
#                     else:
#                         delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
#                 if len(x.shape) == 4:
#                     delta = delta.view(-1, 1, 1, 1)
#                     zero_point = zero_point.view(-1, 1, 1, 1)
#                 elif len(x.shape) == 2:
#                     delta = delta.view(-1, 1)
#                     zero_point = zero_point.view(-1, 1)
#                 elif len(x.shape) == 3:
#                     delta = delta.view(1, 1, -1)
#                     zero_point = zero_point.view(1, 1, -1)
#                 else:
#                     raise NotImplementedError
#             else:
#                 x_clone = x.clone().detach()
#                 x_max = x_clone.max()
#                 x_min = x_clone.min()
#                 best_score = 1e+10
#                 for pct in [1]:
#                 # for pct in [0.999,0.99999, 0.999999999, 0.99999999999, 1]:
#                     try:
#                         new_max = torch.quantile(x_clone.reshape(-1), pct)
#                         new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
#                     except:
#                         new_max = torch.tensor(np.percentile(
#                             x_clone.reshape(-1).cpu(), pct * 100),
#                             device=x_clone.device,
#                             dtype=torch.float32)
#                         new_min = torch.tensor(np.percentile(
#                             x_clone.reshape(-1).cpu(), (1 - pct) * 100),
#                             device=x_clone.device,
#                             dtype=torch.float32)   
#                     x_q = self.quantize(x_clone, new_max, new_min)
#                     score = lp_loss(x_clone, x_q, p=2, reduction='all')
#                     if score < best_score:
#                         best_score = score
#                         delta = (new_max - new_min) / (2 ** self.n_bits - 1)
#                         zero_point = (- new_min / delta).round()

#             return delta, zero_point

#     def quantize(self, x, max, min):
#         delta = (max - min) / (2 ** self.n_bits - 1)
#         zero_point = (- min / delta).round()
#         # we assume weight quantization is always signed
#         x_int = torch.round(x / delta)
#         x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
#         x_float_q = (x_quant - zero_point) * delta
#         # print(f"x_float_q: {x_float_q}")
#         return x_float_q
   

# class LogSqrt2Quantizer(nn.Module):
#     """
#     PyTorch Function that can be used for asymmetric quantization (also called uniform affine
#     quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
#     through' on the backward pass, ignoring the quantization that occurred.
#     Based on https://arxiv.org/abs/1806.08342.
#     :param n_bits: number of bit for quantization
#     :param channel_wise: if True, compute scale and zero_point in each channel
#     """
#     def __init__(self, n_bits: int = 8, channel_wise: bool = False):
#         super(LogSqrt2Quantizer, self).__init__()
#         assert 2 <= n_bits <= 8, 'bitwidth not supported'
#         self.n_bits = n_bits
#         self.n_levels = 2 ** self.n_bits
#         self.delta = None
#         self.inited = False
#         self.channel_wise = channel_wise

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # 初始化激活参数
#         self.diffloss_ss = torch.zeros(6400, device=self.device)
#         self.diffloss_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)
#         # self.diffloss_counts = torch.zeros(6400, device=self.device)
        
#         self.diffloss_ss = torch.zeros(6400, device=self.device)

#         # self.other_activations_ss = torch.zeros(64, device=self.device)
#         self.other_activations_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
#         self.other_activations_inits = torch.zeros(64, dtype=torch.bool, device=self.device)
#         # self.other_activations_counts = torch.zeros(64, device=self.device)
        
#         # 初始化权重参数
#         self.diffloss_weights_ss = torch.zeros(6400, device=self.device)
#         self.diffloss_weights_inits = torch.zeros(6400, dtype=torch.bool, device=self.device)

#         self.other_weights_ss = [torch.zeros(1, device=self.device) for _ in range(64)]
#         self.other_weights_inits = torch.zeros(64, dtype=torch.bool, device=self.device)
#         self.other_weights_counts = torch.zeros(64, device=self.device)


#     def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False):
#         self.i = i
#         device = x.device  # 获取输入张量的设备

#         # if is_weight:
#         # logger1.info(f"Original x: shape={x.shape}, min={x.min()}, max={x.max()}, mean={x.mean()}")

#         if is_weight:  # 处理权重
#             if i is not None and i >= 0:  # diffloss 权重
#                 index = step * 100 + i
#                 if calib5:
#                         self.delta = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
#                         self.diffloss_weights_ss[index]= self.delta
#                         self.diffloss_weights_inits[index] = True
#                         # logger3.info(
#                         #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
#                         #     # f"Scale = {self.delta.item()} for calibration iteration 1, "
#                         #     # f"Total Zero Point = {self.diffloss_weights_zs[index].item()}, Scale = {self.diffloss_weights_ss[index].item()}"
#                         # )
#                 else:
#                     #     logger3.info(
#                     #         f"Weight Timestep {index}: Averaged Zero Point = {self.diffloss_weights_zs[index].item()},Scale = {self.diffloss_weights_ss[index].item()} "
#                     #     )
#                     self.delta = self.diffloss_weights_ss[index]
#                     # logger3.info(
#                     #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
#                     #     )
#             elif i == -1:  # other 权重
#                 index = step
#                 # 其他部分的权重处理
#                 if calib5:
#                         self.delta = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
#                         self.other_weights_ss[index]= self.delta
#                         self.other_weights_inits[index] = True
#                         # logger3.info(
#                         #     f"Weight Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
#                         # )
#                 else:
#                     #     logger3.info(
#                     #         f"Weight Timestep {index}: Averaged Zero Point = {self.other_weights_zs[index].item()},Scale = {self.other_weights_ss[index].item()} "
#                     #     )
#                     #     self.other_weights_counts[index] = 0
#                     self.delta = self.other_weights_ss[index]
#                     # logger3.info(
#                     #     f"Using fixed weight values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
#                     # )
#         else:  # 处理激活
#             if i is not None and i >= 0:  # diffloss 激活
#                 index = step * 100 + i
#                 if calib5:
#                     #     # logger3.info(
#                     #     #     f"Activation Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
#                     #     # )
#                     # else:
#                         self.delta = self.init_quantization_scale(x, self.channel_wise)
#                         self.diffloss_ss[index] = self.delta
#                         self.diffloss_inits[index] = True
#                         # logger3.info(
#                         #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()},Scale = {self.delta.item()}  "
#                         # )
#                 else:
#                     #     # logger3.info(
#                     #     #     f"Activation Timestep {index}: Averaged Zero Point = {self.diffloss_zs[index].item()},Scale = {self.diffloss_ss[index].item()} "
#                     #     # )
#                     #     self.diffloss_counts[index] = 0
#                     self.delta = self.diffloss_ss[index]
#                     # logger3.info(
#                     #     f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
#                     # )
#             elif i == -1:  # other 激活
#                 index = step
#                 # 其他部分的激活处理
#                 if calib5:
#                     #     # logger3.info(
#                     #     #     f"Activation Timestep {index}: Zero Point = {self.zero_point.item()},Scale = {self.delta.item()} "
#                     #     # )
#                     # else:
#                         self.delta = self.init_quantization_scale(x, self.channel_wise)
#                         self.other_activations_ss[index] = self.delta
#                         self.other_activations_inits[index] = True
#                         # logger3.info(
#                         #     f"Activation Timestep {index}: First Initialization Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
#                         # )
#                 else:
#                     #     # logger3.info(
#                     #     #     f"Activation Timestep {index}: Averaged Zero Point = {self.other_activations_zs[index].item()},Scale = {self.other_activations_ss[index].item()} "
#                     #     # )
#                     #     self.other_activations_counts[index] = 0
#                     self.delta = self.other_activations_ss[index]
#                     # logger3.info(
#                         # f"Using fixed activation values for Timestep {index}: Zero Point = {self.zero_point.item()}, Scale = {self.delta.item()}"
#                     # )

#         # start quantization
#         x_dequant = self.quantize(x, self.delta)
#         return x_dequant

#     def init_quantization_scale(self, x: torch.Tensor, is_weight=False):
#         delta = None
#         x_clone = x.clone().detach()
#         delta = x_clone.max()
#         best_score = 1e+10
#         for pct in [1]:
#         # for pct in [0.999,0.99999, 0.999999999, 0.99999999999, 1]:
#             try:
#                 new_delta = torch.quantile(x_clone.reshape(-1), pct)
#             except:
#                 new_delta = torch.tensor(np.percentile(
#                     x_clone.reshape(-1).cpu(), pct * 100),
#                     device=x_clone.device,
#                     dtype=torch.float32)
#             x_q = self.quantize(x_clone, new_delta)
#             score = lp_loss(x_clone, x_q, p=2, reduction='all')
#             if score < best_score:
#                 best_score = score
#                 delta = new_delta

#         return delta

#     def quantize(self, x, delta):      
#         from math import sqrt
#         x_int = torch.round(-1 * (x/delta).log2() * 2)
#         mask = x_int >= self.n_levels
#         x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#         odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
#         x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
#         x_float_q[mask] = 0
        
#         return x_float_q