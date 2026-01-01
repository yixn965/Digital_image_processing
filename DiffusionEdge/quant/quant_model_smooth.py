import torch
import torch.nn as nn
import torch.nn.functional as F
from .build_model import MatMul
from .quant_modules import QuantLinear_ar,QuantLinear_diff, QuantMatMul, SmoothLinear_ar,SmoothLinear_diff,QuantLinearWithGrouping,QuantLinearWithGrouping_diff,QuantLinear_scaling,QuantLinear_diff_scale2,QuantLinear_diff_scale,QuantLinear_ar_scale,QuantLinear_ar_outlier,QuantLinear_diff_outlier,QuantLinear_scale_channels,QuantConv2d_ar,QuantConv2d_scale_channels,QuantConv2d_ar_outlier
# from .smooth_quant import smooth_linear
from copy import deepcopy


def quant_model_smooth(model, input_quant_params={}, weight_quant_params={}):
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True
    module_dict = {}
    
    # 遍历模型中的所有模块
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        
        # 查找父模块
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"父模块 {father_name} 未找到")

        if "first_stage_model" in name or "cond_stage_model" in name or "init_conv_mask" in name:
            continue  # 跳过 encoder 和 decoder 部分
        
        # 针对指定的层应用 W8A8Linear
        if isinstance(m, nn.Linear):
            idx = idx + 1 if idx != 0 else idx 
            new_m = QuantLinear_diff(in_features=m.in_features, out_features=m.out_features,input_quant_params= input_quant_params,weight_quant_params= weight_quant_params) 
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)

        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantConv2d_ar(in_channels=m.in_channels, out_channels=m.out_channels,kernel_size=m.kernel_size,stride=m.stride,padding=m.padding,input_quant_params=input_quant_params,weight_quant_params=weight_quant_params)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)

    return model

def set_quant_state(
    model, 
    input_quant=False, 
    weight_quant=False, 
    include_layers=None, 
    exclude_layers=None
):
    """
    动态设置模型中各层的量化状态。
    
    参数：
    - model: 需要操作的模型。
    - input_quant: 是否启用输入量化。
    - weight_quant: 是否启用权重量化。
    - include_layers: 需要设置量化状态的层名称列表（优先级高于 exclude_layers）。
    - exclude_layers: 需要跳过的层名称列表。
    """
    for name, m in model.named_modules():  # 获取每个模块的名称和实例
        if isinstance(m, (QuantLinear_ar,QuantLinear_diff, QuantMatMul, 
                          SmoothLinear_ar,SmoothLinear_diff,QuantLinearWithGrouping,
                          QuantLinearWithGrouping_diff,QuantLinear_scaling,
                          QuantLinear_diff_scale2,QuantLinear_diff_scale,QuantLinear_ar_scale,
                          QuantLinear_ar_outlier,QuantLinear_diff_outlier,QuantLinear_scale_channels,
                          QuantConv2d_ar,QuantConv2d_scale_channels,QuantConv2d_ar_outlier)):
            if include_layers and name not in include_layers:
                continue  # 不在 include_layers 中，跳过
            
            if exclude_layers and name in exclude_layers:
                continue  # 在 exclude_layers 中，跳过
            
            m.set_quant_state(input_quant, weight_quant)