from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils.logger_utils import logger1, logger2, logger3, logger4, logger5
device = "cuda" if torch.cuda.is_available() else "cpu"

import math
from typing import Any
import os
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
# import matplotlib.pyplot as plt
import numpy as np
import os
# import pandas as pd
import csv

def linear_forward(self, input: torch.Tensor, step=None, d=False, draw=False, layer_name=None, params=None, save_stats = False, adjustment = False, num = None,a = False,num_bsz = -1,**kwargs) -> torch.Tensor:
    # save_layer_statistics_to_csv(input, layer_name, i, step, num_bsz, "/opt/data/private/GaoJing/deeplearnng/mar/data_comparison")
    return F.linear(input, self.weight, self.bias)

def conv_forward(self, input: torch.Tensor, step=None, d=False, draw=False,
                 layer_name=None, params=None, save_stats=False, adjustment=False, num=None,
                 a=False, num_bsz=-1,**kwargs) -> torch.Tensor:
    """
    Forward function for quantized or standard conv2d layer with optional hooks and stats.
    Parameters like `calib5`, `step`, etc. are passed along for compatibility with external systems.
    """
    return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        
def build_model(model,args):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.forward = MethodType(linear_forward, module)
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            module.forward = MethodType(conv_forward, module)
    return model