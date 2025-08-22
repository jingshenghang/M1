import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        """RMS Normaliation module

        Args:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# class MambaRMSNormGated(torch.nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states, gate=None):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)

#         if gate is not None:
#             hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

#         return self.weight * hidden_states.to(input_dtype)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-5, norm_before_gate=True, 
                 group_size=None, **factory_kwargs):
        """
        RMSNorm with Gated Mechanism

        Args:
            dim (int): 输入特征维度
            eps (float, optional): 数值稳定性系数. Default: 1e-5
            norm_before_gate (bool): 是否在门控前进行归一化. Default: True
            group_size (int, optional): 分组归一化的组大小. Default: None
            **factory_kwargs: 包含device和dtype的参数字典
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        
        # 分组设置
        if group_size is None:
            groups = 1
            group_size = dim
        else:
            assert dim % group_size == 0, "dim must be divisible by group_size"
            groups = dim // group_size
        
        self.groups = groups
        self.group_size = group_size
        
        # 归一化缩放参数 (每个特征一个参数)
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))
        
        # 门控参数 (分组门控)
        self.gate_weight = nn.Parameter(
            torch.ones(groups, 1, group_size, **factory_kwargs))
        
        # 控制门控信号强度的可学习参数
        self.gate_scale = nn.Parameter(torch.zeros(groups, **factory_kwargs))

    def forward(self, x):
        """
        Input shape: (batch, ..., dim)
        Output shape: same as input
        """
        orig_shape = x.shape
        x = x.view(-1, self.dim)  # 展平非维度部分
        
        # ===== 门控信号处理 =====
        gate = torch.sigmoid(self.gate_scale) * self.gate_weight
        gate = gate.view(1, -1, self.group_size)  # (1, groups, group_size)
        
        if not self.norm_before_gate:
            # 先门控再归一化
            x = x.view(x.size(0), self.groups, self.group_size)
            x = x * gate  # 应用分组门控
            x = x.view(x.size(0), self.dim)
        
        # ===== RMSNorm 核心计算 =====
        # 分组计算均方根
        x_g = x.view(x.size(0), self.groups, self.group_size)
        rms = torch.sqrt(torch.mean(x_g.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # 归一化处理
        x_norm = x_g / rms
        x_norm = x_norm.view(x.size(0), self.dim)
        
        # 应用缩放参数
        output = self.weight * x_norm
        
        # ===== 后门控处理 =====
        if self.norm_before_gate:
            # 先归一化再门控
            output = output.view(output.size(0), self.groups, self.group_size)
            output = output * gate  # 应用分组门控
            output = output.view(output.size(0), self.dim)
        
        # 恢复原始形状
        return output.view(*orig_shape)