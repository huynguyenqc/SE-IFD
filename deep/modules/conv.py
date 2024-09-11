# This package defines customed implementation of convolutional module to be used in speech application.
# Compared to the convolutional module of PyTorch, this implementation is different in
#   - Weight initialisation: Xavier uniform instead of Kaiming uniform
#   - Padding utility: two special mode including `same` and `causal`, where
#       - In both modes, the output dimension is equal to the input dimension divided by `stride`.
#       - `same`: Add padding to the left (past) and right (future) for the same amount
#       - `causal`: Add padding to the left (past) only
# In summary, this implementation is closer to the Tensorflow convolutional module.

import collections
import math
import torch
from itertools import repeat
from torch import nn
from torch.nn import functional as F
from typing import Any, Optional, Tuple

from deep.base_module import ModuleInterface
from deep.lora import LoRAConfigurations, LoRAInterface


# --- From torch.nn.modules.utils ---
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
# -----------------------------------


def receiptive_field(kernel_size: int, dilation: int) -> int:
    return dilation * (kernel_size - 1) + 1


def same_pad(kernel_size: int, stride: int, dilation: int) -> int:
    return (receiptive_field(kernel_size, dilation) - stride) // 2


def causal_pad(kernel_size: int, stride: int, dilation: int) -> int:
    return receiptive_field(kernel_size, dilation) - stride


def get_conv_params(pad_mode: Optional[str], 
                    kernel_size: int, stride: int, dilation: int
) -> Tuple[int, Tuple[int, int]]:
    if pad_mode == 'same':
        pad_len: int = same_pad(kernel_size, stride, dilation)
        idx_slice: Tuple[int, int] = (0, 0)
    elif pad_mode == 'causal':
        pad_len: int = causal_pad(kernel_size, stride, dilation)
        idx_slice: Tuple[int, int] = (0, pad_len // stride)
    else:  # Default convolution
        pad_len: int = 0
        idx_slice: Tuple[int, int] = (0, 0)
    return pad_len, idx_slice


def get_deconv_params(pad_mode: Optional[str], 
                    kernel_size: int, stride: int, dilation: int
) -> Tuple[int, Tuple[int, int]]:
    if pad_mode == 'same':
        pad_len: int = same_pad(kernel_size, stride, dilation)
        idx_slice: Tuple[int, int] = (0, 0)
    elif pad_mode == 'causal':
        pad_len: int = 0
        idx_slice: Tuple[int, int] = 0, causal_pad(kernel_size, stride, dilation)
    else:  # Default convolution
        pad_len: int = 0
        idx_slice: Tuple[int, int] = (0, 0)
    return pad_len, idx_slice


class _Conv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            pad_mode: Optional[str] = None,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            device=None,
            dtype=None,
            **kwargs) -> None:
        
        pad_len, idx_slice = get_conv_params(pad_mode, kernel_size, stride, dilation)
        self._slice: Tuple[int, int] = idx_slice

        super(_Conv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=pad_len,
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        # Copy from nn.Conv1d(...).forward
        y_bct = F.conv1d(
            x_bct, self.weight, self.bias, self.stride, 
            self.padding, self.dilation, self.groups)

        T = y_bct.size(-1)
        y_bct = y_bct[..., self._slice[0]: T - self._slice[1]]
        return y_bct


class Conv1d(_Conv1d, ModuleInterface, LoRAInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        pad_mode: Optional[str] = None
        dilation: int = 1
        groups: int = 1
        bias: bool = True
        device: Any = None
        dtype: Any = None
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        _Conv1d.__init__(self, *args, **kwargs)
        LoRAInterface.__init__(self)

        if self._configs.LoRA is not None:
            in_dim = (self.in_channels // self.groups) * math.prod(self.kernel_size)
            out_dim = self.out_channels
            rank_dim = math.floor(self._configs.LoRA.rank_ratio * self.max_rank_dim(in_dim, out_dim))

            # LoRA parameters
            if rank_dim > 0:
                self.lora_A, self.lora_B = self.new_LoRA_paramters(self.weight, in_dim, out_dim, rank_dim)
                self.scaling = self._configs.LoRA.alpha / rank_dim
                self.reset_LoRA_parameters(self.lora_A, self.lora_B)
            else:
                self.lora_A, self.lora_B, self.scaling = None, None, None

            self.weight.requires_grad_(False)
    
    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        if  self._configs.LoRA is None or self.lora_A is None:
            adapted_weight = self.weight
        else:
            delta_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
            adapted_weight = self.weight + self.scaling * delta_weight
        y_bct = F.conv1d(
            x_bct, adapted_weight, self.bias, self.stride, 
            self.padding, self.dilation, self.groups)
        T = y_bct.size(-1)
        y_bct = y_bct[..., self._slice[0]: T - self._slice[1]]
        return y_bct

    def LoRA_merge_(self) -> None:
        # Update weight by adding adaptation amount
        if  self._configs.LoRA is not None and self.lora_A is not None:
            with torch.no_grad():
                delta_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
                self.weight += self.scaling * delta_weight

            # Reset adaptation amount
            self.reset_LoRA_parameters(self.lora_A, self.lora_B)


class _ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            pad_mode: Optional[str] = None,
            output_padding: int = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: int = 1,
            device=None,
            dtype=None,
            **kwargs) -> None:
        pad_len, idx_slice = get_deconv_params(pad_mode, kernel_size, stride, dilation)
        self._slice: Tuple[int, int] = idx_slice
            
        super(_ConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad_len,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        y_bct = F.conv_transpose1d(
            x_bct, self.weight, self.bias, self.stride, self.padding,
            _single(self.output_padding), self.groups, self.dilation)

        T = y_bct.size(-1)
        y_bct = y_bct[..., self._slice[0]: T - self._slice[1]]
        return y_bct


class ConvTranspose1d(_ConvTranspose1d, ModuleInterface, LoRAInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        pad_mode: Optional[str] = None
        output_padding: int = 0
        groups: int = 1
        bias: bool = True
        dilation: int = 1
        device: Any = None
        dtype: Any = None
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        _ConvTranspose1d.__init__(self, *args, **kwargs)
        LoRAInterface.__init__(self)

        if self._configs.LoRA is not None:
            in_dim = self.in_channels
            out_dim = (self.out_channels // self.groups) * math.prod(self.kernel_size)
            rank_dim = math.floor(self._configs.LoRA.rank_ratio * self.max_rank_dim(in_dim, out_dim))

            # LoRA parameters
            if rank_dim > 0:
                self.lora_A, self.lora_B = self.new_LoRA_paramters(self.weight, in_dim, out_dim, rank_dim)
                self.scaling = self._configs.LoRA.alpha / rank_dim
                self.reset_LoRA_parameters(self.lora_A, self.lora_B)
            else:
                self.lora_A, self.lora_B, self.scaling = None, None, None

            self.weight.requires_grad_(False)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        if  self._configs.LoRA is None or self.lora_A is None:
            adapted_weight = self.weight
        else:
            delta_weight = (self.lora_B @ self.lora_A).t().view(self.weight.size())
            adapted_weight = self.weight + self.scaling * delta_weight
        y_bct = F.conv_transpose1d(
            x_bct, adapted_weight, self.bias, self.stride, self.padding,
            _single(self.output_padding), self.groups, self.dilation)

        T = y_bct.size(-1)
        y_bct = y_bct[..., self._slice[0]: T - self._slice[1]]
        return y_bct

    def LoRA_merge_(self) -> None:
        if  self._configs.LoRA is not None and self.lora_A is not None:
            # Update weight by adding adaptation amount
            with torch.no_grad():
                delta_weight = (self.lora_B @ self.lora_A).t().view(self.weight.shape)
                self.weight += self.scaling * delta_weight

            # Reset adaptation amount
            self.reset_LoRA_parameters(self.lora_A, self.lora_B)


class _Conv2dFT(nn.Conv2d):
    """ 2D convolution layer on time-frequency representation """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1, 1),
            pad_mode: Optional[str] = None,
            dilation: Tuple[int, int] = (1, 1),
            groups: int = 1,
            bias: bool = True,
            device=None,
            dtype=None,
            **kwargs) -> None:
        
        pad_len_f, idx_slice_f = get_conv_params('same', kernel_size[0], stride[0], dilation[0])
        pad_len_t, idx_slice_t = get_conv_params(pad_mode, kernel_size[1], stride[1], dilation[1])
        self._slice_f: Tuple[int, int] = idx_slice_f
        self._slice_t: Tuple[int, int] = idx_slice_t

        super(_Conv2dFT, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=(pad_len_f, pad_len_t),
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bcft: torch.Tensor) -> torch.Tensor:
        # Copy from nn.Conv1d(...).forward
        y_bcft = F.conv2d(
            x_bcft, self.weight, self.bias, self.stride, 
            self.padding, self.dilation, self.groups)

        T_t = y_bcft.size(-1)
        T_f = y_bcft.size(-2)
        y_bcft = y_bcft[..., 
                        self._slice_f[0]: T_f - self._slice_f[1],
                        self._slice_t[0]: T_t - self._slice_t[1]]
        return y_bcft


class Conv2dFT(_Conv2dFT, ModuleInterface, LoRAInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        in_channels: int
        out_channels: int
        kernel_size: Tuple[int, int]
        stride: Tuple[int, int] = (1, 1)
        pad_mode: Optional[str] = None
        dilation: Tuple[int, int] = (1, 1)
        groups: int = 1
        bias: bool = True
        device: Any = None
        dtype: Any = None
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        _Conv2dFT.__init__(self, *args, **kwargs)
        LoRAInterface.__init__(self)
        self._configs: Conv2dFT.ConstructorArgs

        if self._configs.LoRA is not None:
            in_dim = (self.in_channels // self.groups) * math.prod(self.kernel_size)
            out_dim = self.out_channels
            rank_dim = math.floor(self._configs.LoRA.rank_ratio * self.max_rank_dim(in_dim, out_dim))

            # LoRA parameters
            if rank_dim > 0:
                self.lora_A, self.lora_B = self.new_LoRA_paramters(self.weight, in_dim, out_dim, rank_dim)
                self.scaling = self._configs.LoRA.alpha / rank_dim
                self.reset_LoRA_parameters(self.lora_A, self.lora_B)
            else:
                self.lora_A, self.lora_B, self.scaling = None, None, None

            self.weight.requires_grad_(False)

    def forward(self, x_bcft: torch.Tensor) -> torch.Tensor:
        if  self._configs.LoRA is None or self.lora_A is None:
            adapted_weight = self.weight
        else:
            delta_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
            adapted_weight = self.weight + self.scaling * delta_weight
        y_bcft = F.conv2d(
            x_bcft, adapted_weight, self.bias, self.stride, 
            self.padding, self.dilation, self.groups)

        T_t = y_bcft.size(-1)
        T_f = y_bcft.size(-2)
        y_bcft = y_bcft[..., 
                        self._slice_f[0]: T_f - self._slice_f[1],
                        self._slice_t[0]: T_t - self._slice_t[1]]
        return y_bcft

    def LoRA_merge_(self) -> None:
        # Update weight by adding adaptation amount
        if  self._configs.LoRA is not None and self.lora_A is not None:
            with torch.no_grad():
                delta_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
                self.weight += self.scaling * delta_weight

            # Reset adaptation amount
            self.reset_LoRA_parameters(self.lora_A, self.lora_B)


class _ConvTranspose2dFT(nn.ConvTranspose2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1, 1),
            pad_mode: Optional[str] = None,
            output_padding: Tuple[int, int] = (0, 0),
            groups: int = 1,
            bias: bool = True,
            dilation: Tuple[int, int] = (1, 1),
            device=None,
            dtype=None,
            **kwargs) -> None:
        pad_len_f, idx_slice_f = get_deconv_params('same', kernel_size[0], stride[0], dilation[0])
        pad_len_t, idx_slice_t = get_deconv_params(pad_mode, kernel_size[1], stride[1], dilation[1])
        self._slice_f: Tuple[int, int] = idx_slice_f
        self._slice_t: Tuple[int, int] = idx_slice_t
            
        super(_ConvTranspose2dFT, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(pad_len_f, pad_len_t),
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bcft: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        y_bcft = F.conv_transpose2d(
            x_bcft, self.weight, self.bias, self.stride, self.padding,
            _single(self.output_padding), self.groups, self.dilation)

        T_t = y_bcft.size(-1)
        T_f = y_bcft.size(-2)
        y_bcft = y_bcft[..., 
                        self._slice_f[0]: T_f - self._slice_f[1],
                        self._slice_t[0]: T_t - self._slice_t[1]]
        return y_bcft


class ConvTranspose2dFT(_ConvTranspose2dFT, ModuleInterface, LoRAInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        in_channels: int
        out_channels: int
        kernel_size: Tuple[int, int]
        stride: Tuple[int, int] = (1, 1)
        pad_mode: Optional[str] = None
        output_padding: Tuple[int, int] = (0, 0)
        groups: int = 1
        bias: bool = True
        dilation: Tuple[int, int] = (1, 1)
        device: Any = None
        dtype: Any = None
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        _ConvTranspose2dFT.__init__(self, *args, **kwargs)
        LoRAInterface.__init__(self)

        if self._configs.LoRA is not None:
            in_dim = self.in_channels
            out_dim = (self.out_channels // self.groups) * math.prod(self.kernel_size)
            rank_dim = math.floor(self._configs.LoRA.rank_ratio * self.max_rank_dim(in_dim, out_dim))

            # LoRA parameters
            if rank_dim > 0:
                self.lora_A, self.lora_B = self.new_LoRA_paramters(self.weight, in_dim, out_dim, rank_dim)
                self.scaling = self._configs.LoRA.alpha / rank_dim
                self.reset_LoRA_parameters(self.lora_A, self.lora_B)
            else:
                self.lora_A, self.lora_B, self.scaling = None, None, None

            self.weight.requires_grad_(False)

    def forward(self, x_bcft: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        if  self._configs.LoRA is None or self.lora_A is None:
            adapted_weight = self.weight
        else:
            delta_weight = (self.lora_B @ self.lora_A).t().view(self.weight.size())
            adapted_weight = self.weight + self.scaling * delta_weight
        y_bcft = F.conv_transpose2d(
            x_bcft, adapted_weight, self.bias, self.stride, self.padding,
            _single(self.output_padding), self.groups, self.dilation)

        T_t = y_bcft.size(-1)
        T_f = y_bcft.size(-2)
        y_bcft = y_bcft[..., 
                        self._slice_f[0]: T_f - self._slice_f[1],
                        self._slice_t[0]: T_t - self._slice_t[1]]
        return y_bcft

    def LoRA_merge_(self) -> None:
        if  self._configs.LoRA is not None and self.lora_A is not None:
            # Update weight by adding adaptation amount
            with torch.no_grad():
                delta_weight = (self.lora_B @ self.lora_A).t().view(self.weight.shape)
                self.weight += self.scaling * delta_weight

            # Reset adaptation amount
            self.reset_LoRA_parameters(self.lora_A, self.lora_B)


def sanity_check_conv():
    print('# Same convolution')
    net_configs = {
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'same'}
    net = Conv1d(**net_configs)
    net.weight.data = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[13, 17]]]')
    x_bct = torch.tensor([[[1, 1, 2, 4]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# Causal convolution')
    net_configs = {
        'in_channels': 1, 
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'causal'}
    net = Conv1d(**net_configs)
    net.weight.data = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[7, 25]]]')
    x_bct = torch.tensor([[[1, 1, 2, 4]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# Same deconvolution')
    net_configs = {
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'same'}
    net = ConvTranspose1d(**net_configs)
    net.weight.data = torch.tensor([[[-35/289, 1/13, 4/17, 9/169]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[1, 1, 2, 4]]]')
    x_bct = torch.tensor([[[13, 17]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# Causal deconvolution')
    net_configs = {
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'causal'}
    net = ConvTranspose1d(**net_configs)
    net.weight.data = torch.tensor([[[11 / 625, -3 / 625, 2 / 25, 4 / 25]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[0.1232, -0.0336, 1.0, 1.0]]]')
    x_bct = torch.tensor([[[7, 25]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# Causal convolution + LoRA')
    net = Conv1d(10, 10, 4, stride=2, pad_mode='causal', LoRA={'rank_ratio': 1.0})
    x_bct = torch.randn(1, 10, 8)
    y_bct = net(x_bct)
    print('  - Input size: ', x_bct.size())
    print('  - Output size: ', y_bct.size())
    net.LoRA_merge_()

    print('# Causal deconvolution + LoRA')
    net = ConvTranspose1d(10, 10, 4, stride=2, pad_mode='causal', LoRA={'rank_ratio': 1.0})
    x_bct = torch.randn(1, 10, 8)
    y_bct = net(x_bct)
    print('  - Input size: ', x_bct.size())
    print('  - Output size: ', y_bct.size())
    net.LoRA_merge_()

    print('# Convolution 2d')
    net = Conv2dFT(1, 1, (4, 4), stride=(2, 2), pad_mode='causal')
    x_bcft = torch.randn(1, 1, 14, 14)
    y_bcft = net(x_bcft)
    print('  - Input size: ', x_bcft.size())
    print('  - Output size: ', y_bcft.size())

    print('# Deconvolution 2d')
    net = ConvTranspose2dFT(1, 1, (4, 4), stride=(2, 2), pad_mode='causal')
    x_bcft = torch.randn(1, 1, 7, 7)
    y_bcft = net(x_bcft)
    print('  - Input size: ', x_bcft.size())
    print('  - Output size: ', y_bcft.size())


if __name__ == '__main__':
    sanity_check_conv()
