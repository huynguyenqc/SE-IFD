import torch
from torch import nn
from typing import Optional, Any
from deep.modules.conv import Conv1d, ConvTranspose1d
from deep.base_module import ModuleInterface
from deep.lora import LoRAConfigurations


class ComplexConv1d(nn.Module, ModuleInterface):
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
        nn.Module.__init__(self)

        in_channels: int = self._configs.in_channels
        out_channels: int = self._configs.out_channels
        kernel_size: int = self._configs.kernel_size
        stride: int = self._configs.stride
        pad_mode: Optional[str] = self._configs.pad_mode
        dilation: int = self._configs.dilation
        groups: int = self._configs.groups
        bias: bool = self._configs.bias
        device = self._configs.device
        dtype = self._configs.dtype
        LoRA = (
            self._configs.LoRA.model_dump() 
            if self._configs.LoRA is not None else None)

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.dilation: int = dilation
        self.groups: int = groups
        self.bias: bool = bias
        self.device = device
        self.dtype = dtype

        self.complex_conv = Conv1d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
            LoRA=LoRA)
            
    def forward(self, x_b2ct: torch.Tensor) -> torch.Tensor:
        batch_size, complex_dim, _, _ = x_b2ct.size()
        assert complex_dim == 2, 'This axis must be 2D!'

        xrxi_bct = x_b2ct.flatten(start_dim=0, end_dim=1)
        yryi_bct = self.complex_conv(xrxi_bct)
        y_b2ct = yryi_bct.unflatten(dim=0, sizes=(batch_size, complex_dim))

        yrr_bct, yri_bct = torch.chunk(y_b2ct[:, 0, ...], chunks=2, dim=1)
        yir_bct, yii_bct = torch.chunk(y_b2ct[:, 1, ...], chunks=2, dim=1)

        yr_bct = yrr_bct - yii_bct
        yi_bct = yri_bct + yir_bct
        return torch.stack((yr_bct, yi_bct), dim=1)


class ComplexConvTranspose1d(nn.Module, ModuleInterface):
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

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        in_channels: int = self._configs.in_channels
        out_channels: int = self._configs.out_channels
        kernel_size: int = self._configs.kernel_size
        stride: int = self._configs.stride
        pad_mode: Optional[str] = self._configs.pad_mode
        dilation: int = self._configs.dilation
        groups: int = self._configs.groups
        bias: bool = self._configs.bias
        device = self._configs.device
        dtype = self._configs.dtype
        LoRA = (
            self._configs.LoRA.model_dump() 
            if self._configs.LoRA is not None else None)

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.dilation: int = dilation
        self.groups: int = groups
        self.bias: bool = bias
        self.device = device
        self.dtype = dtype

        self.complex_conv = ConvTranspose1d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
            LoRA=LoRA)
            
    def forward(self, x_b2ct: torch.Tensor) -> torch.Tensor:
        batch_size, complex_dim, _, _ = x_b2ct.size()
        assert complex_dim == 2, 'This axis must be 2D!'

        xrxi_bct = x_b2ct.flatten(start_dim=0, end_dim=1)
        yryi_bct = self.complex_conv(xrxi_bct)
        y_b2ct = yryi_bct.unflatten(dim=0, sizes=(batch_size, complex_dim))

        yrr_bct, yri_bct = torch.chunk(y_b2ct[:, 0, ...], chunks=2, dim=1)
        yir_bct, yii_bct = torch.chunk(y_b2ct[:, 1, ...], chunks=2, dim=1)

        yr_bct = yrr_bct - yii_bct
        yi_bct = yri_bct + yir_bct
        return torch.stack((yr_bct, yi_bct), dim=1)


def sanity_check_complex_conv():
    net = ComplexConv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        pad_mode='same',
        bias=False,
        LoRA={'r': 1})
    net.complex_conv.weight.data = torch.tensor(
        [[[1, 1, 1]], [[1, -1, 1]]],
        dtype=torch.float32)
    x_b2ct = torch.tensor(
        [[[[1, 1, 2, 4]], [[0, 1, -1, 0]]]], dtype=torch.float32)
    y_b2ct = net(x_b2ct)
    print(x_b2ct)
    print(y_b2ct)

    net = ComplexConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        pad_mode='same',
        bias=False,
        LoRA={'r': 1})
    net.complex_conv.weight.data = torch.tensor(
        [[[1, 1, 1], [1, -1, 1]]],
        dtype=torch.float32)
    x_b2ct = torch.tensor(
        [[[[1, 1, 2, 4]], [[0, 1, -1, 0]]]], dtype=torch.float32)
    y_b2ct = net(x_b2ct)
    print(x_b2ct)
    print(y_b2ct)



if __name__ == '__main__':
    sanity_check_complex_conv()
