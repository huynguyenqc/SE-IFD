import math
import torch
from torch import nn
from typing import List, Optional, Tuple

from deep.base_module import ModuleInterface
from deep.complex_modules.conv import ComplexConv1d
from deep.complex_modules.batchnorm import  ComplexBatchNorm
from deep.lora import LoRAConfigurations


class ComplexGatedActivationUnit(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        residual_dim: int
        gate_dim: int
        skip_dim: int = 128
        kernel_size: int = 3
        dilation: int = 1
        bn_momentum: float = 0.25
        pad_mode: str = 'same'
        with_cond: bool = False
        LoRA: LoRAConfigurations = None

    """ WaveNet-like cell """
    def __init__(self, *args, **kwargs) -> None:
        """ Initialize ComplexWNCell module
        Args:
            residual_dim (int): #channels for residual connection.
            gate_dim (int): #channels for gate connection.
            skip_dim (int): #channels for skip connection.
            kernel_size (int): Size of kernel.
            dilation (int): Dilation rate.
            bn_momentum (float): Momentum of batch norm. layers
            pad_mode (str): Padding mode:
                "same": input and output frame length is same
                "causal": output only depends on current 
                    and previous input frames
            with_cond (bool): Whether or not to use
                conditional variable.
        """
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        residual_dim = self._configs.residual_dim
        gate_dim = self._configs.gate_dim
        skip_dim = self._configs.skip_dim
        kernel_size = self._configs.kernel_size
        dilation = self._configs.dilation
        bn_momentum = self._configs.bn_momentum
        pad_mode = self._configs.pad_mode
        with_cond = self._configs.with_cond
        LoRA = (
            self._configs.LoRA.model_dump() 
            if self._configs.LoRA is not None else None)

        self.hidden_dim = gate_dim
        self.dilation = dilation
        self.with_cond: bool = with_cond

        if self.with_cond:
            self.linear_fuse = ComplexConv1d(2*gate_dim, 2*gate_dim, 1, groups=2, LoRA=LoRA)

        self.in_layer = nn.Sequential(
            ComplexConv1d(residual_dim, 2 * gate_dim, kernel_size,
                          dilation=dilation, pad_mode=pad_mode, LoRA=LoRA),
            ComplexBatchNorm(2 * gate_dim, momentum=bn_momentum))
        self.res_layer = nn.Sequential(
            ComplexConv1d(gate_dim, residual_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            ComplexBatchNorm(residual_dim, momentum=bn_momentum))

        self.skip_layer = nn.Sequential(
            ComplexConv1d(gate_dim, skip_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            ComplexBatchNorm(skip_dim, momentum=bn_momentum))

    def forward(
            self, 
            x_b2ct: torch.Tensor,
            c_b2ct: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculate forward propagation
        Args:
             x_b2ct (Tensor): input variable
        Returns:
            Tensor: Output tensor for residual connection 
                (B, 2, residual_channels, T).
            Tensor: Output tensor for skip connection 
                (B, 2, skip_channels, T).
        """
        a_b2ct = self.in_layer(x_b2ct)

        if self.with_cond:
            assert c_b2ct is not None
            a_b2ct = self.linear_fuse(a_b2ct + c_b2ct)
        tanh_b2ct, sigmoid_b2ct = a_b2ct.chunk(chunks=2, dim=2)
        a_b2ct = tanh_b2ct.tanh() * sigmoid_b2ct.sigmoid()

        skip_b2ct = self.skip_layer(a_b2ct)
        res_b2ct = self.res_layer(a_b2ct)

        return (x_b2ct + res_b2ct) * math.sqrt(0.5), skip_b2ct


class ComplexWaveNet(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: int
        dilation_list: List[int]
        bn_momentum: float = 0.25
        n_stages: int = 1
        pad_mode: str = 'same'
        cond_dim: Optional[int] = None
        LoRA: LoRAConfigurations = None

    def __init__(self, **kwargs) -> None:
        # super(ComplexWaveNet, self).__init__()
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        residual_dim = self._configs.residual_dim
        gate_dim = self._configs.gate_dim
        skip_dim = self._configs.skip_dim
        kernel_size = self._configs.kernel_size
        dilation_list = self._configs.dilation_list
        bn_momentum = self._configs.bn_momentum
        n_stages = self._configs.n_stages
        pad_mode = self._configs.pad_mode
        cond_dim = self._configs.cond_dim
        LoRA = (
            self._configs.LoRA.model_dump() 
            if self._configs.LoRA is not None else None)

        assert n_stages > 0 and len(dilation_list) > 0

        self.units = nn.ModuleList()
        for _ in range(n_stages):
            for d in dilation_list:
                self.units.append(
                    ComplexGatedActivationUnit(
                        residual_dim=residual_dim,
                        gate_dim=gate_dim,
                        skip_dim=skip_dim,
                        kernel_size=kernel_size,
                        dilation=d,
                        bn_momentum=bn_momentum,
                        pad_mode=pad_mode,
                        with_cond=cond_dim is not None,
                        LoRA=LoRA))

        if cond_dim is not None:
            self.cond_layer = nn.Sequential(
                ComplexConv1d(cond_dim, 2*gate_dim*len(self.units), 3, pad_mode=pad_mode),
                ComplexBatchNorm(2*gate_dim*len(self.units), momentum=bn_momentum),
                nn.PReLU())
        else:
            self.cond_layer = None

    def forward(
            self,
            x_b2ct: torch.Tensor,
            c_b2ct: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.cond_layer is not None:
            assert c_b2ct is not None
            c_b2ct = self.cond_layer(c_b2ct)
            list_c_b2ct = torch.chunk(
                c_b2ct, chunks=len(self.units), dim=2)
        else:
            list_c_b2ct = [None] * len(self.units)

        skip_b2ct = 0
        for gau_i, ci_b2ct in zip(self.units, list_c_b2ct):
            x_b2ct, _skip_b2ct = gau_i(x_b2ct, ci_b2ct)
            skip_b2ct = skip_b2ct + _skip_b2ct
        skip_b2ct *= math.sqrt(1.0 / len(self.units))
        return skip_b2ct


def sanity_check():
    net = ComplexGatedActivationUnit(
        residual_dim=4, gate_dim=6, skip_dim=8, kernel_size=3)
    x_b2ct = torch.randn(16, 2, 4, 21)
    res_b2ct, skip_b2ct = net(x_b2ct)
    print(x_b2ct.size(), res_b2ct.size(), skip_b2ct.size())

    net = ComplexGatedActivationUnit(
        residual_dim=4, gate_dim=6, skip_dim=8, kernel_size=3,
        with_cond=True)
    x_b2ct = torch.randn(16, 2, 4, 21)
    c_b2ct = torch.randn(16, 2, 12, 21)
    res_b2ct, skip_b2ct = net(x_b2ct, c_b2ct)
    print(x_b2ct.size(), res_b2ct.size(), skip_b2ct.size())

    net = ComplexWaveNet(
        residual_dim=4,
        gate_dim=6,
        skip_dim=8,
        kernel_size=3,
        dilation_list=[1, 2, 4])
    x_b2ct = torch.randn(16, 2, 4, 21)
    y_b2ct = net(x_b2ct)
    print(x_b2ct.size(), y_b2ct.size())

    net = ComplexWaveNet(
        residual_dim=4,
        gate_dim=6,
        skip_dim=8,
        kernel_size=3,
        dilation_list=[1, 2, 4],
        cond_dim=5)
    x_b2ct = torch.randn(16, 2, 4, 21)
    c_b2ct = torch.randn(16, 2, 5, 21)
    y_b2ct = net(x_b2ct, c_b2ct)
    print(x_b2ct.size(), y_b2ct.size())


if __name__ == '__main__':
    sanity_check()
