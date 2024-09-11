import math
import torch
from torch import nn
from typing import List, Optional, Tuple

from deep.base_module import ModuleInterface
from deep.modules.conv import Conv1d, Conv2dFT
from deep.lora import LoRAConfigurations


class GatedActivationUnit(nn.Module, ModuleInterface):
    """ WaveNet-like cell """
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
        mini_mode: bool = False

    def __init__(self, *args, **kwargs) -> None:
        """ Initialize WNCell module
        Args:
            residual_dim (int): #channels for residual connection.
            gate_dim (int): #channels for gate connection.
            skip_dim (int): #channels for skip connection.
            kernel_size (int): Size of kernel.
            dilation (int): Dilation rate.
            bn_momentum (float): Momentum of batch norm. layers
            pad_mode (str): Padding mode:
                "same": input and output frame length is same,
                "causal": output only depends on current 
                    and previous input frames.
            with_cond (bool): Whether or not to use
                conditional variable.
            LoRA: Configurations for LoRA
            mini_mode (bool): Whether or not to use minimum mode
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
        LoRA = self._configs.LoRA
        mini_mode = self._configs.mini_mode

        if mini_mode:
            skip_res_kernel_size = 1
        else:
            skip_res_kernel_size = kernel_size

        self.hidden_dim: int = gate_dim
        self.dilation: int = dilation
        self.with_cond: bool = with_cond

        if self.with_cond:
            self.linear_fuse = Conv1d(2*gate_dim, 2*gate_dim, 1, groups=2, LoRA=LoRA)

        self.in_layer = nn.Sequential(
            Conv1d(residual_dim, 2 * gate_dim, kernel_size,
                   dilation=dilation, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(2 * gate_dim, momentum=bn_momentum))

        self.res_layer = nn.Sequential(
            Conv1d(gate_dim, residual_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(residual_dim, momentum=bn_momentum))

        self.skip_layer = nn.Sequential(
            Conv1d(gate_dim, skip_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(skip_dim, momentum=bn_momentum))

        self.receptive_field_skip: int = (
            (kernel_size - 1) * dilation + 1 + (skip_res_kernel_size - 1))
        self.receptive_field_res: int = (
            (kernel_size - 1) * dilation + 1 + (skip_res_kernel_size - 1))

    def forward(
            self,
            x_bct: torch.Tensor,
            c_bct: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculate forward propagation
        Args:
             x_bct (Tensor): input variable
             c_bct (Optional[Tensor]): conditional variable
        Returns:
            Tensor: Output tensor for residual connection 
                (B, residual_channels, T).
            Tensor: Output tensor for skip connection
                (B, skip_channels, T).
        """
        a_bct = self.in_layer(x_bct)

        if self.with_cond:
            assert c_bct is not None
            a_bct = self.linear_fuse(a_bct + c_bct)

        tanh_bct, sigmoid_bct = torch.chunk(a_bct, chunks=2, dim=1)
        a_bct = tanh_bct.tanh() * sigmoid_bct.sigmoid()
        skip_bct = self.skip_layer(a_bct)
        res_bct = self.res_layer(a_bct)
        return (x_bct + res_bct) * math.sqrt(0.5), skip_bct


class WaveNet(nn.Module, ModuleInterface):
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
        mini_mode: bool = False

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
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
        LoRA = self._configs.LoRA
        mini_mode = self._configs.mini_mode

        assert n_stages > 0 and len(dilation_list) > 0
        assert cond_dim is None or cond_dim > 0

        self.units = nn.ModuleList()
        for _ in range(n_stages):
            for d in dilation_list:
                self.units.append(
                    GatedActivationUnit(
                        residual_dim=residual_dim,
                        gate_dim=gate_dim,
                        skip_dim=skip_dim,
                        kernel_size=kernel_size,
                        dilation=d,
                        bn_momentum=bn_momentum,
                        pad_mode=pad_mode,
                        with_cond=cond_dim is not None,
                        LoRA=LoRA,
                        mini_mode=mini_mode))

        if cond_dim is not None:
            self.cond_layer = nn.Sequential(
                Conv1d(cond_dim, 2*gate_dim*len(self.units), 3 if not mini_mode else 1,
                       pad_mode=pad_mode, LoRA=LoRA),
                nn.BatchNorm1d(2*gate_dim*len(self.units), momentum=bn_momentum),
                nn.PReLU())
        else:
            self.cond_layer = None

        # Calculate receptive field
        receptive_field_res: int = 0
        receptive_field_skip: int = 0
        for gau_i in self.units:
            receptive_field_skip = max(
                receptive_field_skip,
                receptive_field_res + (gau_i.receptive_field_skip - 1))
            receptive_field_res += (gau_i.receptive_field_res - 1)
        self.receptive_field: int = receptive_field_skip

    def forward(
            self, 
            x_bct: torch.Tensor,
            c_bct: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.cond_layer is not None:
            assert c_bct is not None
            c_bct = self.cond_layer(c_bct)
            list_c_bct = torch.chunk(
                c_bct, chunks=len(self.units), dim=1)
        else:
            list_c_bct = [None] * len(self.units)

        skip_bct = 0
        for gau_i, ci_bct in zip(self.units, list_c_bct):
            x_bct, _skip_bct = gau_i(x_bct, ci_bct)
            skip_bct = skip_bct + _skip_bct
        skip_bct *= math.sqrt(1.0 / len(self.units))
        return skip_bct


class GatedActivationUnit2dFT(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: Tuple[int, int] = (3, 3)
        dilation: Tuple[int, int] = (1, 1)
        bn_momentum: float = 0.25
        pad_mode: str = 'same'
        with_cond: bool = False
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        """ Initialize WNCell module
        Args:
            residual_dim (int): #channels for residual connection.
            gate_dim (int): #channels for gate connection.
            skip_dim (int): #channels for skip connection.
            kernel_size (Tuple[int, int]): Size of kernel.
            dilation (Tuple[int, int]): Dilation rate.
            bn_momentum (float): Momentum of batch norm. layers
            pad_mode (str): Padding mode:
                "same": input and output frame length is same,
                "causal": output only depends on current 
                    and previous input frames.
            with_cond (bool): Whether or not to use
                conditional variable.
            LoRA: Configurations for LoRA
        """
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self._configs: GatedActivationUnit2dFT.ConstructorArgs

        residual_dim = self._configs.residual_dim
        gate_dim = self._configs.gate_dim
        skip_dim = self._configs.skip_dim
        kernel_size = self._configs.kernel_size
        dilation = self._configs.dilation
        bn_momentum = self._configs.bn_momentum
        pad_mode = self._configs.pad_mode
        with_cond = self._configs.with_cond
        LoRA = self._configs.LoRA

        self.hidden_dim: int = gate_dim
        self.dilation: int = dilation
        self.with_cond: bool = with_cond

        if self.with_cond:
            self.linear_fuse = Conv2dFT(2*gate_dim, 2*gate_dim, (1, 1), groups=2, LoRA=LoRA)

        self.in_layer = nn.Sequential(
            Conv2dFT(residual_dim, 2 * gate_dim, kernel_size,
                     dilation=dilation, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm2d(2 * gate_dim, momentum=bn_momentum))

        self.res_layer = nn.Sequential(
            Conv2dFT(gate_dim, residual_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm2d(residual_dim, momentum=bn_momentum))

        self.skip_layer = nn.Sequential(
            Conv2dFT(gate_dim, skip_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm2d(skip_dim, momentum=bn_momentum))

    def forward(
            self,
            x_bcft: torch.Tensor,
            c_bcft: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculate forward propagation
        Args:
            x_bcft (Tensor): input variable
            c_bcft (Optional[Tensor]): conditional variable
        Returns:
            Tensor: Output tensor for residual connection 
                (B, residual_channels, F, T).
            Tensor: Output tensor for skip connection
                (B, skip_channels, F, T).
        """
        a_bcft = self.in_layer(x_bcft)

        if self.with_cond:
            assert c_bcft is not None
            a_bcft = self.linear_fuse(a_bcft + c_bcft)

        tanh_bcft, sigmoid_bcft = torch.chunk(a_bcft, chunks=2, dim=1)
        a_bcft = tanh_bcft.tanh() * sigmoid_bcft.sigmoid()
        skip_bcft = self.skip_layer(a_bcft)
        res_bcft = self.res_layer(a_bcft)
        return (x_bcft + res_bcft) * math.sqrt(0.5), skip_bcft


class WaveNet2dFT(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: Tuple[int, int]
        dilation_list: List[Tuple[int, int]]
        bn_momentum: float = 0.25
        n_stages: int = 1
        pad_mode: str = 'same'
        cond_dim: Optional[int] = None
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self._configs: WaveNet2dFT.ConstructorArgs

        residual_dim = self._configs.residual_dim
        gate_dim = self._configs.gate_dim
        skip_dim = self._configs.skip_dim
        kernel_size = self._configs.kernel_size
        dilation_list = self._configs.dilation_list
        bn_momentum = self._configs.bn_momentum
        n_stages = self._configs.n_stages
        pad_mode = self._configs.pad_mode
        cond_dim = self._configs.cond_dim
        LoRA = self._configs.LoRA

        assert n_stages > 0 and len(dilation_list) > 0
        assert cond_dim is None or cond_dim > 0

        self.units = nn.ModuleList()
        for _ in range(n_stages):
            for d in dilation_list:
                self.units.append(
                    GatedActivationUnit2dFT(
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
                Conv2dFT(cond_dim, 2*gate_dim*len(self.units), (3, 3),
                         pad_mode=pad_mode, LoRA=LoRA),
                nn.BatchNorm2d(2*gate_dim*len(self.units), momentum=bn_momentum),
                nn.PReLU())
        else:
            self.cond_layer = None

    def forward(
            self, 
            x_bcft: torch.Tensor,
            c_bcft: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.cond_layer is not None:
            assert c_bcft is not None
            c_bcft = self.cond_layer(c_bcft)
            list_c_bcft = torch.chunk(
                c_bcft, chunks=len(self.units), dim=1)
        else:
            list_c_bcft = [None] * len(self.units)

        skip_bcft = 0
        for gau_i, ci_bct in zip(self.units, list_c_bcft):
            x_bcft, _skip_bcft = gau_i(x_bcft, ci_bct)
            skip_bcft = skip_bcft + _skip_bcft
        skip_bcft *= math.sqrt(1.0 / len(self.units))
        return skip_bcft


def sanity_check():
    from deep import utils
    net = GatedActivationUnit(
        residual_dim=4, gate_dim=6, skip_dim=8, kernel_size=3)
    x_bct = torch.randn(16, 4, 21)
    res_bct, skip_bct = net(x_bct)
    print(x_bct.size(), res_bct.size(), skip_bct.size())

    net = GatedActivationUnit(
        residual_dim=4, gate_dim=6, skip_dim=8, kernel_size=3,
        with_cond=True)
    x_bct, c_bct = torch.randn(16, 4, 21), torch.randn(16, 12, 21)
    res_bct, skip_bct = net(x_bct, c_bct)
    print(x_bct.size(), res_bct.size(), skip_bct.size())

    net = WaveNet(
        residual_dim=4, gate_dim=6, skip_dim=8,
        kernel_size=3, dilation_list=[1, 2, 4])
    x_bct = torch.randn(16, 4, 21)
    y_bct = net(x_bct)
    print(y_bct.size())
    print(utils.count_parameters(net))

    net = WaveNet(
        residual_dim=4, gate_dim=6, skip_dim=8,
        kernel_size=3, dilation_list=[1, 2, 4],
        cond_dim=5)
    x_bct = torch.randn(16, 4, 21)
    c_bct = torch.randn(16, 5, 21)
    y_bct = net(x_bct, c_bct)
    print(y_bct.size())
    print(utils.count_parameters(net))

    net = WaveNet2dFT(
        residual_dim=4, gate_dim=6, skip_dim=8,
        kernel_size=(3, 3), dilation_list=[(1, 1), (2, 2), (4, 4)],
        cond_dim=5)
    x_bcft = torch.randn(16, 4, 21, 21)
    c_bcft = torch.randn(16, 5, 21, 21)
    y_bcft = net(x_bcft, c_bcft)
    print(y_bcft.size())
    print(utils.count_parameters(net))


if __name__ == '__main__':
    sanity_check()
