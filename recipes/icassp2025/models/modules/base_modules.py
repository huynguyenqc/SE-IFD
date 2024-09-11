import math
import torch
from torch import nn
from typing import List, Optional, Tuple
from deep.base_module import ModuleInterface
from deep.lora import LoRAConfigurations
from deep.modules.conv import Conv1d, ConvTranspose1d, Conv2dFT, ConvTranspose2dFT
from deep.modules.wavenet import WaveNet, WaveNet2dFT
from deep.complex_modules.batchnorm import ComplexBatchNorm
from deep.complex_modules.conv import ComplexConv1d
from deep.complex_modules.recurrent import ComplexSplitLSTM
from deep.complex_modules.wavenet import ComplexWaveNet


class Encoder(nn.Module, ModuleInterface):
    """ WaveNet-based encoder module """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        input_dim: int
        output_dim: int
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: int
        dilation_list: List[int]
        bn_momentum: float = 0.25
        n_stages: int = 1
        bn_momentum_conv: float = 0.8
        pad_mode: str = 'same'
        down_sample_factor: int = 2
        cond_dim: Optional[int] = None
        LoRA: LoRAConfigurations = None
        mini_mode: bool = False

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            kernel_size: int,
            dilation_list: List[int],
            bn_momentum: float = 0.25,
            n_stages: int = 1,
            bn_momentum_conv: float = 0.8,
            pad_mode: str = 'same',
            down_sample_factor: int = 2,
            cond_dim: Optional[int] = None,
            LoRA = None,
            mini_mode: bool = False) -> None:
        ModuleInterface.__init__(
            self, input_dim, output_dim, residual_dim, gate_dim, skip_dim, kernel_size,
            dilation_list, bn_momentum, n_stages, bn_momentum_conv, pad_mode, 
            down_sample_factor, cond_dim, LoRA, mini_mode)
        nn.Module.__init__(self)

        in_kernel = 5 if not mini_mode else 1
        out_kernel = kernel_size if not mini_mode else 1

        self.in_layer = nn.Sequential(
            Conv1d(input_dim, residual_dim, in_kernel, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(residual_dim, momentum=bn_momentum_conv),
            nn.PReLU(residual_dim))

        assert down_sample_factor > 0, 'Down-sampling rate must be positive integer!'
        if down_sample_factor > 1:
            receptive_width = (4 + (down_sample_factor % 2)) * down_sample_factor 

            self.re_sampler = nn.Sequential(
                Conv1d(residual_dim, residual_dim, kernel_size=receptive_width,
                       stride=down_sample_factor, pad_mode=pad_mode,
                       LoRA=LoRA),
                nn.BatchNorm1d(residual_dim, momentum=bn_momentum_conv),
                nn.PReLU(residual_dim))
        else:
            self.re_sampler = None

        self.wn = WaveNet(
            residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim, 
            kernel_size=kernel_size, dilation_list=dilation_list, bn_momentum=bn_momentum, 
            n_stages=n_stages, pad_mode=pad_mode, cond_dim=cond_dim, LoRA=LoRA,
            mini_mode=mini_mode)

        self.out_layer = nn.Sequential(
            Conv1d(skip_dim, output_dim, out_kernel, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(output_dim, momentum=bn_momentum_conv),
            nn.PReLU(output_dim),
            Conv1d(output_dim, output_dim, 1, pad_mode=pad_mode, LoRA=LoRA))

        # Calculate receptive field
        ## After input layer
        if down_sample_factor > 1:
            self.resample_rate = down_sample_factor
            self.receptive_field = (
                in_kernel
                + (receptive_width - 1)
                + (self.wn.receptive_field - 1) * down_sample_factor
                + (out_kernel - 1) * down_sample_factor)
        else:
            self.resample_rate = 1
            self.receptive_field = (
                in_kernel
                + (self.wn.receptive_field - 1)
                + (out_kernel - 1))

    def forward(self, x_bct: torch.Tensor, c_bct: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_bct = self.in_layer(x_bct)
        if self.re_sampler is not None:
            h_bct = self.re_sampler(h_bct)
        h_bct = self.wn(h_bct, c_bct)
        y_bct = self.out_layer(h_bct)
        return y_bct


class Decoder(nn.Module, ModuleInterface):
    """ WaveNet-based decoder module """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        input_dim: int
        output_dim: int
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: int
        dilation_list: List[int]
        bn_momentum: float = 0.25
        n_stages: int = 1
        bn_momentum_conv: float = 0.25
        pad_mode: str = 'same'
        up_sample_factor: int = 2
        cond_dim: Optional[int] = None
        LoRA: LoRAConfigurations = None
        mini_mode: bool = False

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            kernel_size: int,
            dilation_list: List[int],
            bn_momentum: float = 0.25,
            n_stages: int = 1,
            bn_momentum_conv: float = 0.25,
            pad_mode: str = 'same',
            up_sample_factor: int = 2,
            cond_dim: Optional[int] = None,
            LoRA = None,
            mini_mode: bool = False) -> None:
        ModuleInterface.__init__(
            self, input_dim, output_dim, residual_dim, gate_dim, skip_dim, kernel_size,
            dilation_list, bn_momentum, n_stages, bn_momentum_conv, pad_mode, up_sample_factor, cond_dim, LoRA)
        nn.Module.__init__(self)

        in_kernel = kernel_size if not mini_mode else 1
        out_kernel_1 = kernel_size if not mini_mode else 1
        out_kernel_2 = 15 if not mini_mode else 1

        self.in_layer = nn.Sequential(
            Conv1d(input_dim, 2*residual_dim, in_kernel, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(2*residual_dim, momentum=bn_momentum_conv),
            nn.GLU(dim=1))

        assert up_sample_factor > 0, 'Down-sampling rate must be positive integer!'
        if up_sample_factor > 1:
            receptive_width = (4 + (up_sample_factor % 2)) * up_sample_factor 

            self.re_sampler = nn.Sequential(
                ConvTranspose1d(residual_dim, 2*residual_dim, kernel_size=receptive_width,
                                stride=up_sample_factor, pad_mode=pad_mode, LoRA=LoRA),
                nn.BatchNorm1d(2*residual_dim, momentum=bn_momentum_conv),
                nn.GLU(dim=1))
        else:
            self.re_sampler = None

        self.wn = WaveNet(
            residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim,
            kernel_size=kernel_size, dilation_list=dilation_list, bn_momentum=bn_momentum,
            n_stages=n_stages, pad_mode=pad_mode, cond_dim=cond_dim, LoRA=LoRA,
            mini_mode=mini_mode)

        self.out_layer = nn.Sequential(
            Conv1d(skip_dim, 2*skip_dim, out_kernel_1, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(2*skip_dim, momentum=bn_momentum_conv),
            nn.GLU(dim=1),
            Conv1d(skip_dim, output_dim, out_kernel_2, pad_mode=pad_mode, LoRA=LoRA))

        # Calculate receptive field
        if up_sample_factor > 1:
            self.resample_rate = up_sample_factor
            self.receptive_field = in_kernel + math.ceil((
                self.wn.receptive_field
                + out_kernel_1 - 1
                + out_kernel_2 - 1) / up_sample_factor) - 1
        else:
            self.resample_rate = 1
            self.receptive_field = (
                in_kernel
                + self.wn.receptive_field - 1
                + out_kernel_1 - 1
                + out_kernel_2 - 1)

    def forward(self, x_bct: torch.Tensor, c_bct: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_bct = self.in_layer(x_bct)
        if self.re_sampler is not None:
            h_bct = self.re_sampler(h_bct)
        h_bct = self.wn(h_bct, c_bct)
        y_bct = self.out_layer(h_bct)
        return y_bct


class Encoder2dFT(nn.Module, ModuleInterface):
    """ WaveNet-based encoder module """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        input_dim: int
        output_dim: int
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: Tuple[int, int]
        dilation_list: List[Tuple[int, int]]
        bn_momentum: float = 0.25
        n_stages: int = 1
        bn_momentum_conv: float = 0.8
        pad_mode: str = 'same'
        down_sample_factor: Tuple[int, int] = (2, 1)
        cond_dim: Optional[int] = None
        LoRA: LoRAConfigurations = None

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            kernel_size: Tuple[int, int],
            dilation_list: List[Tuple[int, int]],
            bn_momentum: float = 0.25,
            n_stages: int = 1,
            bn_momentum_conv: float = 0.8,
            pad_mode: str = 'same',
            down_sample_factor: Tuple[int, int] = (2, 1),
            cond_dim: Optional[int] = None,
            LoRA = None) -> None:
        ModuleInterface.__init__(
            self, input_dim, output_dim, residual_dim, gate_dim, skip_dim, kernel_size,
            dilation_list, bn_momentum, n_stages, bn_momentum_conv, pad_mode, down_sample_factor, cond_dim, LoRA)
        nn.Module.__init__(self)

        assert min(down_sample_factor) > 0, 'Down-sampling rate must be positive integer!'
        if max(down_sample_factor) > 1:
            receptive_width = tuple(
                (1 if dsf_i == 1 else ((4 + (dsf_i % 2)) * dsf_i))
                for dsf_i in down_sample_factor)

            self.re_sampler = nn.Sequential(
                Conv2dFT(input_dim, residual_dim, receptive_width,
                         stride=down_sample_factor, pad_mode=pad_mode,
                         LoRA=LoRA),
                nn.BatchNorm2d(residual_dim, momentum=bn_momentum_conv),
                nn.PReLU(residual_dim))
        else:
            self.re_sampler = nn.Sequential(
                Conv2dFT(input_dim, residual_dim, (5, 5), pad_mode=pad_mode, LoRA=LoRA),
                nn.BatchNorm2d(residual_dim, momentum=bn_momentum_conv),
                nn.PReLU(residual_dim))

        self.wn = WaveNet2dFT(
            residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim, 
            kernel_size=kernel_size, dilation_list=dilation_list, bn_momentum=bn_momentum, 
            n_stages=n_stages, pad_mode=pad_mode, cond_dim=cond_dim, LoRA=LoRA)

        self.out_layer = nn.Sequential(
            Conv2dFT(skip_dim, output_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm2d(output_dim, momentum=bn_momentum_conv),
            nn.PReLU(output_dim),
            Conv2dFT(output_dim, output_dim, (1, 1), pad_mode=pad_mode, LoRA=LoRA))

    def forward(self, x_bcft: torch.Tensor, c_bcft: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_bcft = self.re_sampler(x_bcft)
        h_bcft = self.wn(h_bcft, c_bcft)
        y_bcft = self.out_layer(h_bcft)
        return y_bcft


class Decoder2dFT(nn.Module, ModuleInterface):
    """ WaveNet-based decoder module """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        input_dim: int
        output_dim: int
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: Tuple[int, int]
        dilation_list: List[Tuple[int, int]]
        bn_momentum: float = 0.25
        n_stages: int = 1
        bn_momentum_conv: float = 0.25
        pad_mode: str = 'same'
        up_sample_factor: Tuple[int, int] = (2, 1)
        cond_dim: Optional[int] = None
        LoRA: LoRAConfigurations = None

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            kernel_size: Tuple[int, int],
            dilation_list: List[Tuple[int, int]],
            bn_momentum: float = 0.25,
            n_stages: int = 1,
            bn_momentum_conv: float = 0.25,
            pad_mode: str = 'same',
            up_sample_factor: Tuple[int, int] = (2, 1),
            cond_dim: Optional[int] = None,
            LoRA = None) -> None:
        ModuleInterface.__init__(
            self, input_dim, output_dim, residual_dim, gate_dim, skip_dim, kernel_size,
            dilation_list, bn_momentum, n_stages, bn_momentum_conv, pad_mode, up_sample_factor, cond_dim, LoRA)
        nn.Module.__init__(self)

        self.input_dim = input_dim

        assert min(up_sample_factor) > 0, 'Down-sampling rate must be positive integer!'
        if max(up_sample_factor) > 1:
            receptive_width = tuple(
                (1 if usf_i == 1 else ((4 + (usf_i % 2)) * usf_i))
                for usf_i in up_sample_factor)

            self.re_sampler = nn.Sequential(
                ConvTranspose2dFT(input_dim, 2*residual_dim, receptive_width,
                                  stride=up_sample_factor, pad_mode=pad_mode, LoRA=LoRA),
                nn.BatchNorm2d(2*residual_dim, momentum=bn_momentum_conv),
                nn.GLU(dim=1))
        else:
            self.re_sampler = nn.Sequential(
                Conv2dFT(input_dim, 2*residual_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
                nn.BatchNorm2d(2*residual_dim, momentum=bn_momentum_conv),
                nn.GLU(dim=1))

        self.wn = WaveNet2dFT(
            residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim,
            kernel_size=kernel_size, dilation_list=dilation_list, bn_momentum=bn_momentum,
            n_stages=n_stages, pad_mode=pad_mode, cond_dim=cond_dim, LoRA=LoRA)

        self.out_layer = nn.Sequential(
            Conv2dFT(skip_dim, 2*skip_dim, kernel_size, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm2d(2*skip_dim, momentum=bn_momentum_conv),
            nn.GLU(dim=1),
            Conv2dFT(skip_dim, output_dim, (15, 15), pad_mode=pad_mode, LoRA=LoRA))

    def forward(self, x_bcft: torch.Tensor, c_bcft: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_bcft = self.re_sampler(x_bcft)
        h_bcft = self.wn(h_bcft, c_bcft)
        y_bcft = self.out_layer(h_bcft)
        return y_bcft


class ComplexCRNN(nn.Module, ModuleInterface):
    """ WaveNet-baesd complex-valued convolutional recurrent neural network """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        dim: int
        hidden_dim: int
        kernel_size: int
        dilation_list: List[int]
        wavenet_bn_momentum: float = 0.25
        n_stages: int = 1
        pad_mode: str = 'causal'
        cond_dim: Optional[int] = None
        bn_momentum_conv_in: float = 0.8
        bn_momentum_conv_out: float = 0.25
        n_rnn_layers: int = 1 
        LoRA: LoRAConfigurations = None

    def __init__(
            self,
            dim: int, 
            hidden_dim: int, 
            kernel_size: int, 
            dilation_list: List[int], 
            wavenet_bn_momentum: float = 0.25, 
            n_stages: int = 1, 
            pad_mode: str = 'causal', 
            cond_dim: Optional[int] = None, 
            bn_momentum_conv_in: float = 0.8,
            bn_momentum_conv_out: float = 0.25,
            n_rnn_layers: int = 1,
            LoRA = None
    ) -> None:
        ModuleInterface.__init__(
            self, dim, hidden_dim, kernel_size, dilation_list, wavenet_bn_momentum, n_stages, 
            pad_mode, cond_dim, bn_momentum_conv_in, bn_momentum_conv_out, n_rnn_layers, LoRA)
        nn.Module.__init__(self)

        self.in_layer = nn.Sequential(
            ComplexConv1d(dim, hidden_dim, kernel_size=5, pad_mode=pad_mode, LoRA=LoRA),
            ComplexBatchNorm(hidden_dim, momentum=bn_momentum_conv_in),
            nn.PReLU())

        self.wn = ComplexWaveNet(
            residual_dim=hidden_dim, gate_dim=hidden_dim, skip_dim=hidden_dim,
            kernel_size=kernel_size, dilation_list=dilation_list,
            bn_momentum=wavenet_bn_momentum, n_stages=n_stages, pad_mode=pad_mode, 
            cond_dim=cond_dim, LoRA=LoRA)

        self.rnn = ComplexSplitLSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_rnn_layers)

        self.out_layer = nn.Sequential(
            ComplexConv1d(hidden_dim, dim, kernel_size=3, pad_mode=pad_mode, LoRA=LoRA),
            ComplexBatchNorm(dim, momentum=bn_momentum_conv_out),
            nn.Tanh())

    def forward(self, x_b2ct: torch.Tensor, c_b2ct: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_b2ct = self.in_layer(x_b2ct)
        x_b2ct = self.wn(x_b2ct, c_b2ct)
        x_b2tc = x_b2ct.transpose(-1, -2)
        x_b2tc = self.rnn(x_b2tc)
        x_b2ct = x_b2tc.transpose(-1, -2)
        x_b2ct = self.out_layer(x_b2ct)
        return x_b2ct


class ComplexCRNNScalingOutput(nn.Module, ModuleInterface):
    """ WaveNet-baesd complex-valued convolutional recurrent neural network """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        dim: int
        hidden_dim: int
        kernel_size: int
        dilation_list: List[int]
        wavenet_bn_momentum: float = 0.25
        n_stages: int = 1
        pad_mode: str = 'causal'
        cond_dim: Optional[int] = None
        bn_momentum_conv_in: float = 0.8
        bn_momentum_conv_out: float = 0.25
        n_rnn_layers: int = 1 
        mini_mode: bool = False
        LoRA: LoRAConfigurations = None
        
    def __init__(
            self,
            dim: int, 
            hidden_dim: int, 
            kernel_size: int, 
            dilation_list: List[int], 
            wavenet_bn_momentum: float = 0.25, 
            n_stages: int = 1, 
            pad_mode: str = 'causal', 
            cond_dim: Optional[int] = None, 
            bn_momentum_conv_in: float = 0.8,
            bn_momentum_conv_out: float = 0.25,
            n_rnn_layers: int = 1,
            mini_mode: bool = False,
            LoRA = None
    ) -> None:
        ModuleInterface.__init__(
            self, dim, hidden_dim, kernel_size, dilation_list, wavenet_bn_momentum, n_stages, 
            pad_mode, cond_dim, bn_momentum_conv_in, bn_momentum_conv_out, n_rnn_layers, mini_mode,
            LoRA)
        nn.Module.__init__(self)

        self.in_layer = nn.Sequential(
            ComplexConv1d(dim, hidden_dim, kernel_size=5, pad_mode=pad_mode, LoRA=LoRA),
            ComplexBatchNorm(hidden_dim, momentum=bn_momentum_conv_in),
            nn.PReLU())

        self.wn = ComplexWaveNet(
            residual_dim=hidden_dim, gate_dim=hidden_dim, skip_dim=hidden_dim,
            kernel_size=kernel_size, dilation_list=dilation_list,
            bn_momentum=wavenet_bn_momentum, n_stages=n_stages, pad_mode=pad_mode, 
            cond_dim=cond_dim, LoRA=LoRA)

        self.rnn = ComplexSplitLSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_rnn_layers)

        self.scaling_out_layer = nn.Sequential(
            Conv1d(hidden_dim, dim, kernel_size=3, pad_mode=pad_mode, LoRA=LoRA),
            nn.BatchNorm1d(dim, momentum=bn_momentum_conv_out),
            nn.LeakyReLU())

        self.mini_mode: bool = mini_mode
        if self.mini_mode:
            self.shifting_out_layer = nn.Sequential(
                Conv1d(2*hidden_dim, dim, kernel_size=3, pad_mode=pad_mode, LoRA=LoRA),
                nn.BatchNorm1d(dim, momentum=bn_momentum_conv_out))
        else:
            self.shifting_out_layer = nn.Sequential(
                ComplexConv1d(hidden_dim, dim, kernel_size=3, pad_mode=pad_mode, LoRA=LoRA),
                ComplexBatchNorm(dim, momentum=bn_momentum_conv_out))

    def forward(self, x_b2ct: torch.Tensor, c_b2ct: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_b2ct = self.in_layer(x_b2ct)
        x_b2ct: torch.Tensor = self.wn(x_b2ct, c_b2ct)
        x_b2tc = x_b2ct.transpose(-1, -2)
        x_b2tc: torch.Tensor = self.rnn(x_b2tc)
        x_b2ct = x_b2tc.transpose(-1, -2)

        amp_x_bct = x_b2ct.square().sum(dim=1).clamp(1e-12).log()
        alpha_bct: torch.Tensor = self.scaling_out_layer(amp_x_bct)
        alpha_bct = (-alpha_bct).clamp(math.log(1e-12), math.log(2.0)).exp()

        if self.mini_mode:
            zeta_bct: torch.Tensor  = self.shifting_out_layer(x_b2ct.flatten(start_dim=1, end_dim=2))
            zeta_bct = zeta_bct.tanh().clamp(1e-5 - 1.0, 1.0 - 1e-5)
            zeta_b2ct = torch.stack([(1 - zeta_bct.square()).sqrt(), zeta_bct], dim=1)
        else:
            zeta_b2ct: torch.Tensor  = self.shifting_out_layer(x_b2ct)
            zeta_b2ct = torch.stack([zeta_b2ct[:, 0, :, :].sigmoid(), zeta_b2ct[:, 1, :, :].tanh()], dim=1)
        return alpha_bct, zeta_b2ct


class SimpleCRNN(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        dim: int
        hidden_dim: int
        kernel_size: int = 3
        n_rnn_layers: int = 2
        pad_mode: str = 'causal'
        bn_momentum_conv_in: float = 0.8
        bn_momentum_conv_out: float = 0.25
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self._configs: SimpleCRNN.ConstructorArgs

        self.in_layer = nn.Sequential(
            Conv1d(self._configs.dim, self._configs.hidden_dim, self._configs.kernel_size, 
                   pad_mode=self._configs.pad_mode, LoRA=self._configs.LoRA),
            nn.BatchNorm1d(self._configs.hidden_dim, momentum=self._configs.bn_momentum_conv_in),
            nn.PReLU())

        self.rnn_layer = nn.LSTM(
            self._configs.hidden_dim, self._configs.hidden_dim, self._configs.n_rnn_layers,
            batch_first=True, bidirectional=False)

        self.out_layer = nn.Sequential(
            Conv1d(self._configs.hidden_dim, 2*self._configs.dim, self._configs.kernel_size, 
                   pad_mode=self._configs.pad_mode, LoRA=self._configs.LoRA),
            nn.BatchNorm1d(2*self._configs.dim, momentum=self._configs.bn_momentum_conv_out),
            nn.GLU(dim=1),
            Conv1d(self._configs.dim, self._configs.dim, self._configs.kernel_size, 
                   pad_mode=self._configs.pad_mode, LoRA=self._configs.LoRA))

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        x_bct = self.in_layer(x_bct)
        x_btc = x_bct.transpose(-1, -2)
        x_btc, _ = self.rnn_layer(x_btc)
        x_btc: torch.Tensor
        x_bct = x_btc.transpose(-1, -2)
        x_bct = self.out_layer(x_bct)
        return x_bct
