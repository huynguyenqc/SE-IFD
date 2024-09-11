import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack as scp_fft
from scipy import signal as scp_sig
from typing import Tuple, Optional

from deep.base_module import ModuleInterface


def init_rfft_kernels(
        win_len: int,
        n_fft: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize Real Fast Fourier Transform (RFFT) kernels.

    This function initializes the RFFT kernels that are used for performing
    the real-valued FFT. It returns a tuple of torch tensors representing
    the direct and inverse FFT kernels.

    Args:
        win_len (int): The window length for the FFT.
        n_fft (int): The number of FFT points.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the forward
        and inverse FFT kernels.
    """

    N = n_fft
    M = win_len
    K = n_fft // 2 + 1

    Vc_nk = np.fft.rfft(np.eye(N))
    Vr_nk = np.real(Vc_nk)
    Vi_nk = np.imag(Vc_nk)

    # iV_md @ V_dm = eye(M)
    V_dm = np.concatenate(
        (Vr_nk.T[:K, :M], Vi_nk.T[:K, :M]), axis=0)

    c_1d = np.array(
        ([1] + [2] * (K-2) + [1]) * 2
    )[None, :] / N
    iV_md = np.concatenate(
        (Vr_nk[:M, :K], Vi_nk[:M, :K]), axis=1) * c_1d

    return (
        torch.from_numpy(V_dm.astype(np.float32)),
        torch.from_numpy(iV_md.astype(np.float32)))


def init_stft_kernels(
        win_len: int, 
        n_fft: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize Short-Time Fourier Transform (STFT) kernels.

    This function initializes the STFT kernels that are used for performing
    the Short-Time Fourier Transform. It utilizes the init_rfft_kernels
    function for initializing the Real Fast Fourier Transform kernels. It
    returns a tuple of torch tensors representing the direct and inverse
    STFT kernels.

    Args:
        win_len (int): The window length for the STFT.
        n_fft (int): The number of STFT points.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the forward
        and inverse STFT kernels.
    """

    V_dm, iV_md = init_rfft_kernels(win_len, n_fft)
    V_d1m = V_dm.unsqueeze_(dim=1)
    iV_d1m = iV_md.transpose_(dim0=0, dim1=1).unsqueeze_(dim=1)

    return V_d1m, iV_d1m


def init_dct_kernels(
        n_dct: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize Discrete Cosine Transform (DCT) kernels.

    This function initializes the DCT kernels that are used for performing
    the DCT. It returns a tuple of torch tensors representing the forward
    and inverse DCT kernels.

    Args:
        n_dct (int): The size of the DCT.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the forward
        and inverse DCT kernels.
    """

    V_nn = scp_fft.dct(np.eye(n_dct), type=2, norm='ortho')

    V_qf = V_nn.T
    iV_fq = V_nn
    return (
        torch.from_numpy(V_qf.astype(np.float32)),
        torch.from_numpy(iV_fq.astype(np.float32)))


def init_window(win_len: int, window_type: str) -> torch.Tensor:
    """
    Initializes the window for the Fourier Transform.

    This function creates a window of a specified type and length for the
    Fourier Transform.

    Args:
        win_len (int): The length of the window.
        window_type (str): The type of the window. It can be any window
        type supported by scipy.signal.get_window.

    Returns:
        torch.Tensor: A 3D tensor representing the window. The shape of the
        tensor is (1, 1, win_len).
    """
    W_m = scp_sig.get_window(
        window=window_type, Nx=win_len, fftbins=True)
    W_11m = W_m[None, None, :]
    return torch.from_numpy(W_11m.astype(np.float32))


def init_derivative_window(win_len: int, window_type: str) -> torch.Tensor:
    """
    Initializes the derivative window for instantaneous frequency deviation
    extraction.

    This function creates a first order derivative of a window of a
    specified type and length. Note that, the window is normalized, e.g.,
    if the window function is
        w(t) = 0.5 + 0.5 * cos(2 * pi * t / T) ,
    then the function returns 
        w'(t) = -0.5 * sin(2 * pi * t / T) ,

    Args:
        win_len (int): The length of the window
        window_type (str): The type of the window. Currently, it can only be
        'hann', 'hamming', or 'blackman'.

    Returns:
        torch.Tensor: A 3D tensor representing the window. The shape of the
        tensor is (1, 1, win_len).
    """
    if window_type in ['hanning', 'hann']:
        phs_m = np.linspace(-np.pi, np.pi, win_len + 1)[:-1]
        # w_m = 0.5 + 0.5 * cos(phs_m)
        WDot_m = -0.5 * np.sin(phs_m)
    elif window_type == 'hamming':
        phs_m = np.linspace(-np.pi, np.pi, win_len + 1)[:-1]
        # w_m = 0.54 + 0.46 * cos(phs_m)
        WDot_m = -0.46 * np.sin(phs_m)
    elif window_type == 'blackman':
        phs_m = np.linspace(-np.pi, np.pi, win_len + 1)[:-1]
        # w_m = 0.42 + 0.5 * cos(phs_m) + 0.08 * cos(2 * phs_m)
        WDot_m = -0.42 * np.sin(phs_m) - 0.08 * 2 * np.sin(2 * phs_m)
    else:
        raise KeyError(f'The current derivative window is not '
                       f'available for `{window_type}` window!')

    WDot_11m = WDot_m[None, None, :]
    return torch.from_numpy(WDot_11m.astype(np.float32))


def minimum_power_of_two(n: int) -> int:
    """
    Calculate the minimum power of two that is 
    greater than or equal to the input number.

    This function takes an integer as input and returns the smallest
    integer that is a power of two and is greater than or equal to the
    input number.

    Args:
        n (int): The input integer.

    Returns:
        int: The smallest integer that is a power of two and is greater
        than or equal to the input number.
    """
    return int(2 ** math.ceil(math.log2(n)))


class DCTConfigs(ModuleInterface.ConstructorArgs):
    n_dct: int


class LinearDCT(nn.Module, ModuleInterface):
    ConstructorArgs = DCTConfigs

    def __init__(self, n_dct: int) -> None:
        ModuleInterface.__init__(self, n_dct=n_dct)
        nn.Module.__init__(self)

        self.n_dct: int = n_dct
        V_qf, _ = init_dct_kernels(n_dct)
        self.register_buffer(name='V_qf', tensor=V_qf)
    
    def forward(self, x__f: torch.Tensor) -> torch.Tensor:
        assert x__f.size(-1) == self.n_dct
        x__q = F.linear(x__f, self.V_qf)
        return x__q


class LinearIDCT(nn.Module, ModuleInterface):
    ConstructorArgs = DCTConfigs

    def __init__(self, n_dct: int) -> None:
        ModuleInterface.__init__(self, n_dct=n_dct)
        nn.Module.__init__(self)

        self.n_dct: int = n_dct
        _, iV_fq = init_dct_kernels(n_dct=n_dct)
        self.register_buffer(name='iV_fq', tensor=iV_fq)
    
    def forward(self, x__q: torch.Tensor) -> torch.Tensor:
        assert x__q.size(-1) == self.n_dct
        x__f = F.linear(x__q, self.iV_fq)
        return x__f


class STFTConfigs(ModuleInterface.ConstructorArgs):
    win_len: int 
    hop_len: int
    fft_len: Optional[int] = None
    win_type: str = 'hanning'
    pad_len: Optional[int] = None


class FrameSTFTConfigs(ModuleInterface.ConstructorArgs):
    win_len: int 
    fft_len: Optional[int] = None
    win_type: str = 'hanning'


class ConvFrameSTFT(nn.Module, ModuleInterface):
    ConstructorArgs = FrameSTFTConfigs

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.win_len: int = self._configs.win_len
        self.win_type: str = self._configs.win_type
        self.fft_len: int = (
            minimum_power_of_two(self.win_len)
            if self._configs.fft_len is None
            else self._configs.fft_len)
        self.dim: int = self.fft_len // 2 + 1

        V_d1m, _ = init_stft_kernels(self.win_len, self.fft_len)
        W_11m = init_window(self.win_len, self.win_type)

        V_dm1 = (V_d1m*W_11m).transpose_(-1, -2)
        self.register_buffer(name='V_dm1', tensor=V_dm1)
        self.V_dm1: torch.Tensor

    def forward(self, X_bmt: torch.Tensor) -> torch.Tensor:
        X_bdt = F.conv1d(X_bmt, self.V_dm1)
        X_b2ft = X_bdt.unflatten(dim=1, sizes=(2, self.dim))
        return X_b2ft


class ConvFrameISTFT(nn.Module, ModuleInterface):
    ConstructorArgs = FrameSTFTConfigs

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.win_len: int = self._configs.win_len
        self.win_type: str = self._configs.win_type
        self.fft_len: int = (
            minimum_power_of_two(self.win_len) 
            if self._configs.fft_len is None
            else self._configs.fft_len)
        self.dim: int = self.fft_len // 2 + 1

        _, iV_d1m = init_stft_kernels(self.win_len, self.fft_len)
        W_11m = init_window(self.win_len, self.win_type)

        iV_dm1 = (iV_d1m*W_11m).transpose(-1, -2)
        self.register_buffer(name='iV_dm1', tensor=iV_dm1)
        self.register_buffer(
            name='W2_1m1', 
            tensor=W_11m.square().transpose(-1, -2))
        self.iV_dm1: torch.Tensor
        self.W2_1m1: torch.Tensor

    def forward(self, X_b2ft: torch.Tensor) -> torch.Tensor:
        X_bdt = X_b2ft.flatten(start_dim=1, end_dim=2)
        X_bmt = F.conv_transpose1d(X_bdt, self.iV_dm1)
        X_bmt = X_bmt / (self.W2_1m1 + 1e-8)
        return X_bmt


class ConvSTFT(nn.Module, ModuleInterface):
    ConstructorArgs = STFTConfigs

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: STFTConfigs

        self.hop_len: int = self._configs.hop_len
        self.win_len: int = self._configs.win_len
        self.win_type: str = self._configs.win_type
        self.fft_len: int = (
            minimum_power_of_two(self.win_len)
            if self._configs.fft_len is None
            else self._configs.fft_len)
        self.dim: int = self.fft_len // 2 + 1

        if self._configs.pad_len is None:
            self.pad_len: int = self.win_len - self.hop_len
        else:
            self.pad_len: int = self._configs.pad_len

        V_d1m, _ = init_stft_kernels(self.win_len, self.fft_len)
        W_11m = init_window(self.win_len, self.win_type)
        self.register_buffer(name='V_d1m', tensor=V_d1m*W_11m)

    def forward(self, x_bt: torch.Tensor) -> torch.Tensor:
        x_b1t = x_bt.unsqueeze(1)
        x_b1t = F.pad(x_b1t, [self.pad_len, self.pad_len])
        X_bdt = F.conv1d(x_b1t, self.V_d1m, stride=self.hop_len)
        X_b2ft = X_bdt.unflatten(dim=1, sizes=(2, self.dim))

        return X_b2ft


class ConvISTFT(nn.Module, ModuleInterface):
    ConstructorArgs = STFTConfigs

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self._configs: STFTConfigs

        self.hop_len: int = self._configs.hop_len
        self.win_len: int = self._configs.win_len
        self.win_type: str = self._configs.win_type
        self.fft_len: int = (
            minimum_power_of_two(self.win_len) 
            if self._configs.fft_len is None
            else self._configs.fft_len)
        self.dim: int = self.fft_len // 2 + 1

        if self._configs.pad_len is None:
            self.pad_len: int = self.win_len - self.hop_len
        else:
            self.pad_len: int = self._configs.pad_len

        _, iV_d1m = init_stft_kernels(self.win_len, self.fft_len)
        W_11m = init_window(self.win_len, self.win_type)
        I_m1m = torch.from_numpy(
            np.eye(self.win_len, dtype=np.float32)[:, None, :])
        self.register_buffer(name='iV_d1m', tensor=iV_d1m*W_11m)
        self.register_buffer(
            name='W_1m1', tensor=W_11m.transpose(-1, -2))
        self.register_buffer(name='I_m1m', tensor=I_m1m)

    def forward(self, X_b2ft: torch.Tensor) -> torch.Tensor:
        X_bdt = X_b2ft.flatten(start_dim=1, end_dim=2)
        Y_b1t = F.conv_transpose1d(
            X_bdt, self.iV_d1m, stride=self.hop_len)

        W2_1mt = self.W_1m1.square().repeat(1, 1, X_bdt.size(-1))
        c_1mt = F.conv_transpose1d(
            W2_1mt, self.I_m1m, stride=self.hop_len)

        Y_b1t = Y_b1t / (c_1mt + 1e-8)
        if self.pad_len > 0:
            Y_b1t = Y_b1t[..., self.pad_len: -self.pad_len]

        return Y_b1t.squeeze(dim=1)


class ConvDerivativeSTFT(nn.Module, ModuleInterface):
    ConstructorArgs = STFTConfigs

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.hop_len: int = self._configs.hop_len
        self.win_len: int = self._configs.win_len
        self.win_type: str = self._configs.win_type
        self.fft_len: int = (
            minimum_power_of_two(self.win_len)
            if self._configs.fft_len is None
            else self._configs.fft_len)
        self.dim: int = self.fft_len // 2 + 1
        self.pad_len: int = self.win_len - self.hop_len

        V_d1m, _ = init_stft_kernels(self.win_len, self.fft_len)
        W_11m = init_derivative_window(self.win_len, self.win_type)
        self.register_buffer(name='V_d1m', tensor=V_d1m*W_11m)

    def forward(self, x_bt: torch.Tensor) -> torch.Tensor:
        x_b1t = x_bt.unsqueeze(1)
        x_b1t = F.pad(x_b1t, [self.pad_len, self.pad_len])
        X_bdt = F.conv1d(x_b1t, self.V_d1m, stride=self.hop_len)
        X_b2ft = X_bdt.unflatten(dim=1, sizes=(2, self.dim))

        return X_b2ft


class ConvFrameDerivativeSTFT(nn.Module, ModuleInterface):
    ConstructorArgs = FrameSTFTConfigs

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.win_len: int = self._configs.win_len
        self.win_type: str = self._configs.win_type
        self.fft_len: int = (
            minimum_power_of_two(self.win_len) 
            if self._configs.fft_len is None
            else self._configs.fft_len)
        self.dim: int = self.fft_len // 2 + 1

        V_d1m, _ = init_stft_kernels(self.win_len, self.fft_len)
        W_11m = init_derivative_window(self.win_len, self.win_type)

        V_dm1 = (V_d1m*W_11m).transpose_(-1, -2)
        self.register_buffer(name='V_dm1', tensor=V_dm1)
        self.V_dm1: torch.Tensor

    def forward(self, X_bmt: torch.Tensor) -> torch.Tensor:
        X_bdt = F.conv1d(X_bmt, self.V_dm1)
        X_b2ft = X_bdt.unflatten(dim=1, sizes=(2, self.dim))
        return X_b2ft


def sanity_check():
    import soundfile as sf
    import pypesq
    import pystoi

    torch.manual_seed(20)
    win_len = 320
    hop_len = 160
    fft_len = 512

    conv_stft = ConvSTFT(
        win_len=win_len, hop_len=hop_len,
        fft_len=fft_len, win_type='hann')
    conv_istft = ConvISTFT(
        win_len=win_len, hop_len=hop_len,
        fft_len=fft_len, win_type='hann')

    print(' -- Gaussian random signal -- ')
    x_bt = torch.randn([1, 16000 * 4]) * 0.01

    x_b2ft = conv_stft(x_bt)
    xHat_bt = conv_istft(x_b2ft)

    print(
        'Input signal RMSE',
        (x_bt).square().mean().sqrt().item())
    print(
        'Output signal RMSE',
        (xHat_bt).square().mean().sqrt().item())
    print(
        'Reconstruction RMSE', 
        (x_bt - xHat_bt).square().mean().sqrt().item())

    print(' -- Speech signal -- ')
    x_t, sr = sf.read('sample_data/26-496-0000.wav')
    assert sr == 16000
    x_bt = torch.from_numpy(x_t[None, :].astype(np.float32))
    x_b2ft = conv_stft(x_bt)
    xHat_bt = conv_istft(x_b2ft)

    print(
        'Input signal RMSE', 
        (x_bt).square().mean().sqrt().item())
    print(
        'Output signal RMSE',
        (xHat_bt).square().mean().sqrt().item())
    print(
        'Reconstruction RMSE',
        (x_bt - xHat_bt).square().mean().sqrt().item())
    print(
        'PESQ',
        pypesq.pesq(
            x_bt.detach().cpu().numpy()[0, :],
            xHat_bt.detach().cpu().numpy()[0, :],
            sr))
    print(
        'STOI', 
        pystoi.stoi(
            x_bt.detach().cpu().numpy()[0, :],
            xHat_bt.detach().cpu().numpy()[0, :],
            sr))

    lin_dct, lin_idct = LinearDCT(257), LinearIDCT(257)

    ps_bft = x_b2ft.square().sum(dim=1)
    lps_bft = (ps_bft + 1e-20).log()

    cep_bqt = lin_dct(lps_bft.transpose(1, 2)).transpose(1, 2)
    lpsHat_bft = lin_idct(cep_bqt.transpose(1, 2)).transpose(1, 2)

    print((lpsHat_bft - lps_bft).square().mean().sqrt().item())

    cepTrue_bqt = torch.from_numpy(scp_fft.dct(
        lps_bft.detach().cpu().numpy(),
        type=2, axis=1, norm='ortho'))

    print((cep_bqt - cepTrue_bqt).square().mean().sqrt().item())


if __name__ == '__main__':
    sanity_check()
