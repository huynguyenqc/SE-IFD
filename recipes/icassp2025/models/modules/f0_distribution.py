import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from deep.base_module import ModuleInterface


class F0ImportanceDistribution(nn.Module, ModuleInterface):
    """ Categorical F0 distribution based on significance value """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        f0_min: float = 60.0
        f0_max: float = 300.0
        n_f0_candidates: int = 241
        fs: int = 16000
        n_fft: int = 512
        tau: float = 0.45
        theta: float = 2.0

    def __init__(
            self,
            f0_min: float = 60.,
            f0_max: float = 300.,
            n_f0_candidates: int = 241,
            fs: int = 16000,
            n_fft: int = 512,
            tau: float = 0.45,
            theta: float = 2.0
    ) -> None:
        ModuleInterface.__init__(
            self, f0_min, f0_max, n_f0_candidates, fs, n_fft, tau)
        nn.Module.__init__(self)

        self.f0_min: float = f0_min
        self.f0_max: float = f0_max
        self.n_f0_candidates: int = n_f0_candidates
        self.fs: int = fs
        self.f_nyquist: float = fs / 2
        self.n_fft: int = n_fft
        self.n_freq_bins: int = n_fft // 2 + 1
        self.tau: float = tau       # Temperature for softmax distribution
        self.theta: float = theta   # Threshold for F0 distribution

        # F0 candidates
        self.register_buffer('q_l', torch.linspace(f0_min, f0_max, n_f0_candidates))

        # Integral matrix
        self.U_qf: np.ndarray = np.zeros((self.n_f0_candidates, self.n_freq_bins))
        for l in range(self.n_f0_candidates):
            last_idx = 0
            for k in range(1, int(math.floor(self.f_nyquist / self.q_l[l])) + 1):
                idx = int(math.floor(self.q_l[l] * k * (self.n_freq_bins - 1) / self.f_nyquist))
                self.U_qf[l, idx] += 1 / math.sqrt(k)
                if idx - last_idx > 1:
                    i = int(math.floor((idx + last_idx) / 2.))
                    if (idx - last_idx) % 2 != 0:
                        self.U_qf[l, i] -= 1 / (2 * math.sqrt(k))
                        self.U_qf[l, i+1] -= 1 / (2 * math.sqrt(k))
                    else:
                        self.U_qf[l, i] -= 1 / math.sqrt(k)
                else:
                    self.U_qf[l, idx] -= 1 / (2 * math.sqrt(k))
                    self.U_qf[l, last_idx] -= 1 / (2 * math.sqrt(k))
                last_idx = idx
        self.register_buffer('U_qf1', torch.from_numpy(self.U_qf[:, :, None].astype(np.float32)))

    def forward(self, logXp_bft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logXp_bft (torch.Tensor): Log power spectrogram
        Returns:
            P_bqt (torch.Tensor): (Discrete) probability distribution of F0 candidate
            m_bt (torch.Tensor): Whether or not the frame has harmonic (1: Yes, 0: No)
        """
        # Importance matrix
        Q_bqt = F.conv1d(input=logXp_bft, weight=self.U_qf1)

        # Normalise importance w.r.t. for each sample
        # with torch.no_grad():
        #     muQ_b11 = Q_bqt.mean(dim=(1, 2), keepdim=True).detach()
        #     sigmaQ_b11 = Q_bqt.std(dim=(1, 2), keepdim=True).detach()

        muQ_b11 = Q_bqt.mean(dim=(1, 2), keepdim=True)
        sigmaQ_b11 = Q_bqt.std(dim=(1, 2), keepdim=True)
        normQ_bqt = (Q_bqt - muQ_b11) / (sigmaQ_b11 + 1e-5)

        # Distribution of F0
        P_bqt = F.softmax(normQ_bqt / self.tau, dim=1)
        with torch.no_grad():
            # Entropy of each frame
            h_bt = -(P_bqt * (P_bqt + 1e-12).log()).sum(dim=1)
            # The smaller entropy -> the higher confidence (less uniform)
            m_bt = (h_bt < self.theta).float()
        return P_bqt, m_bt
