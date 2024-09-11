import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, Iterator, Literal, List, Optional, Tuple
from deep.base_module import ModuleInterface
from deep.modules.fourier import (
    STFTConfigs, FrameSTFTConfigs,
    ConvSTFT, ConvISTFT, LinearDCT, 
    ConvFrameSTFT, ConvFrameISTFT, ConvFrameDerivativeSTFT)
from deep.complex_modules import complex_op as c_op
from .modules.base_modules import (
    Encoder, Decoder, ComplexCRNN, 
    ComplexCRNNScalingOutput)
from .modules.f0_distribution import F0ImportanceDistribution
from .modules.unet_autoencoders import VQUNetBase, VQUNet, DenoiseVQUNet
from .modules.utils import (
    future_estimation, 
    past_groundtruth,
    complex_mask_multiplication,
    kl_divergence_categorical_with_mask,
    kl_divergence_exponential,
    kl_divergence_log_normal_unit_variance,
    squared_euclidean_distance,
    negative_signal_to_distortion_ratio_decibel,
    angle_mclaurin_series_expansion,
    angle_multiplication,
    angle_polar_from_mclaurin)


class SpeechVarianceEstimator(nn.Module, ModuleInterface):
    """ Speech variance estimator with cepstrum conditioning """
    class ConstructorArgs(ModuleInterface.ConstructorArgs): 
        dim: int
        look_ahead: int = 3
        vqvae_configs: VQUNetBase.ConstructorArgs
        cep_encoder_configs: Optional[Encoder.ConstructorArgs] = None

    def __init__(self, mode: Literal['pretrain', 'denoise'], **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: SpeechVarianceEstimator.ConstructorArgs

        self.dim = self._configs.dim                  # Number of frequency bins
        self.look_ahead = self._configs.look_ahead    # Number of look-head frames

        if mode == 'pretrain':
            self.vqvae = VQUNet(**self._configs.vqvae_configs.model_dump())
        elif mode == 'denoise':
            self.vqvae = DenoiseVQUNet(**self._configs.vqvae_configs.model_dump())
        else:
            raise ValueError(f'Unknown mode "{mode}" for SpeechVarianceEstimator')

        self.cepstrum_condition: bool = self._configs.cep_encoder_configs is not None
        if self.cepstrum_condition:
            self.dct = LinearDCT(n_dct=self.dim)
            self.cep_encoder = Encoder(**self._configs.cep_encoder_configs.model_dump())

        # Avoid numerical issue
        self.eps: float = 1e-12
        self.log_eps: float = -12 * math.log(10)
        self.log_big_eps: float = -self.log_eps

    def cepstrum_extraction(self, logXp_bft: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.cepstrum_condition:
            return None

        logXp_btf = logXp_bft.transpose(-1, -2)
        Xc_btq = self.dct(logXp_btf)
        Xc_bqt = Xc_btq.transpose(-1, -2)
        return Xc_bqt

    def forward(
            self,
            logXp_bft: torch.Tensor,
            do_quantisation: bool = False,
            do_reservoir_sampling: bool = False
    ) -> Tuple[
            Optional[List[Optional[torch.Tensor]]], 
            torch.Tensor,
            List[torch.Tensor],
            Optional[List[torch.Tensor]],
            List[float]]:

        list_cdx_bct = [None] * len(self.vqvae.encoders.encoders)
        if self.cepstrum_condition:
            Xc_bqt = self.cepstrum_extraction(logXp_bft)
            list_cdx_bct[-1] = self.cep_encoder(Xc_bqt)

        logXpHat_bft, list_zx_bct, list_qx_bct, list_perp_x = self.vqvae(
            logXp_bft[:, 1:, :],        # Remove DC
            list_cdx_bct=list_cdx_bct,
            do_quantisation=do_quantisation,
            do_reservoir_sampling=do_reservoir_sampling)

        logXpHat_bft = future_estimation(
            logXpHat_bft, look_ahead_width=self.look_ahead
        ).clamp(self.log_eps, self.log_big_eps)

        return list_cdx_bct, logXpHat_bft, list_zx_bct, list_qx_bct, list_perp_x

    def encode_and_quantise(
            self, 
            logXp_bft: torch.Tensor
        ) -> Tuple[
            Optional[List[Optional[torch.Tensor]]], 
            List[torch.Tensor], 
            List[float]]:
        list_cdx_bct = [None] * len(self.vqvae.encoders.encoders)
        if self.cepstrum_condition:
            Xc_bqt = self.cepstrum_extraction(logXp_bft)
            list_cdx_bct[-1] = self.cep_encoder(Xc_bqt)
        return list_cdx_bct, *self.vqvae.encode_and_quantise(logXp_bft[:, 1:, :])   # Remove DC

    def encode(
            self, 
            logXp_bft: torch.Tensor
        ) -> Tuple[
            Optional[List[Optional[torch.Tensor]]], 
            List[torch.Tensor]]:
        list_cdx_bct = [None] * len(self.vqvae.encoders.encoders)
        if self.cepstrum_condition:
            Xc_bqt = self.cepstrum_extraction(logXp_bft)
            list_cdx_bct[-1] = self.cep_encoder(Xc_bqt)
        return list_cdx_bct, self.vqvae.encode(logXp_bft[:, 1:, :])     # Remove DC

    def initialise_codebook_from_reservoir(self) -> None:
        return self.vqvae.initialise_codebook_from_reservoir()


class NoiseVarianceEstimator(nn.Module, ModuleInterface):
    """ Noise variance estimator conditioned by speech variance """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        cond_encoder_configs: Encoder.ConstructorArgs
        decoder_configs: Decoder.ConstructorArgs
        look_ahead: int = 3
        input_type: Literal['log-subtract', 'subtract', 'noisy'] = 'log-subtract'
        
    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: NoiseVarianceEstimator.ConstructorArgs

        self.input_type = self._configs.input_type
        self.look_ahead = self._configs.look_ahead
        self.cond_encoder = Encoder(**self._configs.cond_encoder_configs.model_dump())
        self.decoder = Decoder(**self._configs.decoder_configs.model_dump())

        if self.input_type == 'subtract':
            self.speech_decay = nn.Parameter(
                data=torch.tensor(0.0).float(), requires_grad=True)
        else:
            self.speech_decay = None
        self.speech_decay: Optional[nn.Parameter]

        # Avoid numerical issue
        self.eps: float = 1e-12
        self.log_eps: float = -12 * math.log(10)
        self.log_big_eps: float = -self.log_eps

    def forward(
            self, 
            logYp_bft: torch.Tensor, 
            logXp_bft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logYp_bft (torch.Tensor): Noisy log variance
            logXp_bft (torch.Tensor): Speech log variance

        Returns:
            logNpHat_bft (torch.Tensor): Estimated noise log variance
        """
        if self.input_type == 'log-subtract':
            # Emperical subtraction in log domain
            logNpPrior_bft = logYp_bft - logXp_bft
        elif self.input_type == 'subtract':
            # Log spectral subtraction
            logNpPrior_bft = (logYp_bft.exp() - logXp_bft.exp() * self.speech_decay.sigmoid()).clamp(self.eps, self.log_big_eps).log()
        elif self.input_type == 'noisy':
            logNpPrior_bft = logYp_bft

        logNpHat_bft = self.decoder(
            logNpPrior_bft,
            self.cond_encoder(logXp_bft))
        logNpHat_bft = future_estimation(
            logNpHat_bft, look_ahead_width=self.look_ahead
        ).clamp(self.log_eps, self.log_big_eps)
        return logNpHat_bft


class PhaseCorrector(nn.Module, ModuleInterface):
    """ Noise estimator using complex-valued CRNN """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        look_ahead: int = 3
        ccrnn_configs: ComplexCRNN.ConstructorArgs

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: PhaseCorrector.ConstructorArgs

        self.look_ahead = self._configs.look_ahead
        self.phase_corrector = ComplexCRNN(**self._configs.ccrnn_configs.model_dump())

    def forward(self, Y_b2ft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y_b2ft (torch.Tensor): Noisy-phase complex spectrogram

        Returns:
            XHat_b2ft (torch.Tensor): Phase-corrected complex spectrogram
        """
        MPhiHat_b2ft = self.phase_corrector(Y_b2ft)
        MPhiHat_b2ft = future_estimation(
            MPhiHat_b2ft, look_ahead_width=self.look_ahead)

        XHat_b2ft = complex_mask_multiplication(
            X_b2ft=past_groundtruth(Y_b2ft, MPhiHat_b2ft), 
            M_b2ft=MPhiHat_b2ft)

        return XHat_b2ft


class IFDMaskEstimator(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        frame_stft_configs: FrameSTFTConfigs
        ccrnn_configs: ComplexCRNNScalingOutput.ConstructorArgs
        n_components_taylor: Optional[int] = 5
        ifd_extractor: Literal['phase-difference', 'analytical-derivative'] = 'phase-difference'
        look_ahead: int = 3

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: IFDMaskEstimator.ConstructorArgs

        self.look_ahead = self._configs.look_ahead
        self.ifd_mask_estimator = ComplexCRNNScalingOutput(**self._configs.ccrnn_configs.model_dump())
        self.n_components_taylor: Optional[int] = self._configs.n_components_taylor
        self.frame_stft_configs = self._configs.frame_stft_configs
        self.ifd_extractor = self._configs.ifd_extractor

        self.frame_stft = ConvFrameSTFT(**self.frame_stft_configs.model_dump())
        self.frame_istft = ConvFrameISTFT(**self.frame_stft_configs.model_dump())
        self.frame_split_istft = ConvISTFT(hop_len=1, pad_len=0, **self.frame_stft_configs.model_dump())

        if self.ifd_extractor == 'analytical-derivative':
            self.frame_dstft = ConvFrameDerivativeSTFT(**self.frame_stft_configs.model_dump())

        # Centre frequency
        self.fft_dim = self.frame_stft_configs.fft_len // 2 + 1
        self.win_len = self.frame_stft_configs.win_len
        self.register_buffer(
            'omegaC_1f1',
            math.pi * torch.linspace(0, 1, self.fft_dim).view(1, -1, 1))
        self.omegaC_1f1: torch.Tensor
        self.register_buffer(
            'exp1j_omegaC_12f1',
            c_op.p2c(self.omegaC_1f1, c_dim=1))
        self.exp1j_omegaC_12f1: torch.Tensor

    def forward(self, 
                Y_b2ft: torch.Tensor, 
                IFD_scaling: bool = True,
                IFD_shifting: bool = True
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature extraction
        with torch.no_grad():
            Y_bmt: torch.Tensor = self.frame_istft(Y_b2ft)
            Yfuture_bmt: torch.Tensor = F.pad(
                Y_bmt.narrow(dim=1, start=1, length=self.win_len - 1),
                pad=(0, 0, 0, 1))
            Yfuture_b2ft: torch.Tensor = self.frame_stft(Yfuture_bmt)

            # XfutureHatInit_b2ft = Y_b2ft
            XfutureHatInit_b2ft = Yfuture_b2ft

            if self.ifd_extractor == 'phase-difference':
                # Instantaneous frequency
                dPhi_bft = (
                    c_op.c_phs(Yfuture_b2ft, c_dim=1, keepdim=False) 
                    - c_op.c_phs(Y_b2ft, c_dim=1, keepdim=False))
                # Instantaneous frequency deviation
                QY_bft = self.omegaC_1f1 - dPhi_bft
                # Wrapping
                QY_bft = torch.atan2(QY_bft.sin(), QY_bft.cos())
            elif self.ifd_extractor == 'analytical-derivative':
                Yd_b2ft: torch.Tensor = self.frame_dstft(Y_bmt)
                QY_bft = (2 * math.pi / self.win_len) * c_op.c_div_i(
                    Yd_b2ft, Y_b2ft, c_dim=1, keepdim=False, eps=1e-12)
                QY_bft.clamp_(-math.pi, math.pi)
            else:
                raise KeyError('Unexpected extractor method "{}"!'.format(
                    self.ifd_extractor))

            # McLaurin expansion of sine and cosine of angle
            if self.n_components_taylor is not None and self.training:
                exp1j_QY_bdft = angle_mclaurin_series_expansion(
                    QY_bft, dim=1, n_components=self.n_components_taylor)

        # IFD masking
        alphaHat_bft, zetaHat_b2ft = self.ifd_mask_estimator(Y_b2ft)
        alphaHat_bft: torch.Tensor
        zetaHat_b2ft: torch.Tensor

        if not IFD_scaling:
            alphaHat_bft = torch.ones_like(alphaHat_bft)
        if not IFD_shifting:
            zetaHat_b2ft = torch.cat([torch.ones_like(zetaHat_b2ft[:, : 1, :, :]),
                                      torch.zeros_like(zetaHat_b2ft[:, : 1, :, :])], dim=1)

        alphaHat_bft = future_estimation(alphaHat_bft, self.look_ahead)
        zetaHat_b2ft = future_estimation(zetaHat_b2ft, self.look_ahead)
        # Some trick
        # alphaHat_bft = alphaHat_bft + 1
        # zetaHat_b2ft = c_op.c_mul(
        #     zetaHat_b2ft,
        #     c_op.c_conj(self.exp1j_omegaC_12f1, c_dim=1),
        #     c_dim=1)

        # IFD Scaling
        if self.n_components_taylor is not None and self.training:
            # Masking in Taylor's expansion domain
            exp1j_QXHat_bdft = angle_multiplication(
                past_groundtruth(exp1j_QY_bdft, alphaHat_bft), 
                dim=1, alpha__=alphaHat_bft)
            exp1j_QXHat_b2ft = angle_polar_from_mclaurin(exp1j_QXHat_bdft, dim=1)
            exp1j_QXHat_b2ft = c_op.c_normalise(exp1j_QXHat_b2ft, c_dim=1, eps=1e-6)
        else:
            QXHat_bft = past_groundtruth(QY_bft, alphaHat_bft) * alphaHat_bft
            exp1j_QXHat_b2ft = c_op.p2c(QXHat_bft, c_dim=1)

        # IFD Shifting
        exp1j_QXHat_b2ft = complex_mask_multiplication(exp1j_QXHat_b2ft, zetaHat_b2ft)
        
        # IFD to phase difference
        exp1j_dPhiXHat_b2ft = c_op.c_mul(
            self.exp1j_omegaC_12f1,
            c_op.c_conj(exp1j_QXHat_b2ft, c_dim=1),
            c_dim=1)
        
        # Update phase of future frame
        XfutureHat_b2ft = c_op.c_mul(
            past_groundtruth(XfutureHatInit_b2ft, exp1j_dPhiXHat_b2ft),
            exp1j_dPhiXHat_b2ft,
            c_dim=1)

        # Overlap-add in time domain
        batch_size, _, _, n_frames = XfutureHat_b2ft.size()
        XHat_n2f2 = torch.stack(
            [past_groundtruth(Y_b2ft, XfutureHat_b2ft),
             XfutureHat_b2ft], 
            dim=-1
        ).permute(
            0, 3, 1, 2, 4
        ).flatten(
            start_dim=0, end_dim=1)
        XHat_nm: torch.Tensor = self.frame_split_istft(XHat_n2f2)
        XHat_bmt = XHat_nm.narrow(
            dim=-1, start=0, length=self.win_len
        ).unflatten(
            dim=0, sizes=(batch_size, n_frames)
        ).transpose(-1, -2)
        
        # Convert back to frequency domain
        XHat_b2ft: torch.Tensor = self.frame_stft(XHat_bmt)

        return XHat_b2ft, alphaHat_bft, zetaHat_b2ft


def spectral_loss(
        logXp_bft: torch.Tensor, 
        logXpHat_bft: torch.Tensor, 
        spectral_distribution: Literal['exponential', 'log-normal']) -> torch.Tensor:
    if spectral_distribution == 'exponential':
        return kl_divergence_exponential(
            past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
    elif spectral_distribution == 'log-normal':
        return kl_divergence_log_normal_unit_variance(
            past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
    else:
        raise KeyError(f'Unexpected spectral distribution "{spectral_distribution}".')


def f0_distribution_loss(
        logXp_bft: torch.Tensor,
        logXpHat_bft: torch.Tensor,
        f0_dist_model: Optional[F0ImportanceDistribution] = None) -> torch.Tensor:

    if f0_dist_model is None:
        return torch.tensor(0.0, device=logXp_bft.device, dtype=logXp_bft.dtype)

    with torch.no_grad():
        P_bqt, m_bt = f0_dist_model(past_groundtruth(logXp_bft, logXpHat_bft))
    PHat_bqt, _ = f0_dist_model(logXpHat_bft)
    return kl_divergence_categorical_with_mask(P_bqt.detach(), PHat_bqt, m_bt.detach(), dim=1)


def list_distance(
        list_xHat_bct: List[Optional[torch.Tensor]],
        list_x_bct: Optional[List[Optional[torch.Tensor]]]
) -> List[torch.Tensor]:

    if list_x_bct is None:
        return [
            torch.tensor(0, device=xHat_i_bct.device, dtype=xHat_i_bct.dtype)
            for xHat_i_bct in list_xHat_bct]
    assert len(list_xHat_bct) == len(list_x_bct), 'Length mismatch!'
    return [
        squared_euclidean_distance(xHat_i_bct, x_i_bct.detach(), dim=1)
        for xHat_i_bct, x_i_bct in zip(list_xHat_bct, list_x_bct)
        if x_i_bct is not None and xHat_i_bct is not None]


def negative_sisdr(x_bt: torch.Tensor, xHat_bt: torch.Tensor) -> torch.Tensor:
    return negative_signal_to_distortion_ratio_decibel(
        past_groundtruth(x_bt, xHat_bt), xHat_bt)


class ModelInterface:
    def get_training_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError('This method must be implemented in each subclass!')


class EstimateWienerFilter(nn.Module, ModuleInterface, ModelInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        stft_configs: STFTConfigs
        speech_estimator_configs: SpeechVarianceEstimator.ConstructorArgs
        f0_dist_configs: Optional[F0ImportanceDistribution.ConstructorArgs] = None
        gamma_f0: float = 1.0
        noise_estimator_configs: NoiseVarianceEstimator.ConstructorArgs
        spectral_distribution: Literal['exponential', 'log-normal'] = 'exponential'
        wiener_type: Literal['original', 'irm'] = 'original'
        do_quantisation: bool = True
        look_ahead: Optional[int] = None

        name: Literal['EstimateWienerFilter'] = 'EstimateWienerFilter'

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: EstimateWienerFilter.ConstructorArgs

        # Parameters
        self.spectral_distribution = self._configs.spectral_distribution
        self.wiener_type = self._configs.wiener_type
        self.do_quantisation = self._configs.do_quantisation
        self.look_ahead = self._configs.look_ahead

        if self.look_ahead is not None:     # Global look-ahead
            # Reset look-ahead of all sub-modules
            self._configs.speech_estimator_configs.look_ahead = 0
            self._configs.noise_estimator_configs.look_ahead = 0

        # STFT
        self.stft = ConvSTFT(**self._configs.stft_configs.model_dump())
        self.istft = ConvISTFT(**self._configs.stft_configs.model_dump())

        # Clean-input speech variance estimator
        if self.do_quantisation:
            self.clean_speech_est = SpeechVarianceEstimator(
                mode='pretrain',
                **self._configs.speech_estimator_configs.model_dump())
        else:
            self.clean_speech_est = None

        # Speech variance estimator
        self.speech_est = SpeechVarianceEstimator(
            mode='denoise',
            **self._configs.speech_estimator_configs.model_dump())
        self.n_levels = len(self.speech_est.vqvae.encoders.encoders)

        # Noise variance estimator
        self.noise_est = NoiseVarianceEstimator(**self._configs.noise_estimator_configs.model_dump())

        # F0 loss
        self.f0_dist_model = (
            None if self._configs.f0_dist_configs is None else
            F0ImportanceDistribution(**self._configs.f0_dist_configs.model_dump()))
        self.gamma_f0 = self._configs.gamma_f0

        # Avoid numerical issue
        self.eps: float = 1e-12
        self.log_eps: float = -12 * math.log(10)

    def get_training_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.speech_est.parameters()
        yield from self.noise_est.parameters()

    @property
    def output_logging_keys(self) -> List[str]:
        return (
            ['loss', 'recon_x', 'f0_loss', 'recon_n', 'sisdr_db_xHat']
            + [f'commit_{_k}' for _k in range(self.n_levels)]
            + [f'perp_{_k}' for _k in range(self.n_levels)])

    def train(self: 'EstimateWienerFilter', mode: bool = True) -> 'EstimateWienerFilter':
        if not isinstance(mode, bool):
            raise ValueError('training mode is expected to be boolean')
        
        self.training = mode

        self.stft.train(mode=mode)
        self.istft.train(mode=mode)

        if self.clean_speech_est is not None:
            self.clean_speech_est.train(mode=False)

        self.speech_est.train(mode=mode)
        self.noise_est.train(mode=mode)

        if self.f0_dist_model is not None:
            self.f0_dist_model.train(mode=mode)

        return self

    def load_state_dict_from_pretrain_state_dict(
            self,
            pretrain_net_dict: Dict[str, Any]) -> None:
        #* The current model (self) and the state dict must be 
        #* on CPU when running this function!
        model_cls: ModuleInterface = pretrain_net_dict['model_cls']
        pretrain_net_configs = model_cls.ConstructorArgs(**pretrain_net_dict['configs'])
        pretrain_net_state_dict = pretrain_net_dict['state_dict']
        pretrain_net = model_cls(**pretrain_net_configs.model_dump())
        pretrain_net.load_state_dict(pretrain_net_state_dict)

        if self.clean_speech_est is not None and pretrain_net.clean_speech_est is not None:
            self.clean_speech_est.load_state_dict(
                pretrain_net.clean_speech_est.state_dict(), strict=False)

        missing_keys, unexpected_keys = self.speech_est.load_state_dict(
            pretrain_net.speech_est.state_dict(), strict=False)

        print('Warning: Missing keys: ', list(missing_keys))
        print('Warning: Unexpected keys: ', list(unexpected_keys))

    def forward(
            self,
            x_bt: torch.Tensor,
            y_bt: torch.Tensor,
            epoch: int = -1) -> Tuple[torch.Tensor, Dict[str, float]]:

        with torch.no_grad():
            n_bt = y_bt - x_bt

            X_b2ft = self.stft(x_bt)
            Xp_bft = c_op.c_square_mag(X_b2ft, c_dim=1, keepdim=False)
            logXp_bft = (Xp_bft + self.eps).log()

            N_b2ft = self.stft(n_bt)
            Np_bft = c_op.c_square_mag(N_b2ft, c_dim=1, keepdim=False)
            logNp_bft = (Np_bft + self.eps).log()

            Y_b2ft = self.stft(y_bt)
            Yp_bft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=False)
            logYp_bft = (Yp_bft + self.eps).log()

            if self.do_quantisation:
                _, list_qx_bct, _ = self.clean_speech_est.encode_and_quantise(logXp_bft)
            else:
                list_qx_bct = None

        # Speech variance estimation
        _, logXpHat_bft, list_zy_bct, _, list_perp_y = self.speech_est(
            logYp_bft, do_quantisation=self.do_quantisation)
        logXpHat_bft: torch.Tensor

        # Noise variance estimation
        logNpHat_bft: torch.Tensor = self.noise_est(
            past_groundtruth(logYp_bft, logXpHat_bft),
            logXpHat_bft.detach())
        
        # Global look-ahead
        if self.look_ahead is not None:
            logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
            logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

        # Wiener filter
        with torch.no_grad():
            HpHat_bft = 1. / (
                1. + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            XHat_b2ft = HpHat_bft.unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
            xHat_bt: torch.Tensor = self.istft(XHat_b2ft)
            nsisdr_db_xHat = negative_sisdr(x_bt, xHat_bt)

        # Compute loss
        recon_x = spectral_loss(logXp_bft, logXpHat_bft, self.spectral_distribution)
        f0_loss = f0_distribution_loss(logXp_bft, logXpHat_bft, self.f0_dist_model)
        list_commit_y = list_distance(list_xHat_bct=list_zy_bct, list_x_bct=list_qx_bct)
        recon_n = spectral_loss(logNp_bft, logNpHat_bft, self.spectral_distribution)

        loss = recon_x + self.gamma_f0 * f0_loss + sum(list_commit_y) + recon_n

        return loss, {
            'loss': loss.item(),
            'sisdr_db_xHat': -nsisdr_db_xHat.item(),
            'recon_x': recon_x.item(),
            'recon_n': recon_n.item(),
            'f0_loss': f0_loss.item(),
            **{
                f'commit_{_k}': cl.item() 
                for _k, cl in enumerate(list_commit_y)},
            **{
                f'perp_{_k}': cl
                for _k, cl in enumerate(list_perp_y)}}

    def validate(
            self,
            x_bt: torch.Tensor,
            y_bt: torch.Tensor,
            epoch: int = -1) -> Dict[str, Any]:
        sig_len = x_bt.size(-1)
        win_len = self.stft.win_len
        hop_len = self.stft.hop_len
        model_stride = self.speech_est.vqvae.encoders.total_stride
        latent_len = (sig_len + win_len - hop_len) // (hop_len * model_stride)
        input_len = (latent_len * model_stride + 1) * hop_len - win_len
        x_bt = x_bt[..., : input_len]
        y_bt = y_bt[..., : input_len]

        with torch.no_grad():
            n_bt = y_bt - x_bt

            X_b2ft = self.stft(x_bt)
            Xp_bft = c_op.c_square_mag(X_b2ft, c_dim=1, keepdim=False)
            logXp_bft = (Xp_bft + self.eps).log()

            N_b2ft = self.stft(n_bt)
            Np_bft = c_op.c_square_mag(N_b2ft, c_dim=1, keepdim=False)
            logNp_bft = (Np_bft + self.eps).log()

            Y_b2ft = self.stft(y_bt)
            Yp_bft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=False)
            logYp_bft = (Yp_bft + self.eps).log()

            if self.do_quantisation:
                _, list_qx_bct, _ = self.clean_speech_est.encode_and_quantise(logXp_bft)
            else:
                list_qx_bct = None

            # Speech variance estimation
            _, logXpHat_bft, list_zy_bct, _, list_perp_y = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter
            HpHat_bft = 1. / (
                1. + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            XHat_b2ft = HpHat_bft.unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
            xHat_bt: torch.Tensor = self.istft(XHat_b2ft)
            nsisdr_db_xHat = negative_sisdr(x_bt, xHat_bt)

            # Compute loss
            recon_x = spectral_loss(logXp_bft, logXpHat_bft, self.spectral_distribution)
            f0_loss = f0_distribution_loss(logXp_bft, logXpHat_bft, self.f0_dist_model)
            list_commit_y = list_distance(list_xHat_bct=list_zy_bct, list_x_bct=list_qx_bct)
            recon_n = spectral_loss(logNp_bft, logNpHat_bft, self.spectral_distribution)

            loss = recon_x + self.gamma_f0 * f0_loss + sum(list_commit_y) + recon_n

            return {
                'numerical': {
                    'loss': loss.item(),
                    'sisdr_db_xHat': -nsisdr_db_xHat.item(),
                    'recon_x': recon_x.item(),
                    'recon_n': recon_n.item(),
                    'f0_loss': f0_loss.item(),
                    **{
                        f'commit_{_k}': cl.item() 
                        for _k, cl in enumerate(list_commit_y)},
                    **{
                        f'perp_{_k}': cl
                        for _k, cl in enumerate(list_perp_y)}},
                'waveform': {
                    'xHat_bt': xHat_bt.detach().cpu(),
                    'x_bt': x_bt.detach().cpu(),
                    'y_bt': y_bt.detach().cpu()},
                'spectrum': {
                    'logXp_bft': logXp_bft.detach().cpu(),
                    'logYp_bft': logYp_bft.detach().cpu(),
                    'logNp_bft': logNp_bft.detach().cpu(),
                    'logXpHat_bft': logXpHat_bft.detach().cpu(),
                    'logNpHat_bft': logNpHat_bft.detach().cpu()},
                'mask': {
                    'HpHat_bft': HpHat_bft.detach().cpu()
                }
            }

    def enhance(self, y_bt: torch.Tensor,
                return_spec: bool = False,
                use_pad: bool = False) -> torch.Tensor:
        sig_len = y_bt.size(-1)
        win_len = self.stft.win_len
        hop_len = self.stft.hop_len
        model_stride = self.speech_est.vqvae.encoders.total_stride
        look_ahead = (
            (self.speech_est.look_ahead + self.noise_est.look_ahead) 
            if self.look_ahead is None 
            else self.look_ahead)
        if use_pad:
            latent_len = (sig_len + win_len - hop_len - 1) // (hop_len * model_stride) + 1
            input_len = (latent_len * model_stride + look_ahead + 1) * hop_len - win_len
            y_bt = F.pad(y_bt, (0, input_len-sig_len))
        else:
            latent_len = (sig_len + win_len - hop_len) // (hop_len * model_stride)
            input_len = (latent_len * model_stride + 1) * hop_len - win_len
            y_bt = y_bt[..., : input_len]

        with torch.no_grad():
            Y_b2ft = self.stft(y_bt)
            Yp_bft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=False)
            logYp_bft = (Yp_bft + self.eps).log()

            # Speech variance estimation
            _, logXpHat_bft, _, _, _ = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter
            HpHat_bft = 1. / (
                1. + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            XHat_b2ft = HpHat_bft.unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)

            if return_spec:
                return XHat_b2ft

            xHat_bt: torch.Tensor = self.istft(XHat_b2ft)

            return xHat_bt[: sig_len]


class CorrectPhase(nn.Module, ModuleInterface, ModelInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        stft_configs: STFTConfigs
        speech_estimator_configs: SpeechVarianceEstimator.ConstructorArgs
        noise_estimator_configs: NoiseVarianceEstimator.ConstructorArgs
        phase_corrector_configs: PhaseCorrector.ConstructorArgs
        wiener_type: Literal['original', 'irm'] = 'original'
        do_quantisation: bool = True
        look_ahead: Optional[int] = None

        name: Literal['CorrectPhase'] = 'CorrectPhase'

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: CorrectPhase.ConstructorArgs

        # Parameters
        self.wiener_type = self._configs.wiener_type
        self.do_quantisation = self._configs.do_quantisation
        self.look_ahead = self._configs.look_ahead

        if not self.do_quantisation:        # Do not initialise quantisation module
            self._configs.speech_estimator_configs.vqvae_configs.quantiser_configs = None

        if self.look_ahead is not None:     # Global look-ahead
            self._configs.speech_estimator_configs.look_ahead = 0
            self._configs.noise_estimator_configs.look_ahead = 0
            self._configs.phase_corrector_configs.look_ahead = 0

        # STFT
        self.stft = ConvSTFT(**self._configs.stft_configs.model_dump())
        self.istft = ConvISTFT(**self._configs.stft_configs.model_dump())

        # Speech variance estimator
        self.speech_est = SpeechVarianceEstimator(
            mode='denoise',
            **self._configs.speech_estimator_configs.model_dump())

        # Noise variance estimator
        self.noise_est = NoiseVarianceEstimator(
            **self._configs.noise_estimator_configs.model_dump())

        # Phase correction
        self.phase_corrector = PhaseCorrector(
            **self._configs.phase_corrector_configs.model_dump())

        # Avoid numerical issue
        self.eps: float = 1e-12
        self.log_eps: float = -12 * math.log(10)

    def get_training_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.phase_corrector.parameters()

    @property
    def output_logging_keys(self) -> List[str]:
        return ['loss', 'SISDRdB_before', 'SISDRdB_after']

    def train(
            self: 'CorrectPhase', 
            mode: bool = True) -> 'CorrectPhase':
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        self.stft.train(mode=mode)
        self.istft.train(mode=mode)

        self.speech_est.train(mode=False)
        self.noise_est.train(mode=False)

        self.phase_corrector.train(mode=mode)

        return self

    def load_state_dict_from_pretrain_state_dict(
            self,
            pretrain_net_dict: Dict[str, Any]) -> None:

        #* The current model (self) and the state dict must be 
        #* on CPU when running this function!
        model_cls: ModuleInterface = pretrain_net_dict['model_cls']
        pretrain_net_configs = model_cls.ConstructorArgs(**pretrain_net_dict['configs'])
        pretrain_net_state_dict = pretrain_net_dict['state_dict']
        pretrain_net = model_cls(**pretrain_net_configs.model_dump())
        pretrain_net.load_state_dict(pretrain_net_state_dict)
        
        pretrain_net: EstimateWienerFilter

        speech_missing_keys, speech_unexpected_keys = self.speech_est.load_state_dict(
            pretrain_net.speech_est.state_dict(), strict=False)

        noise_missing_keys, noise_unexpected_keys = self.noise_est.load_state_dict(
            pretrain_net.noise_est.state_dict(), strict=False)

        print('Warning: Missing keys: ', list(speech_missing_keys) + list(noise_missing_keys))
        print('Warning: Unexpected keys: ', list(speech_unexpected_keys) + list(noise_unexpected_keys))

    def forward(
            self,
            x_bt: torch.Tensor,
            y_bt: torch.Tensor,
            epoch: int = -1) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        with torch.no_grad():
            # For magnitude enhancement
            Y_b2ft = self.stft(y_bt)
            Yp_b1ft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=True).add(self.eps)
            logYp_bft = Yp_b1ft.log().squeeze(dim=1)

            # Speech variance estimation
            _, logXpHat_bft, _, _, _ = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter (amplitude mask)
            HHat_bft = 1 / (1 + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            # Amplitude-enhanced speech
            XHat_b2ft = past_groundtruth(Y_b2ft, HHat_bft) * HHat_bft.unsqueeze(dim=1)
            xHat_bt = self.istft(XHat_b2ft)
            nSISDRdB_before = negative_sisdr(x_bt, xHat_bt)

        XHatHat_b2ft = self.phase_corrector(XHat_b2ft)
        xHatHat_bt: torch.Tensor = self.istft(XHatHat_b2ft)

        # Compute loss
        nSISDRdB = negative_sisdr(x_bt, xHatHat_bt)
        loss = nSISDRdB

        return loss, {
            'loss':  loss.item(),
            'SISDRdB_before': -nSISDRdB_before.item(),
            'SISDRdB_after': -nSISDRdB.item()}

    def validate(
            self,
            x_bt: torch.Tensor,
            y_bt: torch.Tensor,
            epoch: int = -1) -> Dict[str, Any]:
        sig_len = x_bt.size(-1)
        win_len = self.stft.win_len
        hop_len = self.stft.hop_len
        model_stride = self.speech_est.vqvae.encoders.total_stride
        latent_len = (sig_len + win_len - hop_len) // (hop_len * model_stride)
        input_len = (latent_len * model_stride + 1) * hop_len - win_len
        x_bt = x_bt[..., : input_len]
        y_bt = y_bt[..., : input_len]

        with torch.no_grad():
            # For magnitude enhancement
            Y_b2ft = self.stft(y_bt)
            Yp_b1ft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=True).add(self.eps)
            logYp_bft = Yp_b1ft.log().squeeze(dim=1)

            # Speech variance estimation
            _, logXpHat_bft, _, _, _ = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter (amplitude mask)
            HHat_bft = 1. / (
                1. + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            # Amplitude-enhanced speech
            XHat_b2ft = past_groundtruth(Y_b2ft, HHat_bft) * HHat_bft.unsqueeze(dim=1)
            xHat_bt = self.istft(XHat_b2ft)
            nSISDRdB_before = negative_sisdr(x_bt, xHat_bt)

            # Phase corrector
            XHatHat_b2ft = self.phase_corrector(XHat_b2ft)
            xHatHat_bt: torch.Tensor = self.istft(XHatHat_b2ft)

            # Compute loss
            nSISDRdB = negative_sisdr(x_bt, xHatHat_bt)
            loss = nSISDRdB

            return {
                'numerical': {
                    'loss': loss.item(),
                    'SISDRdB_before': -nSISDRdB_before.item(),
                    'SISDRdB_after': -nSISDRdB.item()},
                'waveform': {
                    'x_bt': x_bt.detach().cpu(),
                    'y_bt': y_bt.detach().cpu(),
                    'xHat_bt': xHat_bt.detach().cpu(),
                    'xHatHat_bt': xHatHat_bt.detach().cpu()}}

    def enhance(self, y_bt: torch.Tensor, use_pad: bool = False) -> Dict[str, Any]:
        sig_len = y_bt.size(-1)
        win_len = self.stft.win_len
        hop_len = self.stft.hop_len
        model_stride = self.speech_est.vqvae.encoders.total_stride

        look_ahead = (
            (self.speech_est.look_ahead + self.noise_est.look_ahead) 
            if self.look_ahead is None 
            else self.look_ahead)
        if use_pad:
            latent_len = (sig_len + win_len - hop_len - 1) // (hop_len * model_stride) + 1
            input_len = (latent_len * model_stride + look_ahead + 1) * hop_len - win_len
            y_bt = F.pad(y_bt, (0, input_len-sig_len))
        else:
            latent_len = (sig_len + win_len - hop_len) // (hop_len * model_stride)
            input_len = (latent_len * model_stride + 1) * hop_len - win_len
            y_bt = y_bt[..., : input_len]

        with torch.no_grad():
            # For magnitude enhancement
            Y_b2ft = self.stft(y_bt)
            Yp_b1ft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=True).add(self.eps)
            logYp_bft = Yp_b1ft.log().squeeze(dim=1)

            # Speech variance estimation
            _, logXpHat_bft, _, _, _ = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter (amplitude mask)
            HHat_bft = 1. / (
                1. + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            # Amplitude-enhanced speech
            XHat_b2ft = past_groundtruth(Y_b2ft, HHat_bft) * HHat_bft.unsqueeze(dim=1)

            # Phase corrector
            XHatHat_b2ft = self.phase_corrector(XHat_b2ft)
            xHatHat_bt: torch.Tensor = self.istft(XHatHat_b2ft)
            return xHatHat_bt


class FrameWiseCorrectPhaseWithIFDMask(nn.Module, ModuleInterface, ModelInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        stft_configs: STFTConfigs
        speech_estimator_configs: SpeechVarianceEstimator.ConstructorArgs
        noise_estimator_configs: NoiseVarianceEstimator.ConstructorArgs
        ifd_mask_estimator_configs: IFDMaskEstimator.ConstructorArgs
        wiener_type: Literal['original', 'irm'] = 'original'
        do_quantisation: bool = True
        look_ahead: Optional[int] = None

        name: Literal['FrameWiseCorrectPhaseWithIFDMask'] = 'FrameWiseCorrectPhaseWithIFDMask'

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self._configs: FrameWiseCorrectPhaseWithIFDMask.ConstructorArgs

        # Parameters
        self.wiener_type = self._configs.wiener_type
        self.do_quantisation = self._configs.do_quantisation
        self.look_ahead = self._configs.look_ahead

        if not self.do_quantisation:        # Do not initialise quantisation module
            self._configs.speech_estimator_configs.vqvae_configs.quantiser_configs = None

        if self.look_ahead is not None:     # Global look-ahead
            self._configs.speech_estimator_configs.look_ahead = 0
            self._configs.noise_estimator_configs.look_ahead = 0
            self._configs.ifd_mask_estimator_configs.look_ahead = 0

        # STFT
        self.stft = ConvSTFT(**self._configs.stft_configs.model_dump())
        self.istft = ConvISTFT(**self._configs.stft_configs.model_dump())

        # Speech variance estimator
        self.speech_est = SpeechVarianceEstimator(
            mode='denoise',
            **self._configs.speech_estimator_configs.model_dump())

        # Noise variance estimator
        self.noise_est = NoiseVarianceEstimator(
            **self._configs.noise_estimator_configs.model_dump())

        # Instantaneous frequency deviation mask estimation
        self.ifd_mask_estimator = IFDMaskEstimator(
            **self._configs.ifd_mask_estimator_configs.model_dump())

        # Avoid numerical issue
        self.eps: float = 1e-12
        self.log_eps: float = -12 * math.log(10)

    def get_training_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.ifd_mask_estimator.parameters()

    @property
    def output_logging_keys(self) -> List[str]:
        return ['loss', 'SISDRdB_before', 'SISDRdB_after']

    def train(
            self: 'FrameWiseCorrectPhaseWithIFDMask', 
            mode: bool = True) -> 'FrameWiseCorrectPhaseWithIFDMask':
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        self.stft.train(mode=mode)
        self.istft.train(mode=mode)

        self.speech_est.train(mode=False)
        self.noise_est.train(mode=False)

        self.ifd_mask_estimator.train(mode=mode)

        return self

    def load_state_dict_from_pretrain_state_dict(
            self,
            pretrain_net_dict: Dict[str, Any]) -> None:

        #* The current model (self) and the state dict must be 
        #* on CPU when running this function!
        model_cls: ModuleInterface = pretrain_net_dict['model_cls']
        pretrain_net_configs = model_cls.ConstructorArgs(**pretrain_net_dict['configs'])
        pretrain_net_state_dict = pretrain_net_dict['state_dict']
        pretrain_net = model_cls(**pretrain_net_configs.model_dump())
        pretrain_net.load_state_dict(pretrain_net_state_dict)
        
        pretrain_net: EstimateWienerFilter

        speech_missing_keys, speech_unexpected_keys = self.speech_est.load_state_dict(
            pretrain_net.speech_est.state_dict(), strict=False)

        noise_missing_keys, noise_unexpected_keys = self.noise_est.load_state_dict(
            pretrain_net.noise_est.state_dict(), strict=False)

        print('Warning: Missing keys: ', list(speech_missing_keys) + list(noise_missing_keys))
        print('Warning: Unexpected keys: ', list(speech_unexpected_keys) + list(noise_unexpected_keys))

    def forward(
            self,
            x_bt: torch.Tensor,
            y_bt: torch.Tensor,
            epoch: int = -1) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        with torch.no_grad():
            # For magnitude enhancement
            Y_b2ft = self.stft(y_bt)
            Yp_b1ft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=True).add(self.eps)
            logYp_bft = Yp_b1ft.log().squeeze(dim=1)

            # Speech variance estimation
            _, logXpHat_bft, _, _, _ = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter (amplitude mask)
            HHat_bft = 1 / (1 + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            # Amplitude-enhanced speech
            XHat_b2ft = past_groundtruth(Y_b2ft, HHat_bft) * HHat_bft.unsqueeze(dim=1)
            xHat_bt = self.istft(XHat_b2ft)
            nSISDRdB_before = negative_sisdr(x_bt, xHat_bt)

        XHatHat_b2ft, _, _ = self.ifd_mask_estimator(XHat_b2ft)
        xHatHat_bt: torch.Tensor = self.istft(XHatHat_b2ft)

        # Compute loss
        nSISDRdB = negative_sisdr(x_bt, xHatHat_bt)
        loss = nSISDRdB

        return loss, {
            'loss':  loss.item(),
            'SISDRdB_before': -nSISDRdB_before.item(),
            'SISDRdB_after': -nSISDRdB.item()}

    def validate(
            self,
            x_bt: torch.Tensor,
            y_bt: torch.Tensor,
            epoch: int = -1,
            IFD_scaling: bool = True,
            IFD_shifting: bool = True) -> Dict[str, Any]:
        sig_len = x_bt.size(-1)
        win_len = self.stft.win_len
        hop_len = self.stft.hop_len
        model_stride = self.speech_est.vqvae.encoders.total_stride
        latent_len = (sig_len + win_len - hop_len) // (hop_len * model_stride)
        input_len = (latent_len * model_stride + 1) * hop_len - win_len
        x_bt = x_bt[..., : input_len]
        y_bt = y_bt[..., : input_len]

        with torch.no_grad():
            # For magnitude enhancement
            Y_b2ft = self.stft(y_bt)
            Yp_b1ft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=True).add(self.eps)
            logYp_bft = Yp_b1ft.log().squeeze(dim=1)

            # Speech variance estimation
            _, logXpHat_bft, _, _, _ = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter (amplitude mask)
            HHat_bft = 1. / (
                1. + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            # Amplitude-enhanced speech
            XHat_b2ft = past_groundtruth(Y_b2ft, HHat_bft) * HHat_bft.unsqueeze(dim=1)
            xHat_bt: torch.Tensor = self.istft(XHat_b2ft)
            nSISDRdB_before = negative_sisdr(x_bt, xHat_bt)

            # Phase corrector
            XHatHat_b2ft, alphaHat_bft, zetaHat_b2ft = self.ifd_mask_estimator(
                XHat_b2ft, IFD_scaling, IFD_shifting)
            XHatHat_b2ft: torch.Tensor
            alphaHat_bft: torch.Tensor
            zetaHat_b2ft: torch.Tensor

            xHatHat_bt: torch.Tensor = self.istft(XHatHat_b2ft)

            # Compute loss
            nSISDRdB = negative_sisdr(x_bt, xHatHat_bt)
            loss = nSISDRdB

            return {
                'numerical': {
                    'loss': loss.item(),
                    'SISDRdB_before': -nSISDRdB_before.item(),
                    'SISDRdB_after': -nSISDRdB.item()},
                'waveform': {
                    'x_bt': x_bt.detach().cpu(),
                    'y_bt': y_bt.detach().cpu(),
                    'xHat_bt': xHat_bt.detach().cpu(),
                    'xHatHat_bt': xHatHat_bt.detach().cpu()},
                'others': {
                    'XHat_b2ft': XHat_b2ft.detach().cpu(),
                    'XHatHat_b2ft': XHatHat_b2ft.detach().cpu(),
                    'logXp_bft': c_op.c_square_mag(self.stft(x_bt), c_dim=1, keepdim=False).add(self.eps).log().detach().cpu(),
                    'logNp_bft': c_op.c_square_mag(self.stft(y_bt-x_bt), c_dim=1, keepdim=False).add(self.eps).log().detach().cpu(),
                    'logYp_bft': c_op.c_square_mag(self.stft(y_bt), c_dim=1, keepdim=False).add(self.eps).log().detach().cpu(),
                },
                'mask': {
                    'alphaHat_bft': alphaHat_bft.detach().cpu() / 2,                                                           # IFD scaling
                    'zetaHat_bft': (c_op.c_phs(zetaHat_b2ft, c_dim=1, keepdim=False).detach().cpu() + math.pi/2) / math.pi,    # IFD shifting
                }}

    def enhance(
            self,
            y_bt: torch.Tensor,
            use_pad: bool = False,
            return_spec: bool = False,
            IFD_scaling: bool = True,
            IFD_shifting: bool = True,
            use_GriffinLim: bool = False) -> Dict[str, Any]:
        sig_len = y_bt.size(-1)
        win_len = self.stft.win_len
        hop_len = self.stft.hop_len
        model_stride = self.speech_est.vqvae.encoders.total_stride

        look_ahead = (
            (self.speech_est.look_ahead + self.noise_est.look_ahead) 
            if self.look_ahead is None 
            else self.look_ahead)
        if use_pad:
            latent_len = (sig_len + win_len - hop_len - 1) // (hop_len * model_stride) + 1
            input_len = (latent_len * model_stride + look_ahead + 1) * hop_len - win_len
            y_bt = F.pad(y_bt, (0, input_len-sig_len))
        else:
            latent_len = (sig_len + win_len - hop_len) // (hop_len * model_stride)
            input_len = (latent_len * model_stride + 1) * hop_len - win_len
            y_bt = y_bt[..., : input_len]

        with torch.no_grad():
            # For magnitude enhancement
            Y_b2ft = self.stft(y_bt)
            Yp_b1ft = c_op.c_square_mag(Y_b2ft, c_dim=1, keepdim=True).add(self.eps)
            logYp_bft = Yp_b1ft.log().squeeze(dim=1)

            # Speech variance estimation
            _, logXpHat_bft, _, _, _ = self.speech_est(
                logYp_bft, do_quantisation=self.do_quantisation)
            logXpHat_bft: torch.Tensor

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.noise_est(
                past_groundtruth(logYp_bft, logXpHat_bft),
                logXpHat_bft.detach())

            # Global look-ahead
            if self.look_ahead is not None:
                logXpHat_bft = future_estimation(logXpHat_bft, look_ahead_width=self.look_ahead)
                logNpHat_bft = future_estimation(logNpHat_bft, look_ahead_width=self.look_ahead)

            # Wiener filter (amplitude mask)
            HHat_bft = 1. / (
                1. + (logNpHat_bft - past_groundtruth(logXpHat_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                HpHat_bft = HpHat_bft.sqrt()

            # Amplitude-enhanced speech
            XHat_b2ft = past_groundtruth(Y_b2ft, HHat_bft) * HHat_bft.unsqueeze(dim=1)

            # Phase corrector
            if IFD_scaling or IFD_shifting:
                XHatHat_b2ft, _, _ = self.ifd_mask_estimator(XHat_b2ft, IFD_scaling, IFD_shifting)
                XHatHat_b2ft: torch.Tensor
            elif use_GriffinLim: 
                n_iters = 32
                momentum = 0.99
                update_rate = momentum / (1 + momentum)
                
                AHatHat_b1ft = c_op.c_mag(XHat_b2ft, c_dim=1, keepdim=True)
                XHatHat_b2ft = XHat_b2ft 
                for _ in range(n_iters):
                    xHatHat_bt: torch.Tensor = self.istft(XHatHat_b2ft)
                    XHatHat_New_b2ft = self.stft(xHatHat_bt)

                    XHatHat_b2ft = XHatHat_New_b2ft - update_rate * XHatHat_b2ft
                    XHatHat_b2ft = XHatHat_b2ft / c_op.c_mag(XHatHat_b2ft, c_dim=1, keepdim=True).add(1e-6)
                    XHatHat_b2ft = XHatHat_b2ft * AHatHat_b1ft
            else:
                XHatHat_b2ft = XHat_b2ft

            if return_spec:
                return XHatHat_b2ft

            xHatHat_bt: torch.Tensor = self.istft(XHatHat_b2ft)
            return xHatHat_bt
