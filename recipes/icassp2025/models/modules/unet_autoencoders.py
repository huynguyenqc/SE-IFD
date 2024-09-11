import numpy as np
import torch
from torch import nn
from typing import List, Optional, Tuple
from deep.modules.vq import EMACodebook
from .base_modules import Encoder, Decoder, Encoder2dFT, Decoder2dFT

from deep.base_module import ModuleInterface


class UNetEncoder(nn.Module, ModuleInterface):
    """ Stack of WaveNet-based encoders for U-Net encoder-decoder """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        encoder_configs: List[Encoder.ConstructorArgs]
        encoder2d_configs: Optional[Encoder2dFT.ConstructorArgs] = None
        use_batchnorm: bool = False
        embedding_dims: List[int]

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self._configs: UNetEncoder.ConstructorArgs

        for enc_cfg, dim in zip(self._configs.encoder_configs, self._configs.embedding_dims):
            assert enc_cfg.output_dim >= dim, \
                'Embeddding dimension must be smaller than output dimension of encoder!'

        self.embedding_dims: List[int] = self._configs.embedding_dims

        if self._configs.encoder2d_configs is not None:
            self.encoders_2d = Encoder2dFT(**self._configs.encoder2d_configs.model_dump())
        else:
            self.encoders_2d = None
        self.encoders_2d: Optional[Encoder2dFT]

        self.encoders = nn.ModuleList([
            Encoder(**enc_cfg.model_dump()) for enc_cfg in self._configs.encoder_configs])
        self.batchnorms = nn.ModuleList([
            (nn.BatchNorm1d(num_features=dim)
             if self._configs.use_batchnorm
             else nn.Identity()) 
            for dim in self.embedding_dims])
        self.total_stride: int = int(np.prod([
            mdl.resample_rate for mdl in self.encoders]))

    def forward(
            self, 
            x_bft: torch.Tensor, 
            list_c_bft: Optional[List[Optional[torch.Tensor]]] = None,
            c_bcft: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        # First, learn local-frequency information
        h_bct = x_bft
        if self.encoders_2d is not None:
            h_bcft = h_bct.unsqueeze(dim=1)     # 1 channel of 2D time-frequency representation
            h_bcft: torch.Tensor = self.encoders_2d(h_bcft, c_bcft)
            h_bct = h_bcft.flatten(start_dim=1, end_dim=2)

        list_z_bct = []
        if list_c_bft is None:
            list_c_bft = [None] * len(self.encoders)
        
        for enc_i, dim_i, bn_i, ci_bct in zip(self.encoders, self.embedding_dims, self.batchnorms, list_c_bft):
            h_bct = enc_i(h_bct, ci_bct)

            out_dim_i = h_bct.size(1)  # Output dimension of encoder
            e_bct, h_bct = torch.split(h_bct, split_size_or_sections=[dim_i, out_dim_i - dim_i], dim=1)
            list_z_bct.append(bn_i(e_bct))

        return list_z_bct


class UNetDecoder(nn.Module, ModuleInterface):
    """ Stack of WaveNet-based decoders for U-Net encoder-decoder """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        decoder_configs: List[Decoder.ConstructorArgs]
        decoder2d_configs: Optional[Decoder2dFT.ConstructorArgs] = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self._configs: UNetDecoder.ConstructorArgs

        self.decoders = nn.ModuleList([
            Decoder(**dec_cfg.model_dump()) for dec_cfg in self._configs.decoder_configs])

        if self._configs.decoder2d_configs is not None:
            self.decoders_2d = Decoder2dFT(**self._configs.decoder2d_configs.model_dump())
        else:
            self.decoders_2d = None
        self.decoders_2d: Optional[Decoder2dFT]

    def forward(
            self,
            list_h_bct: List[torch.Tensor],
            list_c_bct: Optional[List[Optional[torch.Tensor]]] = None,
            c_bcft: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if list_c_bct is None:
            list_c_bct = [None] * len(self.decoders)

        pair_h_bct = []
        for dec_i, hi_bct, ci_bct in zip(self.decoders, list_h_bct, list_c_bct):
            pair_h_bct.append(hi_bct)
            h_bct = dec_i(torch.cat(pair_h_bct, dim=1), ci_bct)
            pair_h_bct = [h_bct]

        h_bct: torch.Tensor = pair_h_bct[0]

        if self.decoders_2d is not None:
            h_bcft = h_bct.unsqueeze(dim=1)
            h_bcft: torch.Tensor = self.decoders_2d(h_bcft, c_bcft)
            h_bct = h_bcft.flatten(start_dim=1, end_dim=2)
        return h_bct


class UNetQuantiserEMA(nn.Module, ModuleInterface):
    """ Stack of vector quantisation modules """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        quantiser_configs: List[EMACodebook.ConstructorArgs]
        reservoir_downsampling_rates: Optional[List[int]] = None
        jitter_for_training: bool = False

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        assert isinstance(self._configs, self.ConstructorArgs)

        self.quantisers = nn.ModuleList([
            EMACodebook(**vq_cfg.model_dump())
            for vq_cfg in self._configs.quantiser_configs])
        self.reservoir_downsampling_rates: List[int] = (
            [1] * len(self.quantisers)
            if self._configs.reservoir_downsampling_rates is None
            else self._configs.reservoir_downsampling_rates)
        self.jitter_for_training: bool = self._configs.jitter_for_training

    def update_reservoir(self, list_z_bct: List[torch.Tensor]) -> None:
        for vq_i, r_i, zi_bct in zip(self.quantisers, self.reservoir_downsampling_rates, list_z_bct):
            vq_i.update_reservoir(zi_bct[..., ::r_i].transpose(-1, -2))

    def initialise_codebook_from_reservoir(self) -> None:
        for vq_i in self.quantisers:
            vq_i.initialise_codebook_from_reservoir()

    def set_codebook_ema_momentum(self, lr: Optional[float] = None) -> None:
        for vq_i in self.quantisers:
            vq_i.set_codebook_ema_momentum(lr)

    def vq_jitter(self, q__t: torch.Tensor) -> torch.Tensor:
        n_time_samples = q__t.size(-1)
        with torch.no_grad():
            p_replace = 0.12
            probabilities = q__t.new_tensor(data=[p_replace/2, 1 - p_replace, p_replace/2])
            p_t = torch.multinomial(probabilities, num_samples=n_time_samples, replacement=True) - 1
            p_t[..., 0].clamp_(0, 1)
            p_t[..., -1].clamp_(-1, 0)
            idx_t = torch.arange(n_time_samples, device=p_t.device, dtype=p_t.dtype) + p_t
            replaced__t = p_t.abs().float().view((1, ) * (q__t.dim() - 1) + (n_time_samples, ))
            qJittered__t = q__t.detach().index_select(dim=-1, index=idx_t)

        return q__t + (qJittered__t - q__t) * replaced__t
    
    def forward(
            self,
            list_z_bct: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Args:
            list_z_bct (List[torch.Tensor]): List of hidden variables
        Return:
            list_zq_bct (List[torch.Tensor]): List of quantised variables with gradients of list_z_bct
            list_q_bct (List[torch.Tensor]): List of quantised variables w.r.t. codebooks
            list_perplexity (List[float]): List of perplexity
        """
        list_perplexity = []
        list_q_bct = []
        list_zq_bct = []

        for vq_i, z_i_bct in zip(self.quantisers, list_z_bct):
            q_i_btc, p_i_btk = vq_i(z_i_bct.transpose(-1, -2))
            with torch.no_grad():
                perp_i = EMACodebook.perplexity(p_i_btk).item()

            q_i_bct = q_i_btc.transpose(-1, -2)

            if self.training and self.jitter_for_training:
                q_i_bct = self.vq_jitter(q_i_bct)

            zq_i_bct = z_i_bct + (q_i_bct - z_i_bct).detach()

            list_perplexity.append(perp_i)
            list_q_bct.append(q_i_bct)
            list_zq_bct.append(zq_i_bct)
        
        return list_zq_bct, list_q_bct, list_perplexity


class VQUNetBase(nn.Module, ModuleInterface):
    """ Hierarchical VQVAE """
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        encoder_configs: UNetEncoder.ConstructorArgs
        decoder_configs: UNetDecoder.ConstructorArgs
        quantiser_configs: Optional[UNetQuantiserEMA.ConstructorArgs] = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        assert isinstance(self._configs, self.ConstructorArgs)

        self.encoders = UNetEncoder(**self._configs.encoder_configs.model_dump())
        self.decoders = UNetDecoder(**self._configs.decoder_configs.model_dump())
        self.quantisers = (
            UNetQuantiserEMA(**self._configs.quantiser_configs.model_dump())
            if self._configs.quantiser_configs is not None
            else None)

        # Calculate receptive field
        receptive_field: int = 1
        for enc, dec in zip(self.encoders.encoders[::-1], self.decoders.decoders):
            enc: Encoder
            dec: Decoder
            receptive_field = enc.receptive_field + enc.resample_rate * (
                receptive_field - 1 + dec.receptive_field - 1)
        self.receptive_field: int = receptive_field

    def load_instance(self, ins: 'VQUNetBase') -> None:
        raise NotImplementedError('The method must be implemented in each subclass.')

    def encode_and_quantise(
            self, 
            x_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None
    ) -> Tuple[List[torch.Tensor], List[float]]:
        # Encode
        list_zx_bct = self.encoders(x_bct, list_cex_bct)

        # Quantise
        if self.quantisers is not None:
            _, list_qx_bct, list_perplx_x = self.quantisers(list_zx_bct) 
        else:
            list_qx_bct = list_zx_bct
            list_perplx_x = [-1] * len(list_zx_bct)

        return list_qx_bct, list_perplx_x

    def encode(
            self,
            x_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None
    ) -> List[torch.Tensor]:
        return self.encoders(x_bct, list_cex_bct)


class VQUNet(VQUNetBase):
    def train(self: 'VQUNet', mode: bool = True) -> 'VQUNet':
        if not isinstance(mode, bool):
            raise ValueError('training mode is expected to be boolean')
        self.training = mode

        self.encoders.train(mode=mode)
        if self.quantisers is not None:
            self.quantisers.train(mode=mode)
        self.decoders.train(mode=mode)

        return self

    def initialise_codebook_from_reservoir(self) -> None:
        if self.quantisers is not None:
            self.quantisers.initialise_codebook_from_reservoir()

    def set_codebook_ema_momentum(self, lr: Optional[float] = None) -> None:
        if self.quantisers is not None:
            self.quantisers.set_codebook_ema_momentum(lr)

    def forward(
            self,
            x_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
            do_quantisation: bool = False,
            do_reservoir_sampling: bool = False
    ) -> Tuple[
            torch.Tensor,
            List[torch.Tensor],
            Optional[List[torch.Tensor]],
            List[float]]:
        """
        Args:
            x_bct (torch.Tensor): Input.
            list_cex_bct (List[torch.Tensor]): Conditional variables for 
                encoders (same resolution with input).
            list_cdx_bct (List[torch.Tensor]): Conditional variables for 
                decoders (same resolution with input).
            do_quantisation (bool): Whether or not to perform quantisation.
            do_reservoir_sampling (bool): Perform reservoir sampling.
        Return:
            xHat_bct (torch.Tensor): Reconstructed input
            list_zx_bct (List[torch.Tensor]): Continuous hidden variables.
            list_qx_bct (List[torch.Tensor], optional): Quantised
                hidden variables, if quantisation is performed.
            list_perplexity (List[float]): List of perplexities.
        """
        # Encode
        list_zx_bct = self.encoders(x_bct, list_cex_bct)

        # Update reservoir sampling
        if self.quantisers is not None:
            if do_reservoir_sampling:
                self.quantisers.update_reservoir(list_z_bct=list_zx_bct)

            # Quantise
            if do_quantisation:
                list_zxq_bct, list_qx_bct, list_perplx_x = self.quantisers(list_zx_bct) 
            else:
                list_zxq_bct = list_zx_bct
                list_qx_bct = None
                list_perplx_x = [-1] * len(list_zx_bct)
        else:
            list_zxq_bct = list_zx_bct
            list_qx_bct = None
            list_perplx_x = [-1] * len(list_zx_bct)

        # Decode
        xHat_bct = self.decoders(list_zxq_bct[::-1], list_cdx_bct)

        return xHat_bct, list_zx_bct, list_qx_bct, list_perplx_x

    def load_instance(self, ins: 'VQUNetBase') -> None:
        self.load_state_dict(ins.state_dict())


class DenoiseVQUNet(VQUNetBase):
    def train(self: 'DenoiseVQUNet', mode: bool = True) -> 'DenoiseVQUNet':
        if not isinstance(mode, bool):
            raise ValueError('training mode is expected to be boolean')
        self.training = mode

        self.encoders.train(mode=mode)
        if self.quantisers is not None:
            self.quantisers.train(mode=False)   # Fixed codebook
        self.decoders.train(mode=mode)

        return self

    def forward(
            self,
            x_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
            do_quantisation: bool = False,
            do_reservoir_sampling: bool = False     # Ignore this flag
    ) -> Tuple[
            torch.Tensor,
            List[torch.Tensor],
            Optional[List[torch.Tensor]],
            List[float]]:
        """
        Args:
            x_bct (torch.Tensor): Input.
            list_cex_bct (List[torch.Tensor]): Conditional variables for
                encoders (same resolution with input).
            list_cdx_bct (List[torch.Tensor]): Conditional variables for
                decoders (same resolution with input).
        Return:
            xHat_bct (torch.Tensor): Reconstructed input
            list_zx_bct (List[torch.Tensor]): Continuous hidden variables.
            list_qx_bct (List[torch.Tensor], optional): Quantised
                hidden variables, if quantisation is performed.
            list_perplexity (List[float]): List of perplexities.
        """
        # Encode
        list_zx_bct = self.encoders(x_bct, list_cex_bct)

        # Quantisation
        if do_quantisation and self.quantisers is not None:
            list_zxq_bct, list_qx_bct, list_perplx_x = self.quantisers(list_zx_bct) 
            list_zxq_bct = [zxq_i_bct.detach() for zxq_i_bct in list_zxq_bct]
        else:
            list_zxq_bct = list_zx_bct
            list_qx_bct = None
            list_perplx_x = [-1] * len(list_zx_bct)

        # Decode
        xHat_bct = self.decoders(list_zxq_bct[::-1], list_cdx_bct)

        return xHat_bct, list_zx_bct, list_qx_bct, list_perplx_x

    def load_instance(self, ins: 'VQUNetBase') -> None:
        # Leave encoder randomly initialised
        if self.quantisers is not None and ins.quantisers is not None:
            self.quantisers.load_state_dict(ins.quantisers.state_dict())
        self.decoders.load_state_dict(ins.decoders.state_dict())
