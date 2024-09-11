import numpy as np
import os
import torch
from scipy.io import wavfile as sio_wav
from typing import  Dict

from utils import matplotlib_template as mpl_template


class MediaDataLogger:
    @classmethod
    def plot_dict_spectrum(
            cls,
            dict_logSpec: Dict[str, torch.Tensor],
            out_dir: str,
            sample_idx: int) -> None:
        
        assert os.path.exists(out_dir), f'{out_dir} does not exist.'
        for spec_name, logSpec_bft in dict_logSpec.items():
            logSpec_ft = logSpec_bft[0, ...].detach().cpu().numpy()
            out_path = os.path.join(out_dir, f'sample_{sample_idx}_{spec_name}.png')

            logSpec_dB_ft = 10 * logSpec_ft / np.log(10)

            with mpl_template.default_subplots(save_path=out_path, show=False) as (fig, ax):
                im = mpl_template.imshow_log_spec(ax, logSpec_dB_ft)
                fig.colorbar(im)

    @classmethod
    def plot_dict_mask(
            cls,
            dict_mask: Dict[str, torch.Tensor],
            out_dir: str,
            sample_idx: int) -> None:
        
        assert os.path.exists(out_dir), f'{out_dir} does not exist.'
        for mask_name, mask_bkt in dict_mask.items():
            mask_kt = mask_bkt[0, ...].detach().cpu().numpy()
            out_path = os.path.join(out_dir, f'sample_{sample_idx}_{mask_name}.png')

            with mpl_template.default_subplots(save_path=out_path, show=False) as (fig, ax):
                im = mpl_template.imshow_binary_mask(ax, mask_kt, vmax=1, vmin=0)
                fig.colorbar(im)

    @classmethod
    def write_dict_wav(
            cls,
            dict_waveform: Dict[str, torch.Tensor],
            out_dir: str,
            sample_idx: int,
            fs: int = 16000) -> None:
        
        assert os.path.exists(out_dir), f'{out_dir} does not exist.'
        for wave_name, sig_bt in dict_waveform.items():
            sig_t = sig_bt[0, ...].detach().cpu().numpy()
            sigQ_t = np.asarray(sig_t * 32767, dtype=np.int16)
            out_path = os.path.join(out_dir, f'sample_{sample_idx}_{wave_name}.wav')

            sio_wav.write(filename=out_path, rate=fs, data=sigQ_t)
