import argparse
import concurrent.futures
import csv
import json
import librosa
import math
import numpy as np
import os
import pesq
import pystoi
import torch
import tqdm
import yaml
from scipy.io import wavfile as sio_wav
from typing import Type, List, Tuple, Dict, Any

from . import (
    models as recipe_models, 
    data_utils)
from deep.utils import AverageLogs


ENHANCE_SAMPLING_RATE = 16000


def inference_worker(
        rank: int,
        audio_dir: str,
        configs_path: str,
        weights_path: str,
        noisy_dataset_list: List[str],
        index_list: List[int],
        **kwargs) -> List[Dict[str, Any]]:

    with open(configs_path, 'r') as f_config:
        json_configs = yaml.safe_load(f_config)
        json_model_configs = json_configs['model']

    model = recipe_models.get_enhancement_model(json_model_configs['name'])(
        **json_model_configs['configs'])
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)

    _ = model.cuda()
    _ = model.eval()

    model: recipe_models.model_2024.FrameWiseCorrectPhaseWithIFDMask

    loop_generator = (
        tqdm.tqdm(range(len(noisy_dataset_list)), position=rank) 
        if rank == 0 else 
        range(len(noisy_dataset_list)))

    out_metrics = []
    for i in loop_generator:
        noisy_file = noisy_dataset_list[i]
        y_t = data_utils.load_wav(noisy_file, sr=ENHANCE_SAMPLING_RATE)

        y_t, _, _ = data_utils.tailor_dB_FS(y_t, target_dB_FS=-27.5, eps=1e-12)

        y_bt = torch.from_numpy(y_t).unsqueeze_(dim=0).cuda()
        xHat_bt = model.enhance(y_bt=y_bt, **kwargs)
        xHat_t = xHat_bt.detach().cpu().squeeze_(dim=0).numpy()

        metrics_dict = {
            'Index': index_list[i],
            'Noisy': noisy_file.decode()}
        out_metrics.append(metrics_dict)

        sio_wav.write(
            filename=os.path.join(audio_dir, f'sample_{index_list[i]}.wav'),
            rate=ENHANCE_SAMPLING_RATE,
            data=(xHat_t * 32768).astype(np.int16))
    return out_metrics


def dict_keep_keys(in_dict: Dict[str, Any], key_list: List[str]) -> Dict[str, Any]:
    key_set = set(key_list)
    return {
        k: v for k, v in in_dict.items() if k in set(key_set)
    }


def main_model_infer(
        configs_path: str,
        weights_path: str,
        wav_list: str,
        out_dir: str = '',
        n_workers: int = 8,
        **kwargs) -> Dict[str, float]:

    with open(wav_list, 'r', encoding='utf-8') as f_filelist:
        noisy_dataset_list = [fp.strip().encode('utf-8') for fp in f_filelist.readlines()]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    audio_dir = os.path.join(out_dir, 'audios')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    out_path = os.path.join(out_dir, 'results.csv')

    with open(out_path, 'w') as f_out:
        csv_dict_writer = csv.DictWriter(
            f_out, 
            fieldnames=['Index', 'Noisy'],
            delimiter=',')
        csv_dict_writer.writeheader()

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures_to_rank = {
                executor.submit(
                    inference_worker,
                    rank,
                    audio_dir,
                    configs_path,
                    weights_path,
                    noisy_dataset_list[rank::n_workers],
                    list(range(rank+1, len(noisy_dataset_list)+1, n_workers)),
                    **kwargs): rank
                for rank in range(n_workers)}
            for future in concurrent.futures.as_completed(futures_to_rank):
                rank = futures_to_rank[future]
                try:
                    out_metrics = future.result()
                except Exception as exc:
                    print('Process rank {} generated an exception: {}.'.format(rank, str(exc)))
                else:
                    print('Process rank {} finished without exception.'.format(rank))
                    csv_dict_writer.writerows(out_metrics)

