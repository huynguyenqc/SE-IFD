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


def si_sdr_dB(x_t: np.ndarray, y_t: np.ndarray) -> float:
    eps = 1e-12
    alpha = (y_t * x_t).mean() / (np.square(x_t).mean() + eps)
    x_t = alpha * x_t
    sisdr_value = float(np.square(x_t).mean() / (np.square(y_t - x_t).mean() + eps))
    sisdr_dB_value = 10 * math.log10(sisdr_value + eps)
    return sisdr_dB_value


def inference_worker(
        rank: int,
        audio_dir: str,
        configs_path: str,
        weights_path: str,
        noisy_dataset_list: List[Tuple[str, str]],
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
        clean_file, noisy_file = noisy_dataset_list[i]
        x_t = data_utils.load_wav(clean_file, sr=ENHANCE_SAMPLING_RATE)
        y_t = data_utils.load_wav(noisy_file, sr=ENHANCE_SAMPLING_RATE)

        assert len(x_t) == len(y_t)

        y_t, _, g = data_utils.tailor_dB_FS(y_t, target_dB_FS=-27.5, eps=1e-12)
        x_t *= g

        y_bt = torch.from_numpy(y_t).unsqueeze_(dim=0).cuda()
        xHat_bt = model.enhance(y_bt=y_bt, **kwargs)
        xHat_t = xHat_bt.detach().cpu().squeeze_(dim=0).numpy()

        target_len = min(len(x_t), len(xHat_t), len(y_t)) - 3000
        x_t = x_t[: target_len]
        y_t = y_t[: target_len]
        xHat_t = xHat_t[: target_len]

        metrics_dict = {
            'Index': index_list[i],
            'Clean': clean_file.decode(),
            'Noisy': noisy_file.decode(),
            'Enhanced PESQ-WB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=xHat_t, mode='wb'),
            'Enhanced PESQ-NB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=xHat_t, mode='nb'),
            'Enhanced STOI': pystoi.stoi(x=x_t, y=xHat_t, fs_sig=ENHANCE_SAMPLING_RATE, extended=False),
            'Enhanced SI-SDR': si_sdr_dB(x_t=x_t, y_t=xHat_t),
            'Noisy PESQ-WB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=y_t, mode='wb'),
            'Noisy PESQ-NB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=y_t, mode='nb'),
            'Noisy STOI': pystoi.stoi(x=x_t, y=y_t, fs_sig=ENHANCE_SAMPLING_RATE, extended=False),
            'Noisy SI-SDR': si_sdr_dB(x_t=x_t, y_t=y_t)}

        out_metrics.append(metrics_dict)

        stacked_tc = np.stack((x_t, y_t, xHat_t), axis=-1)
        sio_wav.write(
            filename=os.path.join(audio_dir, f'sample_{index_list[i]}.wav'),
            rate=ENHANCE_SAMPLING_RATE,
            data=(stacked_tc * 32768).astype(np.int16))
    return out_metrics


def dict_keep_keys(in_dict: Dict[str, Any], key_list: List[str]) -> Dict[str, Any]:
    key_set = set(key_list)
    return {
        k: v for k, v in in_dict.items() if k in set(key_set)
    }


def main_evaluate(
        configs_path: str,
        weights_path: str,
        wav_list: str,
        out_dir: str = '',
        n_workers: int = 8,
        **kwargs) -> Dict[str, float]:

    eval_dataset = data_utils.PreSyntheticNoisyDataset(
        clean_noisy_path=wav_list,
        clean_noisy_limit=None, clean_noisy_offset=None, sr=ENHANCE_SAMPLING_RATE,
        sub_sample_sec=None, target_dB_FS=None, target_dB_FS_floating_value=None,
        f0_data_path=None)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    audio_dir = os.path.join(out_dir, 'audios')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    out_path = os.path.join(out_dir, 'results.csv')
    avg_logs = AverageLogs()

    with open(out_path, 'w') as f_out:
        csv_dict_writer = csv.DictWriter(
            f_out, 
            fieldnames=[
                'Index', 'Clean', 'Noisy',
                'Noisy PESQ-WB', 'Noisy PESQ-NB', 'Noisy STOI', 'Noisy SI-SDR',
                'Enhanced PESQ-WB', 'Enhanced PESQ-NB', 'Enhanced STOI', 'Enhanced SI-SDR'],
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
                    eval_dataset.noisy_dataset_list[rank::n_workers],
                    list(range(rank+1, len(eval_dataset)+1, n_workers)),
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
                    for out_item in out_metrics:
                        avg_logs.update(
                            dict_keep_keys(out_item, [
                                'Noisy PESQ-WB', 'Noisy PESQ-NB', 'Noisy STOI', 'Noisy SI-SDR',
                                'Enhanced PESQ-WB', 'Enhanced PESQ-NB', 'Enhanced STOI', 
                                'Enhanced SI-SDR']))
                    
    print(json.dumps(avg_logs.get_logs(), indent=4))
    return avg_logs.get_logs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-list', type=str, required=True)
    parser.add_argument('--model-dir-path', type=str, required=False, default=None)
    parser.add_argument('--configs-path', type=str, required=False, default=None)
    parser.add_argument('--weights-path', type=str, required=False, default=None)
    parser.add_argument('--out-dir', type=str, required=False, default=None)
    parser.add_argument('--n-workers', type=int, required=False, default=8)

    known_args, unknown_flag_args = parser.parse_known_args()

    if known_args.model_dir_path is None:
        assert known_args.configs_path is not None and known_args.weights_path is not None, \
            'When `--model-dir-path` is not set, `--configs-path`, `--weights-path`, and `--out-dir` are required!'
        parsed_args = {
            'wav_list': known_args.wav_list,
            'configs_path': known_args.configs_path,
            'weights_path': known_args.weights_path,
            'out_dir': known_args.out_dir,
            'n_workers': known_args.n_workers}
    else:
        parsed_args = {
            'wav_list': known_args.wav_list,
            'configs_path': os.path.join(known_args.model_dir_path, 'configs.yml'),
            'weights_path': os.path.join(known_args.model_dir_path, 'checkpoints/epoch_final.pt'),
            'out_dir': os.path.join(known_args.model_dir_path, 'test_results'),
            'n_workers': known_args.n_workers}

    for flag_arg in unknown_flag_args:
        assert flag_arg.startswith('--'), 'Argument `{}` does not follow kwargs patterns!'.format(flag_arg)
        parsed_args[flag_arg[2:].replace('-', '_')] = True

    main_evaluate(**parsed_args)

if __name__ == '__main__':
    main()
