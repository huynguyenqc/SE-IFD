import argparse
import numpy as np
import soundfile as sf
from typing import Tuple


def load_wav(file: str, sr: int = 16000) -> np.ndarray:
    s_t, fs_s = sf.read(
        file, dtype=np.float32, always_2d=True)
    # s_t, fs_s = librosa.load(file, mono=True)
    assert fs_s == sr, \
        (f'The sampling rate of {file} should be {sr} '
         f'Hz instead of {fs_s} Hz!')
    return s_t[:, 0]


def write_wav(x_t: np.ndarray, file: str, sr: int = 16000) -> None:
    sf.write(file, data=x_t, samplerate=sr,
             subtype='PCM_16', format='WAV')


def ceil_positive_division(a: int, b: int) -> int:
    return ((a - 1) // b) + 1


def noise_wrt_clean(
        clean_y: np.ndarray,
        noise_y: np.ndarray) -> np.ndarray:

    clean_len = len(clean_y)
    noise_len = len(noise_y)

    ext_noise_y = np.tile(
        noise_y,
        reps=ceil_positive_division(clean_len, noise_len))

    start_idx = np.random.randint(0, len(ext_noise_y) - clean_len + 1)
    ext_noise_y = ext_noise_y[start_idx: start_idx + clean_len]

    return ext_noise_y


def snr_mix(
        clean_y: np.ndarray,
        noise_y: np.ndarray,
        snr_db: float,
        eps: float = 1e-12
) -> np.ndarray:
    clean_rms = np.sqrt((clean_y ** 2).mean())
    noise_rms = np.sqrt((noise_y ** 2).mean())

    snr = 10 ** (snr_db / 20)
    gain = clean_rms / (noise_rms * snr + eps)
    noisy_y = clean_y + gain * noise_y

    if np.any(np.abs(noisy_y) > 0.999):
        scale_gain = np.amax(np.abs(noisy_y) / (0.999 - eps))
        noisy_y *= scale_gain

    return noisy_y


def mix_wav(
        clean_file: str,
        noise_file: str,
        snr_db: float,
        out_noisy_file: str) -> None:
    
    x_t = load_wav(clean_file, sr=16000)
    n_t = load_wav(noise_file, sr=16000)
    n_t = noise_wrt_clean(x_t, n_t)
    assert len(n_t) == len(x_t), \
        'Speech and noise must have equal length now!'

    y_t = snr_mix(x_t, n_t, snr_db)
    write_wav(y_t, out_noisy_file, sr=16000)



parser = argparse.ArgumentParser()
parser.add_argument('--speech-file', type=str, required=True)
parser.add_argument('--noise-file', type=str, required=True)
parser.add_argument('--snr-db', type=float, required=True)
parser.add_argument('--out-file', type=str, required=True)

args = parser.parse_args()
mix_wav(args.speech_file, args.noise_file, args.snr_db, args.out_file)
