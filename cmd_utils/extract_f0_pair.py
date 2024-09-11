import argparse
import librosa
import numpy as np
import os
import soundfile as sf
from pathlib import Path


def load_wav(file: str, sr: int = 16000) -> np.ndarray:
    s_t, fs_s = sf.read(
        file, dtype=np.float32, always_2d=True)
    # s_t, fs_s = librosa.load(file, mono=True)
    assert fs_s == sr, \
        (f'The sampling rate of {file} should be {sr} '
         f'Hz instead of {fs_s} Hz!')
    return s_t[:, 0]


parser = argparse.ArgumentParser()
parser.add_argument('--in-dir', type=str, required=True)
parser.add_argument('--ext', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)

parser.add_argument('--sr', type=float, required=True)
parser.add_argument('--frame-sec', type=float, required=True)
parser.add_argument('--win-sec', type=float, required=True)
parser.add_argument('--hop-sec', type=float, required=True)

args = parser.parse_args()

file_list = sorted([
    str(f) 
    for f in Path(
        args.in_dir).rglob('*.{}'.format(args.ext))])

frame_len = int(args.sr * args.frame_sec)
win_len = int(args.sr * args.win_sec)
hop_len = int(args.sr * args.hop_sec)
in_dir = args.in_dir
sr = args.sr

print('Frame len: {} | Win len: {} | Hop len: {}'.format(
    frame_len, win_len, hop_len))

pad_len = win_len - hop_len

assert len(os.listdir(args.out_dir)) == 0

def process_file(__fn: str):
    print(f'Start processing file {__fn}...')
    abs_path = __fn 
    rel_path = os.path.relpath(abs_path, in_dir)
    # File name without extension
    file_name = os.path.splitext(os.path.basename(rel_path))[0]
    dir_path = os.path.dirname(rel_path)

    out_dir = os.path.join(args.out_dir, dir_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, f'{file_name}.npy')

    x_t = load_wav(__fn, sr=sr)
    F0_t, _, _ = librosa.pyin(
        np.pad(x_t, (pad_len, pad_len)),
        fmin=60., fmax=440., sr=sr,
        frame_length=frame_len,
        win_length=win_len,
        hop_length=hop_len,
        center=False)
    np.save(out_path, F0_t)
    print(f'Finish processing file {__fn}, writing output to {out_path}...')

for fn in file_list:
    process_file(fn)